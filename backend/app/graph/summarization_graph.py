"""
The summarization workflow as a LangGraph state graph.

    START
      │
    preprocess ── validate, clean, split into sentences
      │
      ├─ tfidf / textrank ─► extractive_summarize ───────────────────────────┐
      ├─ hybrid ─► select_key_sentences (TextRank) ─┐                         │
      └─ bart ──────────────────────────────────────┤                         │
                                                    ▼                         │
                                              check_length                    │
                          fits one pass ┌───────────┴─────────┐ too long      │
                                        ▼                     ▼               │
                         abstractive_single_pass        chunk_document ◄──┐   │
                                        │                     ▼           │   │
                                        │              summarize_chunks   │another
                                        │                     ▼           │round
                                        │              combine_summaries ─┘   │
                                        │        fits ┌───────┴───── done ────┤
                                        │             ▼                       │
                                        │     final_summarization             │
                                        ▼             ▼                       ▼
                                      postprocess ◄───────────────────────────┘
                                          │
                                       evaluate
                                          │
                                         END

What LangGraph contributes
--------------------------
* **Explicit control flow.** The routing decisions (which method, does the
  text fit the model, is another round needed) are visible edges of a graph
  rather than nested if/else and while-loops spread across functions.
  ``draw_mermaid()`` renders the real graph for the documentation.
* **Typed shared state** (``SummarizationState``) that records everything the
  run produced: sentences, selection, chunks, intermediate summaries,
  per-node timings and the path taken. It is useful for the UI and for
  explaining a result.
* **Loops with exit conditions** (the chunk -> summarize -> combine round),
  which is awkward to express as a linear pipeline.

What it deliberately does *not* do: nodes contain no algorithms. Each node
calls the same functions the summarizers use directly (``app.summarizers``,
``app.preprocessing``), so the graph is only orchestration and the
algorithms remain independently testable.
"""

import time
from collections.abc import Callable
from functools import wraps

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.errors import IntelliSumError
from app.evaluation.metrics import summary_statistics
from app.graph.state import SummarizationState
from app.preprocessing.cleaner import count_words
from app.preprocessing.langchain_splitter import make_chunker
from app.preprocessing.pipeline import preprocess, preprocess_documents
from app.preprocessing.sentence_splitter import split_sentences
from app.summarizers.bart import BARTSummarizer, check_abstractive_input_size
from app.summarizers.base import SummaryMethod, length_ratio
from app.summarizers.generation import generate_summary, tidy_generated_text
from app.summarizers.hybrid import HybridSummarizer
from app.summarizers.long_document import (
    can_fuse,
    chunk_limit,
    final_pass,
    next_step,
    round_target,
    summarize_chunks,
)
from app.summarizers.textrank import TextRankSummarizer
from app.summarizers.tfidf import TFIDFSummarizer

Node = Callable[[SummarizationState, RunnableConfig], dict]


def _progress(config: RunnableConfig):
    """The optional on_progress(stage, done, total) callback passed by the caller."""
    return (config or {}).get("configurable", {}).get("on_progress")


def node(name: str):
    """
    Wrap a node function to (1) record the path taken and the time spent,
    and (2) turn a user-facing ``IntelliSumError`` into an ``error`` entry
    in the state, after which every router sends the run to END.
    Unexpected exceptions (bugs) are not caught: they propagate to the API's
    generic error handler.
    """

    def decorator(fn: Node) -> Node:
        @wraps(fn)
        def wrapped(state: SummarizationState, config: RunnableConfig) -> dict:
            start = time.perf_counter()
            try:
                update = fn(state, config) or {}
            except IntelliSumError as exc:
                update = {"error": exc, "errors": [{"node": name, "type": type(exc).__name__, "message": exc.message}]}
            update["path"] = [name]
            update["timings"] = {name: time.perf_counter() - start}
            return update

        return wrapped

    return decorator


def _chunk_pages(chunks, work_pages) -> list[list[int]] | None:
    """Source pages covered by each chunk (provenance), when pages are known."""
    if not work_pages or any(p is None for p in work_pages):
        return None
    if any(not chunk.source_indices for chunk in chunks):
        return None  # e.g. recursive chunks do not align with sentences
    return [sorted({work_pages[i] for i in chunk.source_indices}) for chunk in chunks]


def _unless_error(next_node: str):
    """Edge that continues to ``next_node`` unless a node recorded an error."""
    return lambda state: END if state.get("error") else next_node


def build_summarization_graph(abstractive: BARTSummarizer | None = None):
    """
    Compile the workflow. ``abstractive`` can be injected (e.g. a different
    seq2seq model, or a lightweight stand-in in tests); by default the shared,
    lazily loaded BART model is used. Extractive runs never load it.
    """
    abstractive = abstractive or BARTSummarizer()
    hybrid = HybridSummarizer(abstractive=abstractive)
    extractive = {SummaryMethod.TFIDF: TFIDFSummarizer(), SummaryMethod.TEXTRANK: TextRankSummarizer()}

    def model():
        return abstractive.model  # loaded on first use

    # ---------------------------------------------------------------- nodes

    @node("preprocess")
    def preprocess_node(state, config):
        documents = state.get("documents")
        pre = preprocess_documents(documents) if documents else preprocess(state["original_text"])
        meta = {"num_sentences": pre.sentence_count}
        if documents:
            meta.update(source=documents[0].metadata.get("source"), num_documents=len(documents))
        return {
            "cleaned_text": pre.cleaned_text,
            "sentences": pre.sentences,
            "sentence_pages": pre.sentence_pages,
            "original_word_count": pre.word_count,
            "metadata": meta,
        }

    @node("extractive_summarize")
    def extractive_node(state, config):
        result = extractive[SummaryMethod(state["method"])].summarize(state["sentences"], state["summary_length"])
        return {
            "final_summary": result.summary,
            "selected_sentences": result.selected_indices,
            "sentence_scores": result.sentence_scores,
            "strategy": "extractive",
            "metadata": result.metadata,
        }

    @node("select_key_sentences")
    def select_node(state, config):
        sentences = state["sentences"]
        target, budget, cap = hybrid.plan(sentences, state["summary_length"])
        progress = _progress(config)
        if progress:
            progress("selecting key sentences (TextRank)", 0, 1)
        start = time.perf_counter()
        selected, scores, textrank_meta = hybrid.select_sentences(sentences, budget, cap)
        if progress:
            progress("selecting key sentences (TextRank)", 1, 1)
        pages = state.get("sentence_pages") or [None] * len(sentences)
        return {
            "selected_sentences": selected,
            "sentence_scores": [round(float(s), 6) for s in scores],
            "work_sentences": [sentences[i] for i in selected],
            "work_pages": [pages[i] for i in selected],
            "target_words": target,
            "metadata": {
                "num_selected": len(selected),
                "extractive_stage": hybrid.describe_selection(
                    sentences, selected, budget, cap, time.perf_counter() - start, textrank_meta
                ),
            },
        }

    @node("check_length")
    def check_length_node(state, config):
        work = state.get("work_sentences") or state["sentences"]
        work_pages = state.get("work_pages") if state.get("work_sentences") else state.get("sentence_pages")
        words = sum(count_words(s) for s in work)
        check_abstractive_input_size(words, state["method"])
        # BART sizes its summary from the original document; Hybrid already
        # set the target from the original in select_key_sentences.
        target = state.get("target_words") or words * length_ratio(state["summary_length"])
        tokens = model().count_tokens(" ".join(work))
        return {
            "work_sentences": work,
            "work_pages": work_pages,
            "target_words": target,
            "input_tokens": tokens,
            "fuse": can_fuse(target),
            "chunk_round": 0,
            "metadata": {
                "model": model().spec.hf_id,
                "device": model().device,
                "input_tokens": tokens,
                "max_input_tokens": model().max_input_tokens,
                "target_words": round(target),
            },
        }

    @node("abstractive_single_pass")
    def single_pass_node(state, config):
        text = " ".join(state["work_sentences"])
        summary, trimmed, (low, high) = generate_summary(model(), text, state["target_words"])
        return {
            "final_summary": summary,
            "strategy": "single_pass",
            "metadata": {
                "chunks": 1,
                "min_summary_tokens": low,
                "max_summary_tokens": high,
                "trimmed_incomplete_sentence": trimmed,
            },
        }

    @node("chunk_document")
    def chunk_node(state, config):
        work = state["work_sentences"]
        words = sum(count_words(s) for s in work)

        def count(text: str) -> int:
            return model().count_tokens(text, special_tokens=False)

        # Chunking strategy = a LangChain TextSplitter (sentence-aware by default).
        chunker = make_chunker(count, chunk_limit(model()))
        return {
            "chunk_round": state["chunk_round"] + 1,
            "chunks": chunker(work),
            "round_target_words": round_target(model(), state["target_words"], state["fuse"], words),
        }

    @node("summarize_chunks")
    def summarize_chunks_node(state, config):
        stage = f"level {state['chunk_round']}: summarizing chunks"
        summaries = summarize_chunks(model(), state["chunks"], state["round_target_words"], _progress(config), stage)
        return {"intermediate_summaries": summaries}

    @node("combine_summaries")
    def combine_node(state, config):
        summaries = state["intermediate_summaries"]
        combined = " ".join(summaries)
        step = next_step(model(), combined, state["target_words"], state["fuse"], state["chunk_round"])
        chunks = state["chunks"]
        level = {
            "level": state["chunk_round"],
            "input_words": sum(count_words(s) for s in state["work_sentences"]),
            "target_words": round(state["round_target_words"]),
            "output_words": count_words(combined),
            "num_chunks": len(chunks),
            "chunk_token_counts": [c.token_count for c in chunks],
            "contains_split_sentence": any(c.contains_split_sentence for c in chunks),
            "chunk_pages": _chunk_pages(chunks, state.get("work_pages")),
            "summaries": summaries,
        }
        update = {"combined_text": combined, "next_step": step, "reduction_levels": [level]}
        if step == "another_round":
            update["work_sentences"] = split_sentences(combined, min_words=1)
            update["work_pages"] = None  # summaries of summaries have no single source page
        elif step in ("done", "max_levels"):
            update["final_summary"] = combined
            update["strategy"] = "concatenated" if step == "done" else "max_levels_reached"
        if step == "max_levels":
            update["warnings"] = ["The document needed more summarization rounds than allowed; the result may be long."]
        return update

    @node("final_summarization")
    def final_node(state, config):
        summary, trimmed = final_pass(model(), state["combined_text"], state["target_words"], _progress(config))
        return {"final_summary": summary, "strategy": "fused", "metadata": {"trimmed_incomplete_sentence": trimmed}}

    @node("postprocess")
    def postprocess_node(state, config):
        summary = state.get("final_summary", "")
        if state["strategy"] == "extractive":
            # Copied source sentences: only normalise whitespace. (Trimming an
            # "incomplete" last sentence is only right for generated text.)
            summary = " ".join(summary.split())
        else:
            summary, _ = tidy_generated_text(summary)
        meta = {"strategy": state["strategy"]}
        levels = state.get("reduction_levels") or []
        if levels:
            meta.update(chunks=levels[0]["num_chunks"], reduction_rounds=len(levels))
        return {"final_summary": summary, "metadata": meta}

    @node("evaluate")
    def evaluate_node(state, config):
        elapsed = time.perf_counter() - state["started_at"]
        selected = state.get("selected_sentences")
        metrics = summary_statistics(
            state["original_word_count"],
            state["final_summary"],
            len(state["sentences"]),
            len(selected) if selected is not None else None,
            elapsed,
        )
        return {"metrics": metrics, "processing_time": round(elapsed, 3)}

    # ---------------------------------------------------------------- routers

    def route_method(state) -> str:
        if state.get("error"):
            return END
        method = SummaryMethod(state["method"])
        if method.is_extractive:
            return "extractive_summarize"
        return "select_key_sentences" if method is SummaryMethod.HYBRID else "check_length"

    def route_length(state) -> str:
        if state.get("error"):
            return END
        fits = state["input_tokens"] <= model().max_input_tokens
        return "abstractive_single_pass" if fits else "chunk_document"

    def route_after_combine(state) -> str:
        if state.get("error"):
            return END
        return {
            "final_pass": "final_summarization",
            "another_round": "chunk_document",
        }.get(state["next_step"], "postprocess")

    # ---------------------------------------------------------------- wiring

    graph = StateGraph(SummarizationState)
    for name, fn in [
        ("preprocess", preprocess_node),
        ("extractive_summarize", extractive_node),
        ("select_key_sentences", select_node),
        ("check_length", check_length_node),
        ("abstractive_single_pass", single_pass_node),
        ("chunk_document", chunk_node),
        ("summarize_chunks", summarize_chunks_node),
        ("combine_summaries", combine_node),
        ("final_summarization", final_node),
        ("postprocess", postprocess_node),
        ("evaluate", evaluate_node),
    ]:
        graph.add_node(name, fn)

    graph.add_edge(START, "preprocess")
    graph.add_conditional_edges(
        "preprocess", route_method, ["extractive_summarize", "select_key_sentences", "check_length", END]
    )
    graph.add_conditional_edges("extractive_summarize", _unless_error("postprocess"), ["postprocess", END])
    graph.add_conditional_edges("select_key_sentences", _unless_error("check_length"), ["check_length", END])
    graph.add_conditional_edges("check_length", route_length, ["abstractive_single_pass", "chunk_document", END])
    graph.add_conditional_edges("abstractive_single_pass", _unless_error("postprocess"), ["postprocess", END])
    graph.add_conditional_edges("chunk_document", _unless_error("summarize_chunks"), ["summarize_chunks", END])
    graph.add_conditional_edges("summarize_chunks", _unless_error("combine_summaries"), ["combine_summaries", END])
    graph.add_conditional_edges(
        "combine_summaries", route_after_combine, ["final_summarization", "chunk_document", "postprocess", END]
    )
    graph.add_conditional_edges("final_summarization", _unless_error("postprocess"), ["postprocess", END])
    graph.add_edge("postprocess", "evaluate")
    graph.add_edge("evaluate", END)
    return graph.compile()
