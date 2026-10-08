"""
The state object passed between the nodes of the summarization graph.

LangGraph works like a pipeline over a shared, typed dictionary: every node
receives the current state and returns only the keys it changes; LangGraph
merges those updates into the state. For most keys the new value simply
replaces the old one. Keys annotated with a *reducer* are combined instead,
e.g. ``warnings`` lists from different nodes are concatenated rather than
overwritten.
"""

import operator
from typing import Annotated, Any, TypedDict

from app.preprocessing.chunker import Chunk


def merge_dicts(old: dict | None, new: dict | None) -> dict:
    """Reducer: later nodes add keys to a dict instead of replacing it."""
    return {**(old or {}), **(new or {})}


def add_timings(old: dict | None, new: dict | None) -> dict:
    """Reducer: seconds per node, summed when a node runs more than once (loops)."""
    result = dict(old or {})
    for node, seconds in (new or {}).items():
        result[node] = round(result.get(node, 0.0) + seconds, 4)
    return result


class SummarizationState(TypedDict, total=False):
    # --- Request ---------------------------------------------------------------
    original_text: str  # plain-text input, or ...
    documents: list[Any]  # ... SourceDocuments, e.g. one LangChain Document per PDF page
    method: str  # "tfidf" | "textrank" | "bart" | "hybrid"
    summary_length: str  # "short" | "medium" | "long"
    reference_summary: str | None  # enables ROUGE when provided
    started_at: float  # perf_counter() when the workflow started

    # --- Preprocessing ---------------------------------------------------------
    cleaned_text: str
    sentences: list[str]
    sentence_pages: list[int | None]  # source page of each sentence (None without pages)
    original_word_count: int

    # --- Extractive scoring / Hybrid selection -------------------------------
    selected_sentences: list[int]  # indices into ``sentences``, document order
    sentence_scores: list[float]

    # --- Abstractive working state --------------------------------------------
    work_sentences: list[str]  # what the model must summarize (all sentences, or Hybrid's selection)
    work_pages: list[int | None] | None  # source page of each work sentence (None after a reduction round)
    target_words: float  # requested summary length, relative to the original document
    input_tokens: int
    fuse: bool  # can one final pass write the whole summary?

    # --- Long-document loop ---------------------------------------------------
    chunk_round: int
    round_target_words: float
    chunks: list[Chunk]
    intermediate_summaries: list[str]  # chunk summaries of the latest round
    combined_text: str
    next_step: str  # "final_pass" | "done" | "another_round" | "max_levels"
    reduction_levels: Annotated[list[dict], operator.add]

    # --- Output -------------------------------------------------------------------
    final_summary: str
    strategy: str  # "extractive" | "single_pass" | "fused" | "concatenated" | "max_levels_reached"
    metadata: Annotated[dict, merge_dicts]
    metrics: dict  # statistics + ROUGE (None without a reference summary)
    faithfulness: dict  # experimental "potentially unsupported content" check
    processing_time: float

    # --- Diagnostics --------------------------------------------------------------
    path: Annotated[list[str], operator.add]  # nodes visited, in order
    timings: Annotated[dict, add_timings]  # seconds spent in each node
    warnings: Annotated[list[str], operator.add]  # non-fatal issues shown to the user
    errors: Annotated[list[dict], operator.add]  # fatal errors: {node, type, message}
    error: Any  # the first fatal IntelliSumError, re-raised by the workflow runner
