"""
Service layer: run the LangGraph workflow for one request.

The API (and the experiment scripts) call ``run_summarization`` and receive a
plain ``SummarizationOutput``. They never deal with graph state, and the
workflow can change internally without touching the API.
"""

import time
from dataclasses import dataclass, field
from functools import lru_cache

from app.graph.summarization_graph import build_summarization_graph
from app.summarizers.base import SummaryLength, SummaryMethod
from app.summarizers.long_document import ProgressCallback


@dataclass
class SummarizationOutput:
    summary: str
    method: str
    summary_length: str
    original_word_count: int
    summary_word_count: int
    compression_ratio: float
    processing_time: float
    metrics: dict
    sentences: list[str]  # the document's sentences (for highlighting in the UI)
    selected_indices: list[int] | None  # extractive: summary sentences; hybrid: sentences passed to BART
    sentence_scores: list[float] | None
    strategy: str
    metadata: dict
    intermediate_summaries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)  # graph nodes visited
    timings: dict = field(default_factory=dict)  # seconds per node


@lru_cache(maxsize=1)
def get_graph():
    """The compiled graph is built once per process and reused."""
    return build_summarization_graph()


def run_summarization(
    text: str,
    method: SummaryMethod | str = SummaryMethod.HYBRID,
    length: SummaryLength | str = SummaryLength.MEDIUM,
    reference_summary: str | None = None,
    warnings: list[str] | None = None,
    on_progress: ProgressCallback | None = None,
    graph=None,
) -> SummarizationOutput:
    """
    Summarize ``text``. Raises an ``IntelliSumError`` subclass for invalid
    input or model problems (the API turns these into friendly messages).
    ``warnings`` carries non-fatal issues from document loading, e.g.
    skipped scanned pages, through to the response.
    """
    method = SummaryMethod(method)  # ValueError for unknown values; the API validates first
    length = SummaryLength(length)
    graph = graph or get_graph()

    state = graph.invoke(
        {
            "original_text": text,
            "method": method.value,
            "summary_length": length.value,
            "reference_summary": reference_summary,
            "started_at": time.perf_counter(),
            "warnings": list(warnings or []),
        },
        config={"configurable": {"on_progress": on_progress}},
    )
    if state.get("error"):
        raise state["error"]

    metrics = state["metrics"]
    levels = state.get("reduction_levels") or []
    metadata = dict(state.get("metadata", {}))
    if levels:
        metadata["reduction_levels"] = [{k: v for k, v in lvl.items() if k != "summaries"} for lvl in levels]
    return SummarizationOutput(
        summary=state["final_summary"],
        method=method.value,
        summary_length=length.value,
        original_word_count=metrics["original_word_count"],
        summary_word_count=metrics["summary_word_count"],
        compression_ratio=metrics["compression_ratio"],
        processing_time=state["processing_time"],
        metrics=metrics,
        sentences=state["sentences"],
        selected_indices=state.get("selected_sentences"),
        sentence_scores=state.get("sentence_scores"),
        strategy=state["strategy"],
        metadata=metadata,
        intermediate_summaries=levels[0]["summaries"] if levels else [],
        warnings=state.get("warnings", []),
        path=state.get("path", []),
        timings=state.get("timings", {}),
    )
