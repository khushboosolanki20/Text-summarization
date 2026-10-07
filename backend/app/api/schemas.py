"""Request and response models of the REST API (validated by Pydantic)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.workflow import SummarizationOutput
from app.summarizers.base import SummaryLength, SummaryMethod

# --------------------------------------------------------------------- requests


class SummarizeTextRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Paste a news article, report or essay here...",
                "method": "hybrid",
                "length": "medium",
                "reference_summary": None,
            }
        }
    )

    text: str = Field(..., description="The text to summarize.")
    method: SummaryMethod = Field(SummaryMethod.HYBRID, description="tfidf | textrank | bart | hybrid")
    length: SummaryLength = Field(SummaryLength.MEDIUM, description="short | medium | long")
    reference_summary: str | None = Field(
        None, description="Optional human-written summary. ROUGE is only computed when this is provided."
    )


class EvaluateRequest(BaseModel):
    summary: str = Field(..., description="The summary to evaluate (candidate).")
    reference_summary: str = Field(..., description="Human-written reference summary.")
    original_text: str | None = Field(
        None, description="Optional source text: enables compression ratio and the faithfulness check."
    )


# -------------------------------------------------------------------- responses


class RougeScore(BaseModel):
    precision: float
    recall: float
    f1: float


class Metrics(BaseModel):
    original_word_count: int
    summary_word_count: int
    compression_ratio: float = Field(..., description="% of the original removed: 100 x (1 - summary / original)")
    num_sentences: int
    num_selected_sentences: int | None = Field(None, description="Extractive and hybrid methods only")
    processing_time: float = Field(..., description="Seconds")
    rouge1: float | None = Field(None, description="ROUGE-1 F1, or null without a reference summary")
    rouge2: float | None = Field(None, description="ROUGE-2 F1, or null without a reference summary")
    rougeL: float | None = Field(None, description="ROUGE-L F1, or null without a reference summary")
    rouge: dict[str, RougeScore] | None = Field(None, description="Precision/recall/F1 for rouge1, rouge2, rougeL, rougeLsum")
    rouge_note: str | None = Field(None, description="Why ROUGE was not computed, if it was not")


class SentenceInfo(BaseModel):
    index: int
    text: str
    page: int | None = Field(None, description="Source page (PDF uploads)")
    score: float | None = Field(None, description="TF-IDF / TextRank importance score")
    selected: bool = Field(False, description="In the summary (extractive) or passed to BART (hybrid)")


class SummaryResponse(BaseModel):
    summary: str
    method: SummaryMethod
    length: SummaryLength
    original_word_count: int
    summary_word_count: int
    compression_ratio: float
    processing_time: float
    metrics: Metrics
    strategy: str = Field(..., description="extractive | single_pass | fused | concatenated | max_levels_reached")
    sentences: list[SentenceInfo]
    faithfulness: dict[str, Any] = Field(..., description="Experimental 'potentially unsupported content' check")
    intermediate_summaries: list[str] = Field(default_factory=list, description="Chunk summaries (long documents)")
    warnings: list[str] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict, description="Where the text came from")
    path: list[str] = Field(default_factory=list, description="Workflow nodes visited")
    timings: dict[str, float] = Field(default_factory=dict, description="Seconds per workflow node")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Method-specific details")

    @classmethod
    def from_output(cls, out: SummarizationOutput, source: dict[str, Any]) -> "SummaryResponse":
        selected = set(out.selected_indices or [])
        scores = out.sentence_scores
        pages = out.sentence_pages or [None] * len(out.sentences)
        return cls(
            summary=out.summary,
            method=out.method,
            length=out.summary_length,
            original_word_count=out.original_word_count,
            summary_word_count=out.summary_word_count,
            compression_ratio=out.compression_ratio,
            processing_time=out.processing_time,
            metrics=Metrics(**out.metrics),
            strategy=out.strategy,
            sentences=[
                SentenceInfo(
                    index=i,
                    text=text,
                    page=pages[i],
                    score=scores[i] if scores else None,
                    selected=i in selected,
                )
                for i, text in enumerate(out.sentences)
            ],
            faithfulness=out.faithfulness,
            intermediate_summaries=out.intermediate_summaries,
            warnings=out.warnings,
            source=source,
            path=out.path,
            timings=out.timings,
            metadata=out.metadata,
        )


class EvaluateResponse(BaseModel):
    rouge1: float
    rouge2: float
    rougeL: float
    rouge: dict[str, RougeScore]
    summary_word_count: int
    reference_word_count: int
    compression_ratio: float | None = None
    faithfulness: dict[str, Any] | None = None


class JobProgress(BaseModel):
    stage: str
    done: int
    total: int


class JobResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="queued | running | completed | failed")
    progress: JobProgress | None = None
    result: SummaryResponse | None = None
    error: str | None = None
    created_at: float
    finished_at: float | None = None


class ConfigResponse(BaseModel):
    methods: list[dict[str, Any]]
    lengths: dict[str, float]
    supported_file_types: list[str]
    max_upload_mb: float
    max_input_chars: int
    min_input_words: int
    abstractive_max_input_words: int
    abstractive_model: str
    faithfulness_check: bool
