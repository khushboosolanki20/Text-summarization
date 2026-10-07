"""
Common interface and helpers shared by all summarizers.

Every summarizer receives the document as a list of sentences (produced once
by ``app.preprocessing``) and returns a ``SummaryResult``. This keeps the
algorithms independent of how text was loaded, and lets the workflow treat
TF-IDF, TextRank, BART and Hybrid interchangeably.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.config import get_settings
from app.preprocessing.cleaner import count_words


# similarity(i) -> array of cosine similarities between sentence i and every
# sentence. Computed on demand (one sparse matrix-vector product) rather than
# as a full n x n matrix: selection only needs rows for the few candidates it
# examines, and a full matrix for a ~5,000-sentence document needs ~200 MB.
SimilarityFn = Callable[[int], np.ndarray]


class SummaryLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


def length_ratio(length: SummaryLength | str) -> float:
    """Fraction of the input the summary should keep (see Settings.length_ratios)."""
    return get_settings().length_ratios[SummaryLength(length).value]


def target_sentence_count(num_sentences: int, length: SummaryLength | str) -> int:
    """
    How many sentences an extractive summary should contain.

    ``ceil`` (rather than round) ensures that short documents still get at
    least one sentence, and the summary is always shorter than the input when
    there is more than one sentence.
    """
    if num_sentences <= 1:
        return num_sentences
    # round() first so floating-point noise (e.g. 11.000000000000002) can't
    # push ceil() up by a whole sentence.
    k = math.ceil(round(num_sentences * length_ratio(length), 6))
    return max(1, min(k, num_sentences - 1))


def compression_ratio(original_words: int, summary_words: int) -> float:
    """Percentage of the original text removed: 100 * (1 - summary / original)."""
    if original_words == 0:
        return 0.0
    return round(100.0 * (1 - summary_words / original_words), 2)


@dataclass
class SummaryResult:
    summary: str
    method: str
    original_word_count: int
    summary_word_count: int
    # Extractive methods only: indices (into the input sentence list, in
    # document order) of the sentences that form the summary, and one score
    # per input sentence so the UI can explain *why* sentences were chosen.
    selected_indices: list[int] | None = None
    sentence_scores: list[float] | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        return compression_ratio(self.original_word_count, self.summary_word_count)


class BaseSummarizer(ABC):
    """All summarization methods implement this interface."""

    name: str

    @abstractmethod
    def summarize(self, sentences: list[str], length: SummaryLength | str = SummaryLength.MEDIUM) -> SummaryResult:
        """Summarize a document given as an ordered list of sentences."""


class ExtractiveSummarizer(BaseSummarizer):
    """
    Template for extractive methods. Subclasses only implement
    ``score_sentences``; ranking, redundancy control, selection and
    re-ordering are shared, so TF-IDF and TextRank differ *only* in how
    sentence importance is computed.
    """

    @abstractmethod
    def score_sentences(self, sentences: list[str]) -> tuple[list[float], SimilarityFn | None, dict]:
        """
        Return ``(scores, similarity, metadata)``: one importance score per
        sentence, an optional pairwise similarity function used to avoid
        selecting near-duplicates, and method-specific metadata.
        """

    def summarize(self, sentences: list[str], length: SummaryLength | str = SummaryLength.MEDIUM) -> SummaryResult:
        original_words = sum(count_words(s) for s in sentences)
        if not sentences:
            return SummaryResult("", self.name, 0, 0, [], [])

        scores, similarity, metadata = self.score_sentences(sentences)
        k = target_sentence_count(len(sentences), length)
        selected = select_top_sentences(scores, k, similarity, get_settings().redundancy_threshold)

        summary = " ".join(sentences[i] for i in selected)
        return SummaryResult(
            summary=summary,
            method=self.name,
            original_word_count=original_words,
            summary_word_count=count_words(summary),
            selected_indices=selected,
            sentence_scores=[round(float(s), 6) for s in scores],
            metadata={"num_sentences": len(sentences), "num_selected": len(selected), **metadata},
        )


def select_top_sentences(
    scores: list[float],
    k: int,
    similarity: SimilarityFn | None = None,
    redundancy_threshold: float = 1.0,
) -> list[int]:
    """
    Pick the ``k`` highest-scoring sentences and return their indices in
    **original document order**, so the summary reads in the same sequence
    as the source.

    Redundancy control: walking down the ranking, a sentence is skipped if
    it is more similar than ``redundancy_threshold`` to a sentence already
    chosen. Documents often repeat a key point in near-identical words, and
    both copies would otherwise score highly. If skipping leaves fewer than
    ``k`` sentences, the skipped ones are used to fill up the summary.
    """
    # Sort by score (highest first); ties broken by earlier position.
    ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    selected: list[int] = []
    skipped: list[int] = []
    for i in ranking:
        if len(selected) == k:
            break
        if similarity is not None and selected and similarity(i)[selected].max() > redundancy_threshold:
            skipped.append(i)
        else:
            selected.append(i)
    selected.extend(skipped[: k - len(selected)])
    return sorted(selected)
