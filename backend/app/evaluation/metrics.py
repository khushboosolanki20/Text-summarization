"""
Summary statistics that can be computed for any document (no reference needed).

ROUGE, which needs a human-written reference summary, is added in
``app.evaluation.rouge`` (Phase 10).
"""

from app.preprocessing.cleaner import count_words
from app.summarizers.base import compression_ratio


def summary_statistics(
    original_word_count: int,
    summary: str,
    num_sentences: int,
    num_selected: int | None,
    processing_time: float,
) -> dict:
    summary_words = count_words(summary)
    return {
        "original_word_count": original_word_count,
        "summary_word_count": summary_words,
        # % of the original removed: 100 x (1 - summary / original)
        "compression_ratio": compression_ratio(original_word_count, summary_words),
        "num_sentences": num_sentences,
        # Extractive and hybrid methods only: how many source sentences were used.
        "num_selected_sentences": num_selected,
        "processing_time": round(processing_time, 3),
    }
