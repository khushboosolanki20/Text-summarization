"""
Evaluation of a produced summary.

* ``summary_statistics``: computable for any document (word counts,
  compression, sentence counts, time).
* ``evaluate_summary``: statistics plus ROUGE, but ROUGE only when a
  human-written reference summary is supplied. Without one the ROUGE fields
  are ``None`` and ``rouge_note`` says why: a score is never invented.
"""

from app.evaluation.rouge import rouge_or_none
from app.preprocessing.cleaner import count_words
from app.summarizers.base import compression_ratio

ROUGE_UNAVAILABLE = (
    "ROUGE needs a human-written reference summary to compare against. "
    "No reference was provided, so ROUGE was not computed."
)


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


def rouge_metrics(summary: str, reference_summary: str | None) -> dict:
    """
    ROUGE fields for a response. ``rouge1`` / ``rouge2`` / ``rougeL`` are F1
    scores in [0, 1] (or None); ``rouge`` holds precision/recall/F1 for every
    variant including ROUGE-Lsum.
    """
    rouge = rouge_or_none(summary, reference_summary)
    return {
        "rouge1": rouge["rouge1"]["f1"] if rouge else None,
        "rouge2": rouge["rouge2"]["f1"] if rouge else None,
        "rougeL": rouge["rougeL"]["f1"] if rouge else None,
        "rouge": rouge,
        "rouge_note": None if rouge else ROUGE_UNAVAILABLE,
    }


def evaluate_summary(
    summary: str,
    original_word_count: int,
    num_sentences: int,
    num_selected: int | None,
    processing_time: float,
    reference_summary: str | None = None,
) -> dict:
    return {
        **summary_statistics(original_word_count, summary, num_sentences, num_selected, processing_time),
        **rouge_metrics(summary, reference_summary),
    }
