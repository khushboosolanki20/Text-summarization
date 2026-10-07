"""
Hybrid (TextRank -> BART) tests.

Fast tests run the real TextRank with a stand-in for BART that records what
it receives (see test_long_document.KeepFirstWords), to check what Hybrid
passes on and with which length target. The slow test runs the real model.
"""

import random
import string
import time

import pytest

from app.summarizers.bart import BARTSummarizer
from app.summarizers.generation import max_single_pass_words
from app.summarizers.hybrid import HybridSummarizer
from tests.test_long_document import KeepFirstWords
from tests.test_tfidf import OFF_TOPIC, SOLAR


class StandInModel(KeepFirstWords):
    """KeepFirstWords plus the attributes BARTSummarizer reports in metadata."""

    is_loaded = True
    device = "cpu"
    load_seconds = 0.0

    class spec:
        hf_id = "stand-in"


def hybrid(expansion: float = 3.0) -> tuple[HybridSummarizer, StandInModel]:
    model = StandInModel()
    return HybridSummarizer(abstractive=BARTSummarizer(model=model), expansion=expansion), model


_rng = random.Random(1)
_FILLER = ["".join(_rng.choices(string.ascii_lowercase, k=_rng.randint(5, 9))) for _ in range(500)]


def variants(blocks: int) -> list[str]:
    """
    The solar article repeated ``blocks`` times, each sentence extended with
    six distinct filler words. (Varying only a number would not work: the
    tokenizer ignores digits, so the copies would be identical vectors and
    redundancy control would rightly discard them.)
    """
    return [f"{s[:-1]}, according to {' '.join(_rng.sample(_FILLER, 6))}." for _ in range(blocks) for s in SOLAR]


LONG_DOC = variants(10)  # ~2,200 words: too long for one BART pass


def test_empty_input():
    summarizer, model = hybrid()
    assert summarizer.summarize([]).summary == ""
    assert model.calls == []


def test_selection_is_in_document_order_and_excludes_off_topic():
    summarizer, _ = hybrid()
    # "short": budget = 3 x 12 % of 150 words = 54 words, so TextRank must choose.
    result = summarizer.summarize(SOLAR, "short")
    assert 2 <= len(result.selected_indices) < len(SOLAR)
    assert result.selected_indices == sorted(result.selected_indices)
    assert not OFF_TOPIC & set(result.selected_indices)


def test_short_document_with_long_setting_passes_almost_everything():
    # 3 x 32 % = 96 % of the document: for short texts Hybrid ~ BART, by design.
    summarizer, _ = hybrid()
    result = summarizer.summarize(SOLAR, "long")
    assert len(result.selected_indices) >= len(SOLAR) - 1


def test_bart_receives_only_the_selected_sentences():
    summarizer, model = hybrid()
    result = summarizer.summarize(LONG_DOC, "short")
    bart_input = model.calls[0][0]
    assert bart_input == " ".join(LONG_DOC[i] for i in result.selected_indices)
    assert result.metadata["extractive_stage"]["input_reduction"] > 50


def test_short_target_needs_only_one_bart_pass():
    summarizer, model = hybrid()
    result = summarizer.summarize(LONG_DOC, "short")  # T ~ 180 words: fits one pass
    stage = result.metadata["extractive_stage"]
    assert stage["token_cap"] == 900
    assert len(model.calls) == 1  # no chunking needed
    assert result.metadata["abstractive_stage"]["strategy"] == "single_pass"


def test_length_target_is_relative_to_original_document():
    summarizer, _ = hybrid()
    result = summarizer.summarize(LONG_DOC, "short")
    expected = result.original_word_count * 0.12
    assert result.metadata["abstractive_stage"]["target_words"] == round(expected)
    # The selection is several times longer than the target...
    assert result.metadata["extractive_stage"]["selected_words"] > 2 * expected
    # ...yet the summary aims at the target, not at 12 % of the selection.
    assert 0.6 * expected <= result.summary_word_count <= 1.3 * expected


def test_word_budget_scales_with_expansion():
    small, _ = hybrid(expansion=1.5)
    large, _ = hybrid(expansion=3.0)
    w_small = small.summarize(LONG_DOC, "short").metadata["extractive_stage"]["selected_words"]
    w_large = large.summarize(LONG_DOC, "short").metadata["extractive_stage"]["selected_words"]
    assert w_small < w_large


def test_long_target_selects_without_token_cap():
    summarizer, _ = hybrid()
    doc = variants(60)  # ~13,000 words: T > one pass
    result = summarizer.summarize(doc, "short")
    assert result.metadata["abstractive_stage"]["target_words"] > max_single_pass_words()
    stage = result.metadata["extractive_stage"]
    assert stage["token_cap"] is None
    assert stage["selected_words"] <= stage["word_budget"] * 1.1


def test_near_duplicates_are_skipped():
    summarizer, _ = hybrid()
    doc = SOLAR + [SOLAR[0]]  # exact duplicate of the first sentence
    result = summarizer.summarize(doc, "long")
    assert not {0, len(doc) - 1} <= set(result.selected_indices)
    assert result.metadata["extractive_stage"]["redundant_skipped"] >= 1


def test_metadata_has_both_stages():
    summarizer, _ = hybrid()
    result = summarizer.summarize(LONG_DOC, "medium")
    meta = result.metadata
    assert meta["extractive_stage"]["method"] == "textrank"
    assert "graph" in meta["extractive_stage"]
    assert meta["abstractive_stage"]["model"] == "stand-in"
    assert meta["num_selected"] == len(result.selected_indices)
    assert len(result.sentence_scores) == len(LONG_DOC)


def test_invalid_expansion():
    with pytest.raises(ValueError):
        HybridSummarizer(abstractive=BARTSummarizer(model=StandInModel()), expansion=0.5)


@pytest.mark.slow
def test_real_hybrid_is_faster_than_bart_on_long_document():
    bart = BARTSummarizer()
    bart.model.load()
    start = time.perf_counter()
    bart_result = bart.summarize(LONG_DOC, "short")
    bart_seconds = time.perf_counter() - start

    start = time.perf_counter()
    result = HybridSummarizer(abstractive=bart).summarize(LONG_DOC, "short")
    hybrid_seconds = time.perf_counter() - start

    assert bart_result.metadata["chunks"] >= 2
    assert result.metadata["abstractive_stage"]["strategy"] == "single_pass"
    assert 10 < result.summary_word_count < result.original_word_count * 0.3
    assert "solar" in result.summary.lower()
    assert hybrid_seconds < bart_seconds
    print(f"\nBART {bart_seconds:.1f}s ({bart_result.metadata['chunks']} chunks) vs Hybrid {hybrid_seconds:.1f}s")
