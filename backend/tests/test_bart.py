"""
BART tests.

Fast tests cover length budgets, output cleaning, lazy loading and error
handling without loading the model. Tests marked ``slow`` load the real
facebook/bart-large-cnn (~1.6 GB download on first run); skip them with
``pytest -m "not slow"``.
"""

import pytest

from app.config import get_settings
from app.errors import ContextWindowExceededError, ModelLoadError
from app.summarizers.bart import BARTSummarizer, tidy_generated_text, token_budget
from app.summarizers.models import ModelSpec, Seq2SeqSummarizationModel, get_model
from tests.test_tfidf import SOLAR

# ------------------------------------------------------------ length budget


def test_budget_grows_with_length_setting():
    budgets = [token_budget(500, length) for length in ("short", "medium", "long")]
    assert budgets[0][1] < budgets[1][1] < budgets[2][1]


def test_budget_min_is_below_max():
    for words in (10, 100, 700, 5000):
        for length in ("short", "medium", "long"):
            low, high = token_budget(words, length)
            assert 1 <= low < high


def test_budget_respects_configured_bounds():
    settings = get_settings()
    assert token_budget(100_000, "long")[1] == settings.max_summary_tokens
    assert token_budget(5, "short")[0] >= 1
    assert token_budget(5, "short")[1] >= settings.min_summary_tokens


def test_budget_matches_ratio():
    # 500 words * 22 % * 1.3 tokens/word = 143 tokens target
    low, high = token_budget(500, "medium")
    assert low == int(143 * 0.75) and high == int(143 * 1.25)


# ------------------------------------------------------------ output cleaning


def test_tidy_keeps_complete_text():
    assert tidy_generated_text("A complete summary. Another sentence.") == ("A complete summary. Another sentence.", False)


def test_tidy_trims_incomplete_last_sentence():
    assert tidy_generated_text("First sentence. Second one was cut off in the") == ("First sentence.", True)


def test_tidy_keeps_single_incomplete_sentence():
    # Nothing complete to fall back to: keep the text rather than return "".
    assert tidy_generated_text("only a fragment without end") == ("only a fragment without end", False)


def test_tidy_normalises_spacing():
    assert tidy_generated_text("  Spaces  before , punctuation .") == ("Spaces before, punctuation.", False)


def test_tidy_handles_closing_quote():
    assert tidy_generated_text('He said "it works."')[1] is False


# ------------------------------------------------------- lazy loading


def test_registry_returns_shared_instance_without_loading():
    first, second = get_model("bart"), get_model("bart")
    assert first is second
    assert first.spec.hf_id == get_settings().abstractive_model_name


def test_t5_and_pegasus_are_registered():
    assert get_model("t5").spec.prefix == "summarize: "
    assert get_model("pegasus").spec.hf_id == "google/pegasus-cnn_dailymail"


def test_unknown_model_key():
    with pytest.raises(ValueError):
        get_model("gpt")


def test_load_failure_raises_friendly_error():
    broken = Seq2SeqSummarizationModel(ModelSpec("broken", "./no/such/local/model/dir"))
    with pytest.raises(ModelLoadError, match="Extractive methods"):
        broken.load()
    assert not broken.is_loaded


def test_empty_input_does_not_load_model():
    model = Seq2SeqSummarizationModel(ModelSpec("unused", "./never/loaded"))
    result = BARTSummarizer(model=model).summarize([])
    assert result.summary == "" and not model.is_loaded


# ------------------------------------------------------------ real model

ARTICLE = SOLAR  # ten sentences about a solar-energy report (~150 words)


@pytest.fixture(scope="module")
def bart():
    return BARTSummarizer()


@pytest.mark.slow
def test_generates_shorter_abstractive_summary(bart):
    result = bart.summarize(ARTICLE, "medium")
    assert result.method == "bart"
    assert 5 < result.summary_word_count < result.original_word_count
    assert "solar" in result.summary.lower()
    assert result.selected_indices is None  # abstractive: no selected sentences
    meta = result.metadata
    assert meta["model"] == "facebook/bart-large-cnn"
    assert meta["input_tokens"] <= meta["max_input_tokens"] == 1024
    assert meta["device"] in {"cpu", "cuda"}


@pytest.mark.slow
def test_is_deterministic(bart):
    assert bart.summarize(ARTICLE, "short").summary == bart.summarize(ARTICLE, "short").summary


@pytest.mark.slow
def test_model_is_loaded_only_once(bart):
    bart.summarize(ARTICLE, "short")
    model_before = bart.model.model
    result = bart.summarize(ARTICLE, "short")
    assert bart.model.model is model_before
    assert result.metadata["model_loaded_now"] is False
    assert BARTSummarizer().model is bart.model  # new summarizer instances share the model


@pytest.mark.slow
def test_long_setting_is_not_shorter_than_short(bart):
    long_doc = ARTICLE * 3  # ~450 words: gives the length budgets room to differ
    short = bart.summarize(long_doc, "short").summary_word_count
    long = bart.summarize(long_doc, "long").summary_word_count
    assert long > short


@pytest.mark.slow
def test_model_refuses_to_truncate_over_long_input(bart):
    # The low-level generate() call must never silently cut input off.
    too_long = " ".join(ARTICLE * 12)  # ~1,500 words, well over 1,024 tokens
    with pytest.raises(ContextWindowExceededError, match="1024"):
        bart.model.generate([too_long], 20, 40)


@pytest.mark.slow
def test_long_document_is_summarized_hierarchically(bart):
    # ~1,500 words: does not fit in one pass, so it must be chunked.
    long_doc = [f"{s[:-1]} in region {i}." for i in range(10) for s in ARTICLE]
    progress = []
    result = bart.summarize(long_doc, "short", on_progress=lambda stage, done, total: progress.append((stage, done, total)))
    meta = result.metadata
    assert meta["input_tokens"] > 1024
    assert meta["strategy"] == "fused"
    assert meta["chunks"] >= 2
    level1 = meta["reduction_levels"][0]
    assert all(tokens <= 900 for tokens in level1["chunk_token_counts"])
    assert len(meta["intermediate_summaries"]) == meta["chunks"]
    assert 10 < result.summary_word_count < result.original_word_count * 0.3
    assert progress[-1] == ("final summarization pass", 1, 1)


def test_very_long_input_is_rejected_with_hybrid_suggestion(monkeypatch):
    from app.errors import InputTooLargeError

    monkeypatch.setattr(get_settings(), "abstractive_max_input_words", 50)
    model = Seq2SeqSummarizationModel(ModelSpec("unused", "./never/loaded"))
    with pytest.raises(InputTooLargeError, match="Hybrid"):
        BARTSummarizer(model=model).summarize(ARTICLE * 2)
    assert not model.is_loaded
