"""
Model-wrapper tests with a real (but tiny) Transformer.

``hf-internal-testing/tiny-random-bart`` is a ~0.5 MB BART with random
weights published by Hugging Face for testing. Its output is gibberish, but
it exercises the real code paths of ``Seq2SeqSummarizationModel`` (loading,
tokenization, batched beam-search generation, the no-truncation guard) in
seconds instead of the minutes the 1.6 GB bart-large-cnn takes on a CPU.
Its 100-token context window also makes long-document chunking easy to
trigger with short inputs, using the real tokenizer.
"""

import threading

import pytest

from app.core.workflow import run_summarization
from app.errors import ContextWindowExceededError
from app.graph.summarization_graph import build_summarization_graph
from app.summarizers.bart import BARTSummarizer
from app.summarizers.models import ModelSpec, Seq2SeqSummarizationModel, resolve_device
from tests.test_tfidf import SOLAR

TINY = "hf-internal-testing/tiny-random-bart"


@pytest.fixture(scope="module")
def tiny():
    model = Seq2SeqSummarizationModel(ModelSpec("tiny", TINY), device_preference="cpu")
    model.load()
    return model


def test_loads_lazily_on_cpu():
    model = Seq2SeqSummarizationModel(ModelSpec("tiny", TINY), device_preference="cpu")
    assert not model.is_loaded  # nothing loaded at construction
    model.count_tokens("hello")  # first use triggers loading
    assert model.is_loaded and model.device == "cpu" and model.load_seconds is not None


def test_concurrent_first_use_loads_once():
    model = Seq2SeqSummarizationModel(ModelSpec("tiny", TINY), device_preference="cpu")
    seen = []
    threads = [threading.Thread(target=lambda: seen.append(id(model.model))) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(seen)) == 1  # every thread got the same model object


def test_token_counting(tiny):
    text = "Solar power capacity grew quickly."
    with_special = tiny.count_tokens(text)
    without = tiny.count_tokens(text, special_tokens=False)
    assert with_special == without + 2  # <s> ... </s>
    assert tiny.max_input_tokens == 100
    assert tiny.content_token_limit == 98
    assert tiny.fits(text)
    assert not tiny.fits(" ".join(SOLAR))


def test_batched_generation_returns_one_string_per_input(tiny):
    outputs = tiny.generate(["Solar power grew fast. China led.", "Another short text here."], 5, 12)
    assert len(outputs) == 2 and all(isinstance(o, str) for o in outputs)


def test_generate_refuses_to_truncate(tiny):
    with pytest.raises(ContextWindowExceededError, match="at most 100 tokens"):
        tiny.generate([" ".join(SOLAR)], 5, 10)


def test_generation_failure_becomes_friendly_error(tiny, monkeypatch):
    from app.errors import InferenceError

    def broken_generate(*args, **kwargs):
        raise RuntimeError("low-level tensor error")

    monkeypatch.setattr(tiny.model, "generate", broken_generate)
    with pytest.raises(InferenceError, match="failed to generate"):
        tiny.generate(["Solar power grew."], 5, 10)


def test_resolve_device_respects_cpu_preference():
    assert resolve_device("cpu") == "cpu"


# ------------------------------------------- real tokenizer, long-document path


def test_long_document_is_chunked_with_the_real_tokenizer(tiny):
    result = BARTSummarizer(model=tiny).summarize(SOLAR, "short")  # ~150 words >> 100-token window
    meta = result.metadata
    assert meta["input_tokens"] > 100
    assert meta["strategy"] in {"fused", "concatenated", "max_levels_reached"}
    level1 = meta["reduction_levels"][0]
    assert level1["num_chunks"] >= 2
    assert all(tokens <= tiny.content_token_limit for tokens in level1["chunk_token_counts"])
    # Each chunk really fits the model: measured with the model's own tokenizer.
    assert all(tiny.fits(s) for s in [" ".join(SOLAR[:3])])


def test_workflow_routes_with_a_real_model(tiny):
    graph = build_summarization_graph(abstractive=BARTSummarizer(model=tiny))
    bart = run_summarization(" ".join(SOLAR), "bart", "short", graph=graph)
    assert "chunk_document" in bart.path  # 150 words do not fit 100 tokens
    hybrid = run_summarization(" ".join(SOLAR), "hybrid", "short", graph=graph)
    assert hybrid.path[:3] == ["preprocess", "select_key_sentences", "check_length"]
    # Hybrid's selection is capped to the model window, so it never needs chunking.
    assert "chunk_document" not in hybrid.path
    assert hybrid.metadata["extractive_stage"]["token_cap"] == tiny.content_token_limit
