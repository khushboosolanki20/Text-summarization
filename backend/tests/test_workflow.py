"""
LangGraph workflow tests: routing, loops, state, errors and progress.

Abstractive paths use the stand-in model from test_hybrid so every route can
be exercised in seconds; one slow test runs the real BART end to end.
"""

import pytest

from app.core.workflow import run_summarization
from app.errors import EmptyDocumentError, ModelLoadError, TextTooShortError
from app.graph.summarization_graph import build_summarization_graph
from app.summarizers.bart import BARTSummarizer
from app.summarizers.hybrid import HybridSummarizer
from app.summarizers.tfidf import TFIDFSummarizer
from tests.test_hybrid import StandInModel, variants
from tests.test_tfidf import SOLAR

SOLAR_TEXT = " ".join(SOLAR)


def graph_with(model):
    return build_summarization_graph(abstractive=BARTSummarizer(model=model))


@pytest.fixture
def stand_in():
    return StandInModel()


@pytest.fixture
def graph(stand_in):
    return graph_with(stand_in)


# --------------------------------------------------------------- routing


@pytest.mark.parametrize("method", ["tfidf", "textrank"])
def test_extractive_route_never_touches_the_model(method, graph, stand_in):
    out = run_summarization(SOLAR_TEXT, method, "medium", graph=graph)
    assert out.path == ["preprocess", "extractive_summarize", "postprocess", "evaluate"]
    assert out.strategy == "extractive"
    assert out.selected_indices == sorted(out.selected_indices)
    assert out.summary == " ".join(out.sentences[i] for i in out.selected_indices)
    assert stand_in.calls == []


def test_bart_short_document_single_pass(graph, stand_in):
    out = run_summarization(SOLAR_TEXT, "bart", "medium", graph=graph)
    assert out.path == ["preprocess", "check_length", "abstractive_single_pass", "postprocess", "evaluate"]
    assert out.strategy == "single_pass"
    assert len(stand_in.calls) == 1
    assert out.selected_indices is None


def test_bart_long_document_takes_the_chunk_loop(graph, stand_in):
    out = run_summarization(" ".join(variants(10)), "bart", "short", graph=graph)
    assert out.path == [
        "preprocess", "check_length", "chunk_document", "summarize_chunks",
        "combine_summaries", "final_summarization", "postprocess", "evaluate",
    ]  # fmt: skip
    assert out.strategy == "fused"
    assert out.metadata["chunks"] == len(out.intermediate_summaries) >= 2
    assert len(stand_in.calls) == out.metadata["chunks"] + 1  # chunks + final pass


def test_very_long_target_is_concatenated_without_final_pass(graph):
    out = run_summarization(" ".join(variants(60)), "bart", "short", graph=graph)
    assert out.strategy == "concatenated"
    assert "final_summarization" not in out.path
    assert out.summary == " ".join(out.intermediate_summaries)


def test_loop_runs_several_rounds_and_stops():
    stubborn = StandInModel()
    stubborn.compress = False  # never shrinks the text
    out = run_summarization(" ".join(variants(10)), "bart", "short", graph=graph_with(stubborn))
    assert out.path.count("chunk_document") == 3  # max_reduction_levels
    assert out.strategy == "max_levels_reached"
    assert [lvl["level"] for lvl in out.metadata["reduction_levels"]] == [1, 2, 3]
    assert out.warnings  # the user is told the result may be long


def test_hybrid_route_selects_then_summarizes(graph, stand_in):
    out = run_summarization(" ".join(variants(10)), "hybrid", "short", graph=graph)
    assert out.path == [
        "preprocess", "select_key_sentences", "check_length", "abstractive_single_pass", "postprocess", "evaluate",
    ]  # fmt: skip
    assert stand_in.calls[0][0] == " ".join(out.sentences[i] for i in out.selected_indices)
    # Target is 12 % of the ORIGINAL document (computed from its sentences,
    # which can be a few words shorter than the cleaned text if fragments
    # were dropped), not 12 % of the selection.
    expected = out.original_word_count * 0.12
    assert abs(out.metadata["target_words"] - expected) <= 0.02 * expected
    assert out.metadata["extractive_stage"]["input_reduction"] > 50


# --------------------------------------------- same results as direct calls


def test_graph_matches_direct_extractive_call(graph):
    out = run_summarization(SOLAR_TEXT, "tfidf", "medium", graph=graph)
    direct = TFIDFSummarizer().summarize(out.sentences, "medium")
    assert out.summary == direct.summary
    assert out.sentence_scores == direct.sentence_scores


def test_graph_matches_direct_hybrid_call():
    text = " ".join(variants(10))
    out = run_summarization(text, "hybrid", "short", graph=graph_with(StandInModel()))
    direct = HybridSummarizer(abstractive=BARTSummarizer(model=StandInModel())).summarize(out.sentences, "short")
    assert out.summary == direct.summary
    assert out.selected_indices == direct.selected_indices


# ------------------------------------------------------- state and outputs


def test_metrics_and_diagnostics(graph):
    out = run_summarization(SOLAR_TEXT, "textrank", "short", graph=graph)
    m = out.metrics
    assert m["original_word_count"] == out.original_word_count == len(SOLAR_TEXT.split())
    assert m["summary_word_count"] == len(out.summary.split())
    assert m["compression_ratio"] == out.compression_ratio
    assert 0 < out.compression_ratio < 100
    assert m["num_sentences"] == 10 and m["num_selected_sentences"] == len(out.selected_indices)
    assert out.processing_time >= 0
    assert set(out.timings) == set(out.path)


def test_document_warnings_are_carried_through(graph):
    out = run_summarization(SOLAR_TEXT, "tfidf", warnings=["2 pages were skipped"], graph=graph)
    assert out.warnings == ["2 pages were skipped"]


def test_progress_callback(graph):
    events = []
    run_summarization(" ".join(variants(10)), "bart", "short", graph=graph, on_progress=lambda *e: events.append(e))
    assert any(stage.startswith("level 1") for stage, _, _ in events)
    assert events[-1] == ("final summarization pass", 1, 1)


# ---------------------------------------------------------------- errors


def test_invalid_input_raises_friendly_error_and_stops(graph, stand_in):
    with pytest.raises(EmptyDocumentError):
        run_summarization("   ", "hybrid", graph=graph)
    with pytest.raises(TextTooShortError):
        run_summarization("This sentence is long enough. But the whole text is far too short to summarize.", "bart", graph=graph)
    assert stand_in.calls == []


def test_model_failure_is_reported(graph):
    class BrokenModel(StandInModel):
        def count_tokens(self, text, special_tokens=True):
            raise ModelLoadError()

    with pytest.raises(ModelLoadError):
        run_summarization(SOLAR_TEXT, "bart", graph=graph_with(BrokenModel()))
    # Extractive methods keep working even if the model is broken.
    assert run_summarization(SOLAR_TEXT, "tfidf", graph=graph_with(BrokenModel())).summary


def test_error_details_are_recorded_in_state(stand_in):
    graph = graph_with(stand_in)
    state = graph.invoke({"original_text": "", "method": "tfidf", "summary_length": "short", "started_at": 0.0})
    assert state["path"] == ["preprocess"]
    assert state["errors"][0]["node"] == "preprocess"
    assert state["errors"][0]["type"] == "EmptyDocumentError"


def test_unknown_method_is_rejected(graph):
    with pytest.raises(ValueError):
        run_summarization(SOLAR_TEXT, "gpt", graph=graph)


# ------------------------------------------------------------ real model


@pytest.mark.slow
def test_real_workflow_hybrid():
    out = run_summarization(SOLAR_TEXT, "hybrid", "short")  # default graph with the real BART
    assert out.path[-2:] == ["postprocess", "evaluate"]
    assert "solar" in out.summary.lower()
    assert out.metadata["model"] == "facebook/bart-large-cnn"
