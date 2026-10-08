"""
Edge cases and failure paths found by measuring coverage: fallbacks that
rarely trigger in normal use but must behave sensibly when they do.
"""

import time

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from app.api import health as health_module
from app.core.jobs import JOB_TTL_SECONDS, JobManager
from app.errors import ModelLoadError
from app.evaluation import faithfulness as faithfulness_module
from app.evaluation.faithfulness import check_faithfulness
from app.main import app
from app.preprocessing.chunker import chunk_sentences
from app.summarizers.bart import BARTSummarizer
from app.summarizers.hybrid import HybridSummarizer
from app.summarizers.textrank import TextRankSummarizer
from app.utils import device as device_module
from tests.test_hybrid import StandInModel
from tests.test_tfidf import OFF_TOPIC, SOLAR


def wait(job, timeout=5):
    deadline = time.time() + timeout
    while job.status not in ("completed", "failed") and time.time() < deadline:
        time.sleep(0.02)
    return job


# ------------------------------------------------------------------- jobs


def test_unexpected_job_crash_gives_generic_message():
    manager = JobManager()

    def crash(progress):
        raise RuntimeError("internal detail that must not leak")

    job = wait(manager.submit(crash))
    assert job.status == "failed"
    assert job.error == "An internal error occurred. Please try again."
    assert "internal detail" not in job.error


def test_finished_jobs_expire():
    manager = JobManager()
    job = wait(manager.submit(lambda progress: "done"))
    job.finished_at = time.time() - JOB_TTL_SECONDS - 1  # pretend it finished long ago
    wait(manager.submit(lambda progress: "next"))  # submitting triggers cleanup
    with pytest.raises(Exception, match="No such job"):
        manager.get(job.id)


def test_progress_is_recorded():
    manager = JobManager()

    def task(progress):
        progress("working", 1, 2)
        time.sleep(0.05)
        return "ok"

    job = manager.submit(task)
    time.sleep(0.02)
    assert job.progress in ({"stage": "working", "done": 1, "total": 2}, None) or job.status == "completed"
    assert wait(job).progress == {"stage": "Completed", "done": 1, "total": 1}


# --------------------------------------------------------------- TextRank


def test_textrank_falls_back_to_degree_centrality(monkeypatch):
    def no_convergence(*args, **kwargs):
        raise nx.PowerIterationFailedConvergence(200)

    monkeypatch.setattr(nx, "pagerank", no_convergence)
    result = TextRankSummarizer().summarize(SOLAR, "long")
    assert result.metadata["graph"]["converged"] is False
    assert sum(result.sentence_scores) == pytest.approx(1.0, abs=1e-4)
    assert not OFF_TOPIC & set(result.selected_indices)  # still a sensible ranking


# ---------------------------------------------------------------- chunker


def test_chunks_are_resplit_when_joining_adds_tokens():
    # A counter where joining costs extra (each space counts): sentence sizes
    # alone under-estimate the joined chunk, so verification must split again.
    def chars(text: str) -> int:
        return len(text)

    sentences = [f"Sentence number {i} has some words." for i in range(30)]
    chunks = chunk_sentences(sentences, 120, chars)
    assert all(chars(c.text) <= 120 for c in chunks)
    assert [s for c in chunks for s in c.sentences] == sentences


def test_overlap_is_dropped_when_there_is_no_room():
    # Each sentence is 60% of the limit: a carried-over sentence plus a new one
    # would not fit, so chunks must start fresh rather than stall.
    sentences = [" ".join(["w"] * 6) for _ in range(5)]
    chunks = chunk_sentences(sentences, 10, lambda t: len(t.split()), overlap_sentences=1)
    assert all(c.token_count <= 10 for c in chunks)
    assert [s for c in chunks for s in c.sentences] == sentences


def test_long_sentence_split_with_nonlinear_counter():
    long_sentence = " ".join(f"tok{i}" for i in range(200))

    def costly(text: str) -> int:  # superlinear: long texts cost disproportionately more
        n = len(text.split())
        return n + n * n // 50

    chunks = chunk_sentences([long_sentence], 40, costly)
    assert all(costly(c.text) <= 40 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == long_sentence.split()


# ----------------------------------------------------------------- Hybrid


def test_hybrid_keeps_top_sentence_even_if_it_exceeds_the_window():
    giant = "Solar " + " ".join(["capacity growth"] * 400) + "."
    model = StandInModel()
    summarizer = HybridSummarizer(abstractive=BARTSummarizer(model=model))
    selected, _, _ = summarizer.select_sentences([giant, "Short unrelated line here."], 100, token_cap=50)
    assert selected  # never an empty selection
    assert len(selected) == 1


def test_hybrid_reports_progress_when_called_directly():
    events = []
    summarizer = HybridSummarizer(abstractive=BARTSummarizer(model=StandInModel()))
    summarizer.summarize(SOLAR, "short", on_progress=lambda *e: events.append(e))
    assert ("selecting key sentences (TextRank)", 1, 1) in events


# ----------------------------------------------------------- faithfulness


def test_faithfulness_of_empty_summary():
    report = check_faithfulness("", SOLAR)
    assert report["flagged_count"] == 0 and report["sentences"] == []


def test_faithfulness_without_vocabulary():
    # Only stop words: TF-IDF has no vocabulary, so similarity is unavailable.
    report = check_faithfulness("It is what it is.", ["This is that.", "It was there."])
    assert report["sentences"][0]["similarity"] == 0.0


def test_faithfulness_when_spacy_model_is_missing(monkeypatch):
    import spacy

    def missing(*args, **kwargs):
        raise OSError("model not installed")

    faithfulness_module._ner_pipeline.cache_clear()
    monkeypatch.setattr(spacy, "load", missing)
    try:
        assert faithfulness_module._ner_pipeline() is None
        report = check_faithfulness("The World Bank praised solar growth.", SOLAR)
        assert report["sentences"][0]["unsupported_entities"] == []
    finally:
        faithfulness_module._ner_pipeline.cache_clear()


# ----------------------------------------------------------------- warmup


def test_warmup_survives_model_load_failure(monkeypatch):
    class Broken:
        is_loaded = False

        class spec:
            hf_id = "broken/model"

        def load(self):
            raise ModelLoadError()

    monkeypatch.setattr(health_module, "get_model", lambda key: Broken())
    client = TestClient(app)
    assert client.post("/api/warmup").json() == {"status": "loading"}
    time.sleep(0.1)
    assert client.get("/api/health").status_code == 200  # server unaffected


# ---------------------------------------------------- GPU detection (device)


class FakeCuda:
    def __init__(self, available: bool):
        self.available = available

    def is_available(self):
        return self.available

    def get_device_name(self, index):
        return "Fake GPU"


@pytest.fixture
def fake_torch(monkeypatch):
    import torch

    device_module.cuda_probe.cache_clear()
    device_module.get_device_info.cache_clear()
    yield torch, monkeypatch
    device_module.cuda_probe.cache_clear()
    device_module.get_device_info.cache_clear()


def test_no_gpu_means_cpu(fake_torch):
    torch, monkeypatch = fake_torch
    monkeypatch.setattr(torch, "cuda", FakeCuda(False))
    assert device_module.cuda_probe() == (False, "no CUDA-capable GPU detected")
    assert device_module.get_device_info()["device"] == "cpu"


def test_gpu_reported_available_but_unusable_falls_back_to_cpu(fake_torch):
    # The situation on the development laptop: an outdated driver makes
    # torch.cuda.is_available() True while every GPU operation fails.
    torch, monkeypatch = fake_torch
    monkeypatch.setattr(torch, "cuda", FakeCuda(True))

    def failing_ones(*args, **kwargs):
        raise RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")

    monkeypatch.setattr(torch, "ones", failing_ones)
    usable, reason = device_module.cuda_probe()
    assert usable is False and "busy or unavailable" in reason
    info = device_module.get_device_info()
    assert info["cuda_available"] is False and info["device"] == "cpu"
    assert info["gpu_name"] == "Fake GPU" and "busy or unavailable" in info["gpu_problem"]


def test_working_gpu_is_used(fake_torch):
    torch, monkeypatch = fake_torch
    monkeypatch.setattr(torch, "cuda", FakeCuda(True))

    class Tensor:
        def __mul__(self, other):
            return self

        def sum(self):
            return self

        def item(self):
            return 4.0

    monkeypatch.setattr(torch, "ones", lambda *a, **k: Tensor())
    assert device_module.cuda_probe() == (True, None)
    assert device_module.get_device_info()["device"] == "cuda"
