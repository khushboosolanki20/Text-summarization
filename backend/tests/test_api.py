"""
REST API tests: every endpoint, validation, error messages and the async job
flow. The workflow graph is injected with the fast stand-in model (FastAPI
dependency override); one slow test runs the real BART end to end.
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.api import health as health_module
from app.api import summarize as summarize_module
from app.config import get_settings
from app.core.workflow import get_graph
from app.graph.summarization_graph import build_summarization_graph
from app.main import app
from app.summarizers.bart import BARTSummarizer
from tests.conftest import SAMPLE_TEXT, make_docx, make_pdf, make_scanned_pdf
from tests.test_hybrid import StandInModel, variants
from tests.test_tfidf import SOLAR

TEXT = " ".join(SOLAR)
REFERENCE = "Solar capacity grew faster than any other source, driven by falling panel prices."
PAGES = [" ".join(SOLAR[:5]), " ".join(SOLAR[5:])]


@pytest.fixture
def client():
    stand_in_graph = build_summarization_graph(abstractive=BARTSummarizer(model=StandInModel()))
    app.dependency_overrides[get_graph] = lambda: stand_in_graph
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def summarize(client, **body):
    return client.post("/api/summarize/text", json={"text": TEXT, **body})


# ---------------------------------------------------------------- service


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["device"] in {"cpu", "cuda"}


def test_config_lists_methods_and_limits(client):
    body = client.get("/api/config").json()
    assert [m["id"] for m in body["methods"]] == ["tfidf", "textrank", "bart", "hybrid"]
    assert set(body["lengths"]) == {"short", "medium", "long"}
    assert body["supported_file_types"] == [".txt", ".pdf", ".docx"]


def test_warmup_loads_model_in_background(client, monkeypatch):
    model = StandInModel()
    model.is_loaded = False
    loaded = []
    model.load = lambda: loaded.append(True)
    monkeypatch.setattr(health_module, "get_model", lambda key: model)
    assert client.post("/api/warmup").json() == {"status": "loading"}
    time.sleep(0.2)
    assert loaded == [True]
    model.is_loaded = True
    assert client.post("/api/warmup").json() == {"status": "loaded"}


# ----------------------------------------------------------- text summary


@pytest.mark.parametrize("method", ["tfidf", "textrank", "bart", "hybrid"])
def test_every_method(client, method):
    response = summarize(client, method=method, length="medium")
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == method and body["length"] == "medium"
    assert body["summary"]
    assert 0 < body["compression_ratio"] < 100
    assert body["metrics"]["summary_word_count"] == body["summary_word_count"]
    assert len(body["sentences"]) == 10
    assert body["path"][0] == "preprocess" and body["path"][-1] == "evaluate"


def test_default_method_and_length(client):
    body = client.post("/api/summarize/text", json={"text": TEXT}).json()
    assert body["method"] == "hybrid" and body["length"] == "medium"


def test_extractive_response_explains_selection(client):
    body = summarize(client, method="textrank").json()
    selected = [s for s in body["sentences"] if s["selected"]]
    assert body["summary"] == " ".join(s["text"] for s in selected)
    assert all(s["score"] is not None for s in body["sentences"])
    assert body["faithfulness"]["applicable"] is False


def test_rouge_null_without_reference(client):
    metrics = summarize(client, method="tfidf").json()["metrics"]
    assert metrics["rouge1"] is None and metrics["rouge2"] is None and metrics["rougeL"] is None
    assert "reference" in metrics["rouge_note"]


def test_rouge_computed_with_reference(client):
    metrics = summarize(client, method="tfidf", reference_summary=REFERENCE).json()["metrics"]
    assert 0 < metrics["rouge1"] <= 1 and metrics["rouge"]["rougeLsum"]["f1"] > 0
    assert metrics["rouge_note"] is None


def test_long_document_reports_chunks(client):
    body = client.post("/api/summarize/text", json={"text": " ".join(variants(10)), "method": "bart", "length": "short"}).json()
    assert body["strategy"] == "fused"
    assert len(body["intermediate_summaries"]) >= 2
    assert body["metadata"]["reduction_levels"][0]["num_chunks"] == len(body["intermediate_summaries"])


# ------------------------------------------------------- validation errors


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({}, "'text' is required"),
        ({"text": TEXT, "method": "gpt"}, "method: Input should be 'tfidf', 'textrank', 'bart' or 'hybrid'"),
        ({"text": TEXT, "length": "tiny"}, "length:"),
        ({"text": 42}, "text:"),
    ],
    ids=["missing-text", "bad-method", "bad-length", "wrong-type"],
)
def test_malformed_requests_get_readable_messages(client, body, fragment):
    response = client.post("/api/summarize/text", json=body)
    assert response.status_code == 422
    assert fragment in response.json()["detail"]
    assert isinstance(response.json()["detail"], str)


def test_invalid_json(client):
    response = client.post("/api/summarize/text", content="{bad", headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["detail"] == "The request body is not valid JSON."


@pytest.mark.parametrize(
    "text, status, fragment",
    [
        ("", 422, "empty"),
        ("   ", 422, "empty"),
        ("This is far too short to summarize. Really it is.", 422, "too short"),
        ("word " * 200_000, 413, "too long"),
    ],
    ids=["empty", "blank", "too-short", "too-long"],
)
def test_unusable_text(client, text, status, fragment):
    response = client.post("/api/summarize/text", json={"text": text, "method": "tfidf"})
    assert response.status_code == status and fragment in response.json()["detail"]


def test_internal_errors_never_expose_tracebacks(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(summarize_module, "run_summarization", boom)
    response = summarize(client, method="tfidf")
    assert response.status_code == 500
    assert response.json() == {"detail": "An internal error occurred. Please try again."}


# ---------------------------------------------------------------- uploads


def upload(client, data: bytes, filename: str, **form):
    return client.post("/api/summarize/file", files={"file": (filename, data)}, data={"method": "textrank", **form})


def test_pdf_upload_reports_pages(client):
    body = upload(client, make_pdf(PAGES), "report.pdf").json()
    assert body["source"] == {"type": "file", "filename": "report.pdf", "file_type": "pdf", "pages": 2, "title": None}
    assert [s["page"] for s in body["sentences"]] == [1] * 5 + [2] * 5


def test_docx_and_txt_uploads(client):
    assert upload(client, make_docx([TEXT]), "notes.docx").json()["source"]["file_type"] == "docx"
    assert upload(client, TEXT.encode(), "notes.txt").json()["source"]["file_type"] == "txt"


def test_upload_with_reference_and_method(client):
    body = upload(client, TEXT.encode(), "notes.txt", method="bart", length="short", reference_summary=REFERENCE).json()
    assert body["method"] == "bart" and body["length"] == "short" and body["metrics"]["rouge1"] is not None


def test_partially_scanned_pdf_warns(client):
    body = upload(client, make_scanned_pdf(1, text_pages=[TEXT]), "mixed.pdf").json()
    assert "scanned" in body["warnings"][0]


@pytest.mark.parametrize(
    "data, filename, status, fragment",
    [
        (b"abc", "image.png", 415, "Unsupported file type"),
        (b"abc", "old.doc", 415, ".docx"),
        (b"not really a pdf", "fake.pdf", 422, "not a valid PDF"),
        (b"PK\x03\x04 broken", "broken.docx", 422, "corrupted"),
        (b"", "empty.txt", 422, "empty"),
        (make_scanned_pdf(2), "scan.pdf", 422, "OCR"),
    ],
    ids=["png", "legacy-doc", "fake-pdf", "broken-docx", "empty", "scanned-pdf"],
)
def test_bad_uploads(client, data, filename, status, fragment):
    response = upload(client, data, filename)
    assert response.status_code == status
    assert fragment in response.json()["detail"]


def test_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_upload_mb", 0.001)
    response = upload(client, b"x" * 5000, "big.txt")
    assert response.status_code == 413 and "too large" in response.json()["detail"]


def test_missing_file(client):
    response = client.post("/api/summarize/file", data={"method": "tfidf"})
    assert response.status_code == 422 and "'file' is required" in response.json()["detail"]


# --------------------------------------------------------------- evaluate


def test_evaluate(client):
    body = client.post("/api/evaluate", json={"summary": "Solar capacity grew fast.", "reference_summary": REFERENCE}).json()
    assert 0 < body["rouge1"] <= 1 and set(body["rouge"]) == {"rouge1", "rouge2", "rougeL", "rougeLsum"}
    assert body["compression_ratio"] is None and body["faithfulness"] is None


def test_evaluate_with_original_text(client):
    body = client.post(
        "/api/evaluate",
        json={"summary": "Solar installations rose by 15 percent.", "reference_summary": REFERENCE, "original_text": TEXT},
    ).json()
    assert body["compression_ratio"] > 0
    assert body["faithfulness"]["flagged_count"] == 1  # 15 is not in the source (it says 50)


@pytest.mark.parametrize("payload", [{"summary": "x", "reference_summary": " "}, {"summary": " ", "reference_summary": "x"}])
def test_evaluate_rejects_empty_texts(client, payload):
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 422 and "empty" in response.json()["detail"]


# ------------------------------------------------------------- async jobs


def wait_for(client, job_id: str, timeout: float = 30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_async_text_job(client):
    start = client.post("/api/summarize/text/async", json={"text": TEXT, "method": "tfidf"})
    assert start.status_code == 202 and start.json()["status"] in ("queued", "running", "completed")
    job = wait_for(client, start.json()["job_id"])
    assert job["status"] == "completed"
    assert job["progress"] == {"stage": "Completed", "done": 1, "total": 1}
    assert job["result"]["summary"] == summarize(client, method="tfidf").json()["summary"]


def test_async_file_job(client):
    start = client.post("/api/summarize/file/async", files={"file": ("r.pdf", make_pdf(PAGES))}, data={"method": "textrank"})
    job = wait_for(client, start.json()["job_id"])
    assert job["status"] == "completed" and job["result"]["source"]["pages"] == 2


def test_async_job_failure_has_friendly_error(client):
    start = client.post("/api/summarize/text/async", json={"text": "Too short to summarize here. Yes.", "method": "tfidf"})
    job = wait_for(client, start.json()["job_id"])
    assert job["status"] == "failed" and "too short" in job["error"] and job["result"] is None


def test_async_upload_is_validated_immediately(client):
    response = client.post("/api/summarize/file/async", files={"file": ("x.txt", b"")}, data={"method": "tfidf"})
    assert response.status_code == 422  # read and checked before a job is created


def test_unknown_job(client):
    response = client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404 and "No such job" in response.json()["detail"]


# ------------------------------------------- integration with the real model


@pytest.mark.slow
def test_text_to_api_to_real_summary():
    """Integration: text -> API -> LangGraph workflow -> real BART -> response."""
    app.dependency_overrides.clear()
    client = TestClient(app)
    response = client.post("/api/summarize/text", json={"text": SAMPLE_TEXT + " " + TEXT, "method": "hybrid", "length": "short"})
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["model"] == "facebook/bart-large-cnn"
    assert body["summary"] and body["summary_word_count"] < body["original_word_count"]
    assert body["faithfulness"]["applicable"] is True
