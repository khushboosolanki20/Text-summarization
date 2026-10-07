"""
Summarization endpoints.

Synchronous (the response is the summary):
    POST /api/summarize/text         JSON body
    POST /api/summarize/file         multipart upload (TXT, PDF, DOCX)

Asynchronous, with progress (for long documents):
    POST /api/summarize/text/async   -> {"job_id": ...}
    POST /api/summarize/file/async   -> {"job_id": ...}
    GET  /api/jobs/{job_id}          -> status, progress, result or error

All endpoints run the same LangGraph workflow (``app.core.workflow``).
Endpoint functions are plain ``def`` so FastAPI runs them in its thread pool
and CPU-heavy summarization does not block the server's event loop.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.schemas import JobResponse, SummarizeTextRequest, SummaryResponse
from app.config import get_settings
from app.core.jobs import Job, get_job_manager
from app.core.workflow import get_graph, run_summarization
from app.documents.langchain_loaders import UploadedFileLoader
from app.errors import EmptyDocumentError, InputTooLargeError
from app.summarizers.base import SummaryLength, SummaryMethod

router = APIRouter(tags=["summarization"])

Graph = Annotated[object, Depends(get_graph)]  # overridable in tests


def _summarize_text(request: SummarizeTextRequest, graph, on_progress=None) -> SummaryResponse:
    out = run_summarization(
        request.text,
        request.method,
        request.length,
        reference_summary=request.reference_summary,
        on_progress=on_progress,
        graph=graph,
    )
    return SummaryResponse.from_output(out, {"type": "text"})


def _read_upload(file: UploadFile) -> bytes:
    """Read an upload, refusing to load more than the size limit into memory."""
    limit = int(get_settings().max_upload_mb * 1024 * 1024)
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise InputTooLargeError(f"The file is too large. The limit is {get_settings().max_upload_mb:g} MB.")
    if not data:
        raise EmptyDocumentError("The uploaded file is empty.")
    return data


def _summarize_file(
    data: bytes,
    filename: str,
    method: SummaryMethod,
    length: SummaryLength,
    reference_summary: str | None,
    graph,
    on_progress=None,
) -> SummaryResponse:
    loader = UploadedFileLoader(data, filename)
    documents = loader.load()  # one LangChain Document per PDF page; validates the file
    out = run_summarization(
        documents=documents,
        method=method,
        length=length,
        reference_summary=reference_summary or None,
        warnings=loader.warnings,
        on_progress=on_progress,
        graph=graph,
    )
    first = documents[0].metadata if documents else {}
    source = {
        "type": "file",
        "filename": filename,
        "file_type": first.get("file_type"),
        "pages": first.get("total_pages"),
        "title": first.get("title"),
    }
    return SummaryResponse.from_output(out, source)


def _job_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )


# ------------------------------------------------------------------ synchronous


@router.post("/summarize/text", response_model=SummaryResponse, summary="Summarize text")
def summarize_text(request: SummarizeTextRequest, graph: Graph) -> SummaryResponse:
    """
    Summarize pasted text with the chosen method and length. ROUGE is
    computed only if `reference_summary` is provided; otherwise the ROUGE
    fields are `null` and `metrics.rouge_note` explains why.
    """
    return _summarize_text(request, graph)


@router.post("/summarize/file", response_model=SummaryResponse, summary="Summarize an uploaded file")
def summarize_file(
    graph: Graph,
    file: UploadFile = File(..., description="A .txt, .pdf or .docx file"),
    method: SummaryMethod = Form(SummaryMethod.HYBRID),
    length: SummaryLength = Form(SummaryLength.MEDIUM),
    reference_summary: str | None = Form(None),
) -> SummaryResponse:
    """
    Summarize a TXT, PDF or DOCX file. PDFs are read page by page, so each
    sentence in the response carries its page number. Scanned (image-only)
    PDFs, corrupted files and unsupported types are rejected with a clear message.
    """
    return _summarize_file(_read_upload(file), file.filename or "", method, length, reference_summary, graph)


# ----------------------------------------------------------------- asynchronous


@router.post("/summarize/text/async", response_model=JobResponse, status_code=202, summary="Start a text summarization job")
def summarize_text_async(request: SummarizeTextRequest, graph: Graph) -> JobResponse:
    """Start summarizing in the background; poll `GET /api/jobs/{job_id}` for progress and the result."""
    job = get_job_manager().submit(lambda progress: _summarize_text(request, graph, progress))
    return _job_response(job)


@router.post("/summarize/file/async", response_model=JobResponse, status_code=202, summary="Start a file summarization job")
def summarize_file_async(
    graph: Graph,
    file: UploadFile = File(...),
    method: SummaryMethod = Form(SummaryMethod.HYBRID),
    length: SummaryLength = Form(SummaryLength.MEDIUM),
    reference_summary: str | None = Form(None),
) -> JobResponse:
    """Like `/summarize/file`, but returns a job id immediately. The file is read before returning."""
    data, filename = _read_upload(file), file.filename or ""
    job = get_job_manager().submit(
        lambda progress: _summarize_file(data, filename, method, length, reference_summary, graph, progress)
    )
    return _job_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse, summary="Get job status, progress and result")
def get_job(job_id: str) -> JobResponse:
    return _job_response(get_job_manager().get(job_id))
