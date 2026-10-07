"""
Service endpoints:

    GET  /api/health   liveness, hardware and model status (frontend status pill)
    GET  /api/config   methods, lengths and limits, so the UI never hard-codes them
    POST /api/warmup   start loading the BART model in the background
"""

import threading

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.schemas import ConfigResponse
from app.config import get_settings
from app.documents.loader import SUPPORTED_EXTENSIONS
from app.errors import IntelliSumError
from app.summarizers.base import SummaryMethod
from app.summarizers.models import get_model
from app.utils.device import get_device_info

router = APIRouter(tags=["service"])

METHOD_INFO = {
    SummaryMethod.TFIDF: ("TF-IDF", "extractive", "Ranks sentences by similarity to the document's TF-IDF centroid."),
    SummaryMethod.TEXTRANK: ("TextRank", "extractive", "Ranks sentences with PageRank over a sentence-similarity graph."),
    SummaryMethod.BART: ("BART", "abstractive", "A Transformer (bart-large-cnn) writes a new summary."),
    SummaryMethod.HYBRID: ("Hybrid", "hybrid", "TextRank selects key sentences, BART rewrites them."),
}


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    torch_installed: bool
    cuda_available: bool
    device: str
    gpu_name: str | None
    gpu_problem: str | None
    abstractive_model: str
    # False until the first abstractive request (the model loads lazily), so
    # the UI can warn that the first BART/Hybrid summary takes longer.
    abstractive_model_loaded: bool


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    model = get_model("bart")
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        abstractive_model=model.spec.hf_id,
        abstractive_model_loaded=model.is_loaded,
        **get_device_info(),
    )


@router.get("/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    settings = get_settings()
    return ConfigResponse(
        methods=[
            {"id": method.value, "name": name, "type": kind, "description": description}
            for method, (name, kind, description) in METHOD_INFO.items()
        ],
        lengths=settings.length_ratios,
        supported_file_types=list(SUPPORTED_EXTENSIONS),
        max_upload_mb=settings.max_upload_mb,
        max_input_chars=settings.max_input_chars,
        min_input_words=settings.min_input_words,
        abstractive_max_input_words=settings.abstractive_max_input_words,
        abstractive_model=settings.abstractive_model_name,
        faithfulness_check=settings.faithfulness_check,
    )


class WarmupResponse(BaseModel):
    status: str  # "loaded" | "loading"


@router.post("/warmup", response_model=WarmupResponse, status_code=202)
def warmup() -> WarmupResponse:
    """
    Start loading the abstractive model in the background, so the first BART
    or Hybrid summary does not have to wait for it. Safe to call repeatedly.
    """
    model = get_model("bart")
    if model.is_loaded:
        return WarmupResponse(status="loaded")

    def load() -> None:
        try:
            model.load()  # thread-safe: concurrent calls load only once
        except IntelliSumError:
            pass  # reported to the user by the next abstractive request

    threading.Thread(target=load, name="model-warmup", daemon=True).start()
    return WarmupResponse(status="loading")
