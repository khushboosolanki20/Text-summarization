"""GET /api/health: liveness check used by the frontend status indicator."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings
from app.summarizers.models import get_model
from app.utils.device import get_device_info

router = APIRouter(tags=["health"])


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
