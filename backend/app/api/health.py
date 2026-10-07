"""GET /api/health: liveness check used by the frontend status indicator."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings
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


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        **get_device_info(),
    )
