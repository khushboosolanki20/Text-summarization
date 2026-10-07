"""
FastAPI application entry point.

Run from the ``backend/`` directory with::

    uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health
from app.config import get_settings
from app.errors import IntelliSumError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("intellisum")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.app_version,
        description="Intelligent text summarization using classical NLP and Transformers.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(IntelliSumError)
    async def intellisum_error_handler(request: Request, exc: IntelliSumError) -> JSONResponse:
        # Expected, user-caused errors: return the friendly message as-is.
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log the full traceback server-side, but never leak it to the client.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "An internal error occurred. Please try again."})

    app.include_router(health.router, prefix="/api")
    return app


app = create_app()
