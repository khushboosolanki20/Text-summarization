"""
FastAPI application entry point.

Run from the ``backend/`` directory with::

    uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import evaluate, health, summarize
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

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default is a list of JSON objects; the UI expects one
        # readable sentence, e.g. "method: Input should be 'tfidf', 'textrank', ...".
        return JSONResponse(status_code=422, content={"detail": describe_validation_error(exc)})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log the full traceback server-side, but never leak it to the client.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "An internal error occurred. Please try again."})

    app.include_router(health.router, prefix="/api")
    app.include_router(summarize.router, prefix="/api")
    app.include_router(evaluate.router, prefix="/api")
    return app


def describe_validation_error(exc: RequestValidationError) -> str:
    """Turn Pydantic's error list into a short human-readable message."""
    messages = []
    for error in exc.errors()[:3]:
        location = [str(part) for part in error.get("loc", ()) if part not in ("body", "query", "path", "form")]
        field = ".".join(location)
        if error.get("type") == "missing":
            messages.append(f"'{field}' is required." if field else "A required value is missing.")
        elif error.get("type") == "json_invalid":
            messages.append("The request body is not valid JSON.")
        else:
            messages.append(f"{field}: {error.get('msg')}" if field else str(error.get("msg")))
    return " ".join(messages) or "The request is invalid."


app = create_app()
