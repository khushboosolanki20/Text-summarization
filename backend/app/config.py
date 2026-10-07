"""
Central application configuration.

All tunable values live here so that experiments and deployments can change
behaviour through environment variables (prefixed ``INTELLISUM_``) or a
``backend/.env`` file without touching code. Example::

    INTELLISUM_DEVICE=cpu
    INTELLISUM_MAX_UPLOAD_MB=5
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INTELLISUM_", env_file=".env", extra="ignore")

    # --- Application -------------------------------------------------------
    app_name: str = "IntelliSum"
    app_version: str = "0.1.0"
    # Origins allowed to call the API directly (the Vite dev server proxies
    # /api, but a production build served elsewhere needs CORS).
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # --- Input limits ------------------------------------------------------
    max_upload_mb: float = 10.0
    max_input_chars: int = 500_000
    # Below these thresholds a summary is meaningless, so the input is rejected.
    min_input_words: int = 40
    min_input_sentences: int = 3

    # --- Preprocessing -----------------------------------------------------
    spacy_model: str = "en_core_web_sm"
    # How sentence boundaries are detected:
    #   "parser": spaCy dependency parser (most accurate, default)
    #   "senter": spaCy's small statistical sentence segmenter (~4x faster)
    #   "rule":   punctuation rules only (no model download needed)
    sentence_segmenter: str = "parser"
    # Fragments shorter than this (headings, "Yes!", page labels) are not
    # treated as candidate sentences for summarization.
    min_sentence_words: int = 3

    # --- Abstractive model -------------------------------------------------
    abstractive_model_name: str = "facebook/bart-large-cnn"
    # "auto" picks CUDA when available, otherwise CPU. Can force "cpu"/"cuda".
    device: str = "auto"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
