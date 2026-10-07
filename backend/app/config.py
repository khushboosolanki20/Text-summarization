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
    # Background summarization jobs that may run at the same time (the rest
    # queue). 1 suits CPU inference; a GPU server could use 2-4.
    max_concurrent_jobs: int = 1
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

    # --- Summary length ----------------------------------------------------
    # Target summary size as a fraction of the input. For extractive methods
    # this is the fraction of sentences selected. Override with JSON, e.g.
    # INTELLISUM_LENGTH_RATIOS='{"short": 0.1, "medium": 0.2, "long": 0.3}'
    length_ratios: dict[str, float] = {"short": 0.12, "medium": 0.22, "long": 0.32}

    # --- Extractive summarization -----------------------------------------
    # A candidate sentence is skipped if its cosine similarity to an already
    # selected sentence exceeds this value (avoids near-duplicate sentences).
    redundancy_threshold: float = 0.8
    # TextRank: probability of following an edge (vs. jumping to a random
    # sentence) in PageRank, and the minimum cosine similarity for two
    # sentences to be connected in the graph.
    textrank_damping: float = 0.85
    textrank_similarity_threshold: float = 0.05
    # Each sentence keeps edges only to its N most similar sentences (a
    # k-nearest-neighbour graph). Bounds graph size on long documents whose
    # sentences all share vocabulary; documents with <= N+1 sentences are
    # unaffected. 0 disables the limit.
    textrank_max_neighbors: int = 50

    # --- Abstractive model -------------------------------------------------
    abstractive_model_name: str = "facebook/bart-large-cnn"
    # "auto" picks CUDA when available, otherwise CPU. Can force "cpu"/"cuda".
    device: str = "auto"
    # Half precision halves GPU memory and speeds up inference, at a small
    # risk of numerical differences. Ignored on CPU.
    use_fp16: bool = False

    # --- Generation (decoding) settings -------------------------------------
    # Defaults follow the generation config facebook/bart-large-cnn was tuned
    # with. Sampling is never used, so output is deterministic.
    num_beams: int = 4
    length_penalty: float = 2.0  # > 1 favours longer beams (counteracts the bias toward short outputs)
    no_repeat_ngram_size: int = 3  # never repeat any 3-gram -> prevents "the the the" loops
    # Summary length in tokens is derived from the input length and the
    # short/medium/long ratio, then clamped to these bounds per generation pass.
    min_summary_tokens: int = 20
    max_summary_tokens: int = 400
    # BART's byte-pair tokenizer produces ~1.3 tokens per English word; used
    # to convert word targets into token budgets.
    tokens_per_word: float = 1.3

    # --- Long documents (hierarchical summarization) ------------------------
    # Maximum tokens of document text per chunk. Kept below BART's 1,024-token
    # window for headroom; smaller chunks give more, shorter partial summaries.
    chunk_max_tokens: int = 900
    # Sentences repeated from the end of one chunk at the start of the next.
    chunk_overlap_sentences: int = 0
    # How the workflow cuts long documents into chunks (LangChain TextSplitter):
    #   "sentence":  balanced chunks of whole sentences (default)
    #   "recursive": LangChain's RecursiveCharacterTextSplitter (may cut sentences)
    chunking_strategy: str = "sentence"
    # Maximum number of chunk -> summarize -> combine rounds before the result
    # is returned as is.
    max_reduction_levels: int = 3
    # Abstractive methods run one model pass per chunk, which is slow on CPU.
    # Longer inputs are rejected with a suggestion to use Hybrid (which first
    # shrinks the document extractively) instead of running for many minutes.
    abstractive_max_input_words: int = 20_000

    # --- Experimental faithfulness check -------------------------------------
    # Flags abstractive summary sentences that may not be supported by the
    # source (a heuristic, not a hallucination detector). Can be disabled.
    faithfulness_check: bool = True
    # A summary sentence is flagged if less than this share of its content
    # words occur anywhere in the source.
    faithfulness_min_coverage: float = 0.6

    # --- Hybrid (TextRank -> BART) -------------------------------------------
    # TextRank selects about this many times the requested summary length,
    # giving BART more material than it needs so it can still choose and
    # rephrase. (A design choice to be tuned on validation data, Phase 14.)
    hybrid_expansion: float = 3.0


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
