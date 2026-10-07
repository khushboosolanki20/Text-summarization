"""
Loading and running Hugging Face sequence-to-sequence summarization models.

Design goals
------------
* **Load once, lazily.** BART-large has ~400 million parameters (1.6 GB).
  Loading takes several seconds, so a model is loaded the first time it is
  needed and the same tokenizer/model objects are reused for every later
  request. The server starts instantly and extractive methods never wait
  for it.
* **Thread-safe.** FastAPI may handle requests concurrently; a lock ensures
  two simultaneous first requests don't load the model twice.
* **Model-agnostic.** BART, T5 and PEGASUS are all encoder-decoder models
  served by ``AutoModelForSeq2SeqLM``. The only differences are captured in
  ``ModelSpec`` (e.g. T5 expects the prefix "summarize: "), so adding a model
  is a one-line registry entry.
* **Never truncate silently.** ``count_tokens`` lets callers check whether
  input fits the context window; ``generate`` refuses over-long input instead
  of letting the tokenizer cut it off.
"""

import logging
import threading
import time
from dataclasses import dataclass

from app.config import get_settings
from app.errors import ContextWindowExceededError, InferenceError, ModelLoadError
from app.utils.device import cuda_probe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    prefix: str = ""  # text prepended to every input (T5 needs a task prefix)


MODEL_SPECS: dict[str, ModelSpec] = {
    "bart": ModelSpec("bart", "facebook/bart-large-cnn"),
    # Ready for future comparison; downloaded only if actually used.
    "t5": ModelSpec("t5", "google-t5/t5-base", prefix="summarize: "),
    "pegasus": ModelSpec("pegasus", "google/pegasus-cnn_dailymail"),
}


def resolve_device(preference: str) -> str:
    """'auto' / 'cuda' use the GPU only if it actually works (see cuda_probe)."""
    if preference == "cpu":
        return "cpu"
    usable, reason = cuda_probe()
    if not usable and preference == "cuda":
        logger.warning("CUDA requested but unusable (%s); using CPU.", reason)
    return "cuda" if usable else "cpu"


class Seq2SeqSummarizationModel:
    """A tokenizer + model pair, loaded on first use."""

    def __init__(self, spec: ModelSpec, device_preference: str = "auto", use_fp16: bool = False):
        self.spec = spec
        self.device_preference = device_preference
        self.use_fp16 = use_fp16
        self._tokenizer = None
        self._model = None
        self.device: str | None = None
        self.load_seconds: float | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- loading

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # loaded by another thread while we waited
                return
            start = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                from transformers.utils import logging as hf_logging

                hf_logging.disable_progress_bar()  # keep server logs readable
                device = resolve_device(self.device_preference)
                dtype = torch.float16 if (self.use_fp16 and device == "cuda") else torch.float32
                logger.info("Loading %s on %s (%s)...", self.spec.hf_id, device, dtype)
                tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.spec.hf_id, dtype=dtype)
                model.to(device)
                model.eval()  # inference mode: disables dropout
            except Exception as exc:  # network, disk, out-of-memory, missing torch ...
                logger.exception("Failed to load model %s", self.spec.hf_id)
                raise ModelLoadError() from exc
            self._tokenizer, self._model, self.device = tokenizer, model, device
            self.load_seconds = round(time.perf_counter() - start, 2)
            logger.info("Loaded %s in %.1fs", self.spec.hf_id, self.load_seconds)

    @property
    def tokenizer(self):
        self.load()
        return self._tokenizer

    @property
    def model(self):
        self.load()
        return self._model

    # ----------------------------------------------------------- inference

    @property
    def max_input_tokens(self) -> int:
        """Size of the encoder's context window (1,024 tokens for BART)."""
        config = self.model.config
        return int(getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", 512))

    def count_tokens(self, text: str, special_tokens: bool = True) -> int:
        """
        Number of tokens for ``text``. With ``special_tokens`` (default) this is
        exactly what the model sees: prefix + text + begin/end markers. Without,
        it counts the text alone, which is what the chunker needs per sentence.
        """
        if special_tokens:
            text = self.spec.prefix + text
        return len(self.tokenizer(text, add_special_tokens=special_tokens)["input_ids"])

    @property
    def content_token_limit(self) -> int:
        """Tokens available for document text once prefix and special tokens are counted."""
        return self.max_input_tokens - self.count_tokens("")

    def fits(self, text: str) -> bool:
        """Whether ``text`` can be summarized in a single pass (no truncation)."""
        return self.count_tokens(text) <= self.max_input_tokens

    def generate(self, texts: list[str], min_tokens: int, max_tokens: int) -> list[str]:
        """
        Summarize each text with beam search, producing between ``min_tokens``
        and ``max_tokens`` summary tokens. Texts are processed as one padded
        batch, which is much faster on a GPU than one at a time.
        """
        import torch

        settings = get_settings()
        inputs = [self.spec.prefix + t for t in texts]
        # truncation=False: the tokenizer must never cut input off silently.
        encoded = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=False)
        longest = int(encoded["attention_mask"].sum(dim=1).max())
        if longest > self.max_input_tokens:
            raise ContextWindowExceededError(
                f"The text is {longest} tokens long, but the model reads at most {self.max_input_tokens} tokens at once."
            )
        encoded = encoded.to(self.device)
        try:
            with torch.inference_mode():  # no gradient bookkeeping: faster, less memory
                output_ids = self.model.generate(
                    **encoded,
                    do_sample=False,  # deterministic: beam search only, no random sampling
                    num_beams=settings.num_beams,
                    length_penalty=settings.length_penalty,
                    no_repeat_ngram_size=settings.no_repeat_ngram_size,
                    early_stopping=True,
                    # For encoder-decoder models these bound the length of the
                    # generated summary only (not input + output). They
                    # override the fixed 56-142 range bart-large-cnn ships with.
                    min_length=min_tokens,
                    max_length=max_tokens,
                )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise InferenceError(
                "The GPU ran out of memory while summarizing. Try a shorter document or run the server on CPU."
            ) from exc
        except Exception as exc:
            logger.exception("Generation failed with %s", self.spec.hf_id)
            raise InferenceError() from exc
        # Spacing is normalised by the caller (bart.tidy_generated_text).
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


_registry: dict[str, Seq2SeqSummarizationModel] = {}
_registry_lock = threading.Lock()


def get_model(key: str = "bart") -> Seq2SeqSummarizationModel:
    """
    Return the shared model wrapper for ``key``. Creating the wrapper is
    cheap; the weights are only loaded on first use.
    """
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{key}'. Available: {', '.join(MODEL_SPECS)}")
    with _registry_lock:
        if key not in _registry:
            settings = get_settings()
            spec = MODEL_SPECS[key]
            if key == "bart" and settings.abstractive_model_name != spec.hf_id:
                spec = ModelSpec("bart", settings.abstractive_model_name)  # configurable checkpoint
            _registry[key] = Seq2SeqSummarizationModel(spec, settings.device, settings.use_fp16)
        return _registry[key]
