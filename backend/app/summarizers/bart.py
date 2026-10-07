"""
Abstractive summarization with BART (facebook/bart-large-cnn).

What BART is
------------
BART (Lewis et al., 2019) is a Transformer **encoder-decoder**:

* the **encoder** reads the whole input at once. Self-attention lets every
  token look at every other token, so the representation of "it" can draw on
  the noun it refers to three sentences earlier;
* the **decoder** writes the summary one token at a time, attending both to
  what it has written so far and (through cross-attention) to every encoder
  position, i.e. at each step it "looks back" at the most relevant parts of
  the source.

BART is pre-trained as a *denoising autoencoder*: text is corrupted (spans
masked, sentences shuffled) and the model learns to reconstruct the original.
That teaches it both to understand text and to generate fluent text.
``bart-large-cnn`` is then fine-tuned on ~287k CNN/DailyMail article/summary
pairs, so it has learned what a news-style summary looks like.

Decoding
--------
We use **beam search** (4 beams): instead of greedily taking the single most
likely next token, the 4 most likely partial summaries are kept at every step
and the best complete one is returned. Sampling is disabled, so the same input
always produces the same summary. ``no_repeat_ngram_size=3`` forbids repeating
any 3-word sequence, a common failure of neural decoders.

Length control
--------------
The target summary length is ``ratio x input words`` (12/22/32 % for
short/medium/long), converted to tokens (~1.3 tokens per word) and used as
the minimum / maximum summary length, clamped to sensible bounds.

Context window
--------------
BART reads at most 1,024 tokens (~750 words). Input that does not fit is
never truncated: this module raises ``ContextWindowExceededError``, and long
documents are handled by hierarchical chunking (``app.summarizers``, Phase 6).
"""

import re
import time

from app.config import get_settings
from app.preprocessing.cleaner import count_words
from app.summarizers.base import BaseSummarizer, SummaryLength, SummaryResult, length_ratio
from app.summarizers.models import Seq2SeqSummarizationModel, get_model

_SENTENCE_END = re.compile(r"[.!?][\"')\]]?$")
_LAST_SENTENCE_END = re.compile(r"[.!?][\"')\]]?(?=\s|$)")


def token_budget(input_words: int, length: SummaryLength | str) -> tuple[int, int]:
    """
    (min_tokens, max_tokens) of the summary for an input of ``input_words`` words.

    The target is ``ratio x input words`` tokens-converted; the model may stop
    anywhere between 75 % and 125 % of it, so it can end on a natural sentence
    boundary instead of being forced to an exact length.
    """
    settings = get_settings()
    target = input_words * length_ratio(length) * settings.tokens_per_word
    max_tokens = int(min(settings.max_summary_tokens, max(settings.min_summary_tokens + 10, target * 1.25)))
    min_tokens = int(min(max_tokens - 5, max(settings.min_summary_tokens, target * 0.75)))
    return max(1, min_tokens), max_tokens


def tidy_generated_text(text: str) -> tuple[str, bool]:
    """
    Clean decoder output. If generation hit the token limit mid-sentence,
    drop the incomplete trailing fragment, as long as at least one complete
    sentence remains. Returns (text, trimmed?).
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if not text or _SENTENCE_END.search(text):
        return text, False
    ends = list(_LAST_SENTENCE_END.finditer(text))
    if ends:
        return text[: ends[-1].end()].strip(), True
    return text, False


class AbstractiveSummarizer(BaseSummarizer):
    """Summarizer backed by any registered seq2seq model (BART, T5, PEGASUS)."""

    model_key = "bart"

    def __init__(self, model: Seq2SeqSummarizationModel | None = None):
        self._model = model

    @property
    def model(self) -> Seq2SeqSummarizationModel:
        if self._model is None:
            self._model = get_model(self.model_key)  # shared, loaded lazily
        return self._model

    def summarize(self, sentences: list[str], length: SummaryLength | str = SummaryLength.MEDIUM) -> SummaryResult:
        text = " ".join(sentences)
        input_words = count_words(text)
        if not sentences:
            return SummaryResult("", self.name, 0, 0)

        min_tokens, max_tokens = token_budget(input_words, length)
        was_loaded = self.model.is_loaded
        start = time.perf_counter()
        raw = self.model.generate([text], min_tokens, max_tokens)[0]
        inference_seconds = round(time.perf_counter() - start, 3)
        summary, trimmed = tidy_generated_text(raw)

        settings = get_settings()
        return SummaryResult(
            summary=summary,
            method=self.name,
            original_word_count=input_words,
            summary_word_count=count_words(summary),
            metadata={
                "model": self.model.spec.hf_id,
                "device": self.model.device,
                "input_tokens": self.model.count_tokens(text),
                "max_input_tokens": self.model.max_input_tokens,
                "min_summary_tokens": min_tokens,
                "max_summary_tokens": max_tokens,
                "num_beams": settings.num_beams,
                "inference_seconds": inference_seconds,
                # True when this request paid the one-off model loading cost.
                "model_loaded_now": not was_loaded,
                "model_load_seconds": self.model.load_seconds,
                "trimmed_incomplete_sentence": trimmed,
                "chunks": 1,
            },
        )


class BARTSummarizer(AbstractiveSummarizer):
    name = "bart"
    model_key = "bart"
