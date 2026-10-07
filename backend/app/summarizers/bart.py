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
BART reads at most 1,024 tokens (~750 words). Input that fits is summarized
in one pass. Longer input is **never truncated**: it goes through
hierarchical chunked summarization (``app.summarizers.long_document``).
"""

import time

from app.config import get_settings
from app.errors import InputTooLargeError
from app.preprocessing.cleaner import count_words
from app.summarizers.base import BaseSummarizer, SummaryLength, SummaryResult, length_ratio
from app.summarizers.generation import generate_summary, tidy_generated_text, tokens_for_words
from app.summarizers.long_document import ProgressCallback, hierarchical_summarize
from app.summarizers.models import Seq2SeqSummarizationModel, get_model

__all__ = ["AbstractiveSummarizer", "BARTSummarizer", "tidy_generated_text", "token_budget"]


def check_abstractive_input_size(input_words: int, method_name: str) -> None:
    """Reject inputs that would need too many slow model passes (see Settings)."""
    limit = get_settings().abstractive_max_input_words
    if input_words > limit:
        raise InputTooLargeError(
            f"The document is too long for {method_name.upper()} ({input_words:,} words; limit {limit:,}). "
            "Use Hybrid, which first selects the key sentences, or an extractive method."
        )


def token_budget(input_words: int, length: SummaryLength | str) -> tuple[int, int]:
    """(min_tokens, max_tokens) of a single-pass summary for an input of ``input_words`` words."""
    return tokens_for_words(input_words * length_ratio(length))


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

    def summarize(
        self,
        sentences: list[str],
        length: SummaryLength | str = SummaryLength.MEDIUM,
        on_progress: ProgressCallback | None = None,
    ) -> SummaryResult:
        input_words = sum(count_words(s) for s in sentences)
        return self.summarize_to_target(sentences, input_words * length_ratio(length), on_progress)

    def summarize_to_target(
        self,
        sentences: list[str],
        target_words: float,
        on_progress: ProgressCallback | None = None,
    ) -> SummaryResult:
        """
        Summarize ``sentences`` to about ``target_words`` words: one pass if
        the text fits the context window, hierarchical chunking otherwise.
        Hybrid calls this directly so the target can be relative to the
        original document rather than to the (already reduced) input.
        """
        settings = get_settings()
        text = " ".join(sentences)
        input_words = count_words(text)
        if not sentences:
            return SummaryResult("", self.name, 0, 0)
        check_abstractive_input_size(input_words, self.name)

        was_loaded = self.model.is_loaded
        start = time.perf_counter()
        input_tokens = self.model.count_tokens(text)  # also triggers lazy loading
        metadata: dict = {
            "model": self.model.spec.hf_id,
            "device": self.model.device,
            "input_tokens": input_tokens,
            "max_input_tokens": self.model.max_input_tokens,
            "num_beams": settings.num_beams,
            "target_words": round(target_words),
        }

        if input_tokens <= self.model.max_input_tokens:
            # Short document: one pass over the whole text.
            summary, trimmed, (min_tokens, max_tokens) = generate_summary(self.model, text, target_words)
            metadata.update(
                strategy="single_pass",
                chunks=1,
                min_summary_tokens=min_tokens,
                max_summary_tokens=max_tokens,
                trimmed_incomplete_sentence=trimmed,
            )
        else:
            # Long document: chunk -> summarize chunks -> combine -> final pass.
            result = hierarchical_summarize(self.model, sentences, target_words, on_progress)
            summary = result.summary
            metadata.update(
                strategy=result.strategy,
                chunks=len(result.levels[0].chunks),
                reduction_levels=[level.describe() for level in result.levels],
                intermediate_summaries=result.intermediate_summaries,
                trimmed_incomplete_sentence=result.final_pass_trimmed,
            )

        metadata.update(
            inference_seconds=round(time.perf_counter() - start, 3),
            # True when this request paid the one-off model loading cost.
            model_loaded_now=not was_loaded,
            model_load_seconds=self.model.load_seconds,
        )
        return SummaryResult(
            summary=summary,
            method=self.name,
            original_word_count=input_words,
            summary_word_count=count_words(summary),
            metadata=metadata,
        )


class BARTSummarizer(AbstractiveSummarizer):
    name = "bart"
    model_key = "bart"
