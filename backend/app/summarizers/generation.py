"""
Helpers shared by every abstractive generation pass: turning a target length
in words into a token budget, running one pass, and cleaning the output.
"""

import re
from typing import Protocol

from app.config import get_settings

_SENTENCE_END = re.compile(r"[.!?][\"')\]]?$")
_LAST_SENTENCE_END = re.compile(r"[.!?][\"')\]]?(?=\s|$)")


class TextGenerator(Protocol):
    """
    What the abstractive code needs from a model. ``Seq2SeqSummarizationModel``
    implements it; keeping the orchestration code on this interface lets it
    run with any seq2seq model (or a lightweight stand-in in unit tests).
    """

    @property
    def max_input_tokens(self) -> int: ...

    @property
    def content_token_limit(self) -> int: ...

    def count_tokens(self, text: str, special_tokens: bool = True) -> int: ...

    def fits(self, text: str) -> bool: ...

    def generate(self, texts: list[str], min_tokens: int, max_tokens: int) -> list[str]: ...


def tokens_for_words(target_words: float) -> tuple[int, int]:
    """
    (min_tokens, max_tokens) for a summary of about ``target_words`` words.

    Words are converted to tokens (~1.3 per word) and the model may stop
    anywhere between 75 % and 125 % of the target, so it can end on a natural
    sentence boundary, within configured absolute bounds.
    """
    settings = get_settings()
    target = target_words * settings.tokens_per_word
    max_tokens = int(min(settings.max_summary_tokens, max(settings.min_summary_tokens + 10, target * 1.25)))
    min_tokens = int(min(max_tokens - 5, max(settings.min_summary_tokens, target * 0.75)))
    return max(1, min_tokens), max_tokens


def max_single_pass_words() -> float:
    """Longest summary (in words) one generation pass can produce."""
    settings = get_settings()
    return settings.max_summary_tokens / settings.tokens_per_word


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


def generate_summary(generator: TextGenerator, text: str, target_words: float) -> tuple[str, bool, tuple[int, int]]:
    """One generation pass. Returns (summary, trimmed?, (min_tokens, max_tokens))."""
    budget = tokens_for_words(target_words)
    raw = generator.generate([text], *budget)[0]
    summary, trimmed = tidy_generated_text(raw)
    return summary, trimmed, budget
