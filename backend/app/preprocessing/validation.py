"""Input validation shared by the text and file entry points."""

from app.config import get_settings
from app.errors import EmptyDocumentError, InputTooLargeError, TextTooShortError


def validate_raw_text(text: str | None) -> str:
    """Reject missing, blank or oversized input before any processing is done."""
    settings = get_settings()
    if text is None or not text.strip():
        raise EmptyDocumentError()
    if len(text) > settings.max_input_chars:
        raise InputTooLargeError(
            f"The text is too long ({len(text):,} characters). "
            f"The maximum is {settings.max_input_chars:,} characters."
        )
    return text


def validate_content(word_count: int, sentence_count: int) -> None:
    """Reject text that is too short to produce a meaningful summary."""
    settings = get_settings()
    if word_count == 0 or sentence_count == 0:
        raise EmptyDocumentError("No readable sentences were found in the document.")
    if word_count < settings.min_input_words or sentence_count < settings.min_input_sentences:
        raise TextTooShortError(
            f"The text is too short to summarize ({word_count} words, {sentence_count} sentences). "
            f"Please provide at least {settings.min_input_words} words "
            f"in {settings.min_input_sentences} or more sentences."
        )
