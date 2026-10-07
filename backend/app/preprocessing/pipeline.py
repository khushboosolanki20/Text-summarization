"""
The preprocessing pipeline: validate -> clean -> split into sentences -> validate.

The result is computed once per request and shared by every summarizer, so
the (relatively expensive) spaCy pass is never repeated.
"""

from dataclasses import dataclass

from app.preprocessing.cleaner import clean_text, count_words
from app.preprocessing.sentence_splitter import split_sentences
from app.preprocessing.validation import validate_content, validate_raw_text


@dataclass
class PreprocessedText:
    cleaned_text: str
    sentences: list[str]
    word_count: int

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


def preprocess(text: str) -> PreprocessedText:
    """Clean and segment ``text``; raises an ``IntelliSumError`` for unusable input."""
    validate_raw_text(text)
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    word_count = count_words(cleaned)
    validate_content(word_count, len(sentences))
    return PreprocessedText(cleaned_text=cleaned, sentences=sentences, word_count=word_count)
