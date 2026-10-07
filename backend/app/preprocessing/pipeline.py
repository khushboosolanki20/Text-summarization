"""
The preprocessing pipeline: validate -> clean -> split into sentences -> validate.

The result is computed once per request and shared by every summarizer, so
the (relatively expensive) spaCy pass is never repeated.

Input can be plain text or a list of documents (e.g. one LangChain
``Document`` per PDF page). For documents, each sentence is mapped back to
the page it starts on (``sentence_pages``), so results can say where a
selected sentence came from.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass, field

from app.documents.base import SourceDocument, TextDocument
from app.preprocessing.cleaner import clean_text, count_words
from app.preprocessing.sentence_splitter import split_sentences_with_offsets
from app.preprocessing.validation import validate_content, validate_raw_text

_ENDS_SENTENCE = re.compile(r"[.!?:;][\"')\]]?$")


@dataclass
class PreprocessedText:
    cleaned_text: str
    sentences: list[str]
    word_count: int
    # Page number each sentence starts on (None when the input had no pages).
    sentence_pages: list[int | None] = field(default_factory=list)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


def join_pages(pages: list[str]) -> tuple[str, list[int]]:
    """
    Join cleaned page texts into one text, returning it with the character
    offset at which each page starts.

    Page breaks usually fall between paragraphs (joined with a blank line),
    but a sentence often continues onto the next page. If a page does not end
    a sentence and the next page starts in lowercase, the pages are joined
    with a space (or, for "summari-" + "zation", with nothing), so the
    sentence is not cut into two fragments at the page break.
    """
    text = ""
    starts: list[int] = []
    for page in pages:
        if text and page:
            if text.endswith("-") and page[0].islower():
                text, separator = text[:-1], ""  # word hyphenated across the page break
            elif not _ENDS_SENTENCE.search(text) and page[0].islower():
                separator = " "  # sentence continues on the next page
            else:
                separator = "\n\n"
        else:
            separator = ""
        starts.append(len(text) + len(separator))
        text += separator + page
    return text, starts


def preprocess_documents(documents: list[SourceDocument]) -> PreprocessedText:
    """Clean and segment a list of documents (e.g. pages); raises ``IntelliSumError`` for unusable input."""
    validate_raw_text("\n\n".join(d.page_content for d in documents))
    pages = [clean_text(d.page_content) for d in documents]
    page_numbers = [d.metadata.get("page") for d in documents]

    text, starts = join_pages(pages)
    with_offsets = split_sentences_with_offsets(text)
    sentences = [s for s, _ in with_offsets]
    # bisect: the last page whose start offset is <= the sentence's offset.
    sentence_pages = [page_numbers[bisect_right(starts, offset) - 1] for _, offset in with_offsets]

    word_count = count_words(text)
    validate_content(word_count, len(sentences))
    return PreprocessedText(text, sentences, word_count, sentence_pages)


def preprocess(text: str) -> PreprocessedText:
    """Clean and segment plain text; raises an ``IntelliSumError`` for unusable input."""
    validate_raw_text(text)  # also rejects None before it is wrapped
    return preprocess_documents([TextDocument(text)])
