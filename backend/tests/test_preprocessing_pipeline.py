import pytest

from app.config import get_settings
from app.documents.loader import load_document
from app.errors import EmptyDocumentError, InputTooLargeError, TextTooShortError
from app.preprocessing.pipeline import preprocess
from tests.conftest import make_pdf


def test_preprocess_valid_text(sample_text):
    result = preprocess(sample_text)
    assert result.sentence_count == 5
    assert result.word_count == len(sample_text.split())
    assert "\n" not in result.cleaned_text  # single paragraph


@pytest.mark.parametrize("text", [None, "", "   \n\t "])
def test_empty_input(text):
    with pytest.raises(EmptyDocumentError):
        preprocess(text)


def test_too_short_input():
    with pytest.raises(TextTooShortError, match="too short"):
        preprocess("This is far too short to summarize. Really.")


def test_too_long_input(monkeypatch):
    monkeypatch.setattr(get_settings(), "max_input_chars", 100)
    with pytest.raises(InputTooLargeError):
        preprocess("word " * 100)


def test_only_fragments_counts_as_empty():
    with pytest.raises(EmptyDocumentError):
        preprocess("1.\n\n2.\n\n3.")


def test_pdf_to_sentences_end_to_end(sample_text):
    # Simulate a hard-wrapped PDF page: lines break mid-sentence.
    wrapped = sample_text.replace("summarization condenses", "summari-\nzation condenses").replace(". ", ".\n")
    document = load_document(make_pdf([wrapped]), "paper.pdf")
    result = preprocess(document.text)
    assert result.sentence_count == 5
    assert "summarization condenses" in result.sentences[0]
