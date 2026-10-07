from io import BytesIO

import pymupdf
import pytest

from app.documents.docx_loader import load_docx
from app.documents.loader import load_document
from app.documents.pdf_loader import load_pdf, remove_repeated_headers_footers
from app.documents.text_loader import decode_text, load_text
from app.errors import (
    CorruptedFileError,
    EmptyDocumentError,
    InputTooLargeError,
    OCRRequiredError,
    PasswordProtectedError,
    UnsupportedFileTypeError,
)
from tests.conftest import SAMPLE_TEXT, make_docx, make_pdf, make_scanned_pdf

# ---------------------------------------------------------------- plain text


def test_text_utf8():
    doc = load_text("Café résumé naïve.".encode("utf-8"))
    assert doc.text == "Café résumé naïve."
    assert doc.file_type == "txt"


def test_text_utf8_bom_is_stripped():
    assert decode_text(b"\xef\xbb\xbfHello") == "Hello"


def test_text_utf16_with_bom():
    assert decode_text("Hello world".encode("utf-16")) == "Hello world"


def test_text_cp1252_fallback():
    assert decode_text("It’s “quoted”".encode("cp1252")) == "It’s “quoted”"


def test_binary_file_named_txt_is_rejected():
    with pytest.raises(CorruptedFileError):
        load_text(b"\x89PNG\r\n\x1a\n\x00\x00\x00binary")


# ----------------------------------------------------------------------- PDF


def test_pdf_text_is_extracted_page_by_page():
    doc = load_pdf(make_pdf(["First page talks about summarization.", "Second page covers evaluation."]))
    assert doc.page_count == 2
    assert "First page" in doc.pages[0]
    assert "Second page" in doc.pages[1]
    assert "summarization" in doc.text and "evaluation" in doc.text
    assert doc.warnings == []


TOPICS = ["tokenization methods", "graph ranking", "neural attention", "evaluation metrics"]


def test_pdf_repeated_header_and_footer_removed():
    pages = [
        f"Journal of NLP Research\nThis page discusses {topic} in some depth.\nPage {i} of 4"
        for i, topic in enumerate(TOPICS, start=1)
    ]
    doc = load_pdf(make_pdf(pages))
    assert "Journal of NLP Research" not in doc.text
    assert "Page 1 of 4" not in doc.text
    for topic in TOPICS:
        assert f"This page discusses {topic}" in doc.text


def test_repeated_phrase_in_body_is_kept():
    # The same sentence in the middle of every page is content, not a header.
    pages = [f"Header {t}\nIntro line {t}\nRepeated body sentence.\nOutro line {t}\nFooter {t}" for t in TOPICS]
    cleaned = remove_repeated_headers_footers(pages)
    assert all("Repeated body sentence." in p for p in cleaned)


def test_header_removal_skips_short_documents():
    pages = ["Header\nbody one", "Header\nbody two"]
    assert remove_repeated_headers_footers(pages) == pages


def test_scanned_pdf_requires_ocr():
    with pytest.raises(OCRRequiredError):
        load_pdf(make_scanned_pdf(n_pages=2))


def test_partially_scanned_pdf_warns():
    doc = load_pdf(make_scanned_pdf(n_pages=1, text_pages=[SAMPLE_TEXT]))
    assert "Automatic text summarization" in doc.text
    assert len(doc.warnings) == 1
    assert "1 of 2 pages" in doc.warnings[0]


def test_blank_pdf_is_empty():
    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(EmptyDocumentError):
        load_pdf(doc.tobytes())


def test_password_protected_pdf():
    doc = pymupdf.open(stream=make_pdf([SAMPLE_TEXT]), filetype="pdf")
    buffer = BytesIO()
    doc.save(buffer, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    with pytest.raises(PasswordProtectedError):
        load_pdf(buffer.getvalue())


def test_corrupted_pdf():
    with pytest.raises(CorruptedFileError):
        load_pdf(b"%PDF-1.7\nthis is not really a pdf")


# ---------------------------------------------------------------------- DOCX


def test_docx_paragraphs_extracted_in_order():
    doc = load_docx(make_docx(["First paragraph.", "", "Second paragraph."]))
    assert doc.text == "First paragraph.\n\nSecond paragraph."
    assert doc.metadata["paragraph_count"] == 2


def test_docx_tables_produce_warning():
    doc = load_docx(make_docx(["Some paragraph text."], with_table=True))
    assert "Quarter" not in doc.text
    assert "1 table(s)" in doc.warnings[0]


def test_empty_docx():
    with pytest.raises(EmptyDocumentError):
        load_docx(make_docx([]))


def test_corrupted_docx():
    with pytest.raises(CorruptedFileError):
        load_docx(b"PK\x03\x04 truncated zip archive")


# ------------------------------------------------------------- dispatcher


@pytest.mark.parametrize(
    "filename, data",
    [
        ("notes.txt", SAMPLE_TEXT.encode()),
        ("paper.PDF", make_pdf([SAMPLE_TEXT])),
        ("report.docx", make_docx([SAMPLE_TEXT])),
    ],
    ids=["txt", "pdf-uppercase-ext", "docx"],
)
def test_dispatch_by_extension(filename, data):
    doc = load_document(data, filename)
    assert "Automatic text summarization" in doc.text


@pytest.mark.parametrize("filename", ["image.png", "archive.zip", "noextension", ""])
def test_unsupported_extension(filename):
    with pytest.raises(UnsupportedFileTypeError):
        load_document(b"data", filename)


def test_legacy_doc_has_specific_message():
    with pytest.raises(UnsupportedFileTypeError, match=r"\.docx"):
        load_document(b"data", "old.doc")


def test_empty_upload():
    with pytest.raises(EmptyDocumentError):
        load_document(b"", "empty.pdf")


def test_oversized_upload(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 0.001)  # ~1 KB
    with pytest.raises(InputTooLargeError):
        load_document(b"x" * 5000, "big.txt")


@pytest.mark.parametrize("filename", ["fake.pdf", "fake.docx"])
def test_wrong_signature_is_rejected(filename):
    with pytest.raises(CorruptedFileError, match="not a valid"):
        load_document(b"just some plain text pretending", filename)


def test_whitespace_only_text_file():
    with pytest.raises(EmptyDocumentError):
        load_document(b"   \n\n  ", "blank.txt")
