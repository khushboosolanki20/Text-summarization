"""
Single entry point for turning an uploaded file into text.

Validation happens in this order so the cheapest checks run first:
file type (by extension) -> empty -> size limit -> file signature -> extraction.
"""

from collections.abc import Callable
from pathlib import Path

from app.config import get_settings
from app.documents.base import ExtractedDocument
from app.documents.docx_loader import load_docx
from app.documents.pdf_loader import load_pdf
from app.documents.text_loader import load_text
from app.errors import CorruptedFileError, EmptyDocumentError, InputTooLargeError, UnsupportedFileTypeError

_LOADERS: dict[str, Callable[[bytes, str], ExtractedDocument]] = {
    ".txt": load_text,
    ".pdf": load_pdf,
    ".docx": load_docx,
}

SUPPORTED_EXTENSIONS = tuple(_LOADERS)


def _check_signature(extension: str, data: bytes) -> None:
    """
    Verify the file's "magic bytes" match its extension, so a renamed file
    (e.g. an image saved as .pdf) gets a clear error message.
    """
    if extension == ".pdf" and b"%PDF-" not in data[:1024]:
        raise CorruptedFileError("The file has a .pdf extension but is not a valid PDF document.")
    if extension == ".docx" and not data.startswith(b"PK\x03\x04"):
        # A .docx file is a ZIP archive, which always starts with "PK\x03\x04".
        raise CorruptedFileError("The file has a .docx extension but is not a valid Word document.")


def load_document(data: bytes, filename: str) -> ExtractedDocument:
    """Validate an uploaded file and extract its raw text."""
    extension = Path(filename or "").suffix.lower()
    if extension == ".doc":
        raise UnsupportedFileTypeError(
            "Legacy .doc files are not supported. Please save the document as .docx and try again."
        )
    if extension not in _LOADERS:
        raise UnsupportedFileTypeError()

    if not data:
        raise EmptyDocumentError("The uploaded file is empty.")

    max_mb = get_settings().max_upload_mb
    if len(data) > max_mb * 1024 * 1024:
        raise InputTooLargeError(f"The file is too large ({len(data) / 1_048_576:.1f} MB). The limit is {max_mb:g} MB.")

    _check_signature(extension, data)
    document = _LOADERS[extension](data, filename)

    if not document.text.strip():
        raise EmptyDocumentError("No text could be extracted from the file.")
    return document
