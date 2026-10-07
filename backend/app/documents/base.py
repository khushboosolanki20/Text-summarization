"""Common document types."""

from dataclasses import dataclass, field
from typing import Any, Protocol


class SourceDocument(Protocol):
    """
    A piece of input text plus metadata, e.g. one page of a PDF.

    This is the boundary between document loading and the NLP pipeline.
    ``langchain_core.documents.Document`` satisfies it (it has exactly these two
    attributes), and so does ``TextDocument`` below. Preprocessing and the
    workflow only rely on this protocol, so the LangChain loaders could be
    replaced without touching the summarizers.
    """

    page_content: str
    metadata: dict[str, Any]


@dataclass
class TextDocument:
    """Minimal framework-free ``SourceDocument``."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """
    Raw text extracted from an uploaded file, before cleaning.

    ``pages`` holds per-page text for PDFs (one entry per page, so page
    numbers can be reported); for TXT/DOCX it holds a single entry.
    ``warnings`` lists non-fatal problems the user should know about, e.g.
    "2 of 10 pages are scanned images and were skipped".
    """

    text: str
    file_type: str  # "txt" | "pdf" | "docx"
    filename: str
    pages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)
