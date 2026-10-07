"""Common result type returned by every document loader."""

from dataclasses import dataclass, field


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
