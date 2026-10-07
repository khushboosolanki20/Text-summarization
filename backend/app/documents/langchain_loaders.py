"""
LangChain document loaders.

LangChain's ``BaseLoader`` is the standard interface for turning a source
(file, URL, database ...) into ``Document`` objects: ``page_content`` plus a
``metadata`` dict. Using it here means:

* every input, whether an uploaded PDF, DOCX, TXT or pasted text, enters the
  pipeline in the same shape;
* each PDF page becomes its own ``Document`` with ``{"page": n}`` metadata,
  so later stages can report *where* a sentence came from;
* the loaders plug into the wider LangChain ecosystem (``load_and_split``,
  other splitters, vector stores) without adapters.

The actual extraction and validation is not re-implemented: the loaders call
the Phase 2 modules (``app.documents.loader``), which detect scanned or
corrupted files and enforce size limits.
"""

from collections.abc import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from app.documents.loader import load_document


class UploadedFileLoader(BaseLoader):
    """
    Load an uploaded TXT, PDF or DOCX file.

    PDF: one ``Document`` per page that contains text (``metadata["page"]`` is
    1-based). TXT/DOCX: a single ``Document``. Non-fatal problems found while
    extracting (e.g. skipped scanned pages, ignored tables) are available in
    ``self.warnings`` after loading.
    """

    def __init__(self, data: bytes, filename: str):
        self.data = data
        self.filename = filename
        self.warnings: list[str] = []

    def lazy_load(self) -> Iterator[Document]:
        extracted = load_document(self.data, self.filename)  # validates; raises IntelliSumError
        self.warnings = list(extracted.warnings)
        common = {
            "source": self.filename,
            "file_type": extracted.file_type,
            **extracted.metadata,
        }
        if extracted.file_type == "pdf":
            common["total_pages"] = extracted.page_count
            for number, page in enumerate(extracted.pages, start=1):
                if page.strip():
                    yield Document(page_content=page, metadata={**common, "page": number})
        else:
            yield Document(page_content=extracted.text, metadata=common)


class PastedTextLoader(BaseLoader):
    """Wrap text typed or pasted into the UI as a single ``Document``."""

    def __init__(self, text: str, source: str = "pasted text"):
        self.text = text
        self.source = source

    def lazy_load(self) -> Iterator[Document]:
        yield Document(page_content=self.text, metadata={"source": self.source, "file_type": "text"})
