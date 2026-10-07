"""
PDF loader built on PyMuPDF.

Text is extracted page by page. Two PDF-specific problems are handled:

1. **Scanned PDFs.** A scanned page is just an image: it has no text layer,
   so extraction returns nothing. Rather than silently returning an empty or
   partial document, we detect pages that contain images but no text. If the
   whole document is like that we raise ``OCRRequiredError``; if only some
   pages are, we continue and add a warning.

2. **Running headers and footers.** Lines such as a journal name or
   "Confidential - Company X" repeat on every page and would otherwise be
   picked up as "important" by frequency-based methods like TF-IDF. Lines
   that appear at the top or bottom of most pages are removed.
"""

import re
from collections import Counter

import pymupdf

from app.documents.base import ExtractedDocument
from app.errors import CorruptedFileError, EmptyDocumentError, OCRRequiredError, PasswordProtectedError

# Don't let MuPDF print parser warnings for damaged files to the server
# console; we report problems through our own exceptions instead.
pymupdf.TOOLS.mupdf_display_errors(False)

# A page with fewer characters than this is treated as having no text layer.
_MIN_PAGE_CHARS = 20
# Header/footer detection: inspect this many lines at the top and bottom of
# each page, and remove a line if it appears on at least this share of pages.
_EDGE_LINES = 2
_REPEAT_THRESHOLD = 0.6
_MIN_PAGES_FOR_REPEAT_DETECTION = 3

_DIGITS = re.compile(r"\d+")


def _normalise_line(line: str) -> str:
    # "Page 3 of 10" and "Page 4 of 10" should count as the same footer.
    return _DIGITS.sub("#", line.strip().lower())


def _edge_indices(lines: list[str]) -> set[int]:
    """
    Indices of the lines in the header/footer zone of a page: the first and
    last ``_EDGE_LINES`` non-empty lines, but never more than a third of the
    page each, so that the body of a short page is never treated as a header.
    """
    non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
    k = min(_EDGE_LINES, len(non_empty) // 3)
    if k == 0:
        return set()
    return set(non_empty[:k] + non_empty[-k:])


def remove_repeated_headers_footers(pages: list[str]) -> list[str]:
    """Remove lines that recur at the top/bottom of most pages."""
    text_pages = [p for p in pages if p.strip()]
    if len(text_pages) < _MIN_PAGES_FOR_REPEAT_DETECTION:
        return pages

    counts: Counter[str] = Counter()
    for page in text_pages:
        lines = page.split("\n")
        # set(): count each line at most once per page
        counts.update({_normalise_line(lines[i]) for i in _edge_indices(lines)})

    min_occurrences = max(2, int(len(text_pages) * _REPEAT_THRESHOLD))
    repeated = {line for line, n in counts.items() if n >= min_occurrences and line}
    if not repeated:
        return pages

    cleaned_pages = []
    for page in pages:
        lines = page.split("\n")
        # Only lines in the header/footer zone are candidates for removal, so
        # a repeated phrase inside the body text is never deleted.
        edge_idx = _edge_indices(lines)
        kept = [ln for i, ln in enumerate(lines) if not (i in edge_idx and _normalise_line(ln) in repeated)]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


def load_pdf(data: bytes, filename: str = "document.pdf") -> ExtractedDocument:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types for bad files
        raise CorruptedFileError("The PDF file is corrupted or invalid and could not be opened.") from exc

    with doc:
        if doc.needs_pass:
            raise PasswordProtectedError()
        if doc.page_count == 0:
            raise EmptyDocumentError("The PDF has no pages.")

        pages: list[str] = []
        image_only_pages: list[int] = []
        try:
            for number, page in enumerate(doc, start=1):
                # sort=True orders text blocks top-to-bottom, left-to-right,
                # which fixes the reading order of many multi-block layouts.
                text = page.get_text("text", sort=True)
                if len(text.strip()) < _MIN_PAGE_CHARS and page.get_images(full=False):
                    image_only_pages.append(number)
                    text = ""
                pages.append(text)
        except Exception as exc:
            raise CorruptedFileError("The PDF is damaged: some pages could not be read.") from exc

        metadata = {k: v for k, v in (doc.metadata or {}).items() if v and k in ("title", "author", "subject")}

    pages = remove_repeated_headers_footers(pages)
    pages_with_text = [p for p in pages if len(p.strip()) >= _MIN_PAGE_CHARS]

    if not pages_with_text:
        if image_only_pages:
            raise OCRRequiredError()
        raise EmptyDocumentError("No text could be extracted from the PDF.")

    warnings = []
    if image_only_pages:
        listed = ", ".join(map(str, image_only_pages[:10])) + ("…" if len(image_only_pages) > 10 else "")
        warnings.append(
            f"{len(image_only_pages)} of {len(pages)} pages appear to be scanned images and were skipped "
            f"(page {listed}). Their content is not included in the summary."
        )

    return ExtractedDocument(
        text="\n\n".join(p.strip() for p in pages if p.strip()),
        file_type="pdf",
        filename=filename,
        pages=pages,
        warnings=warnings,
        metadata=metadata,
    )
