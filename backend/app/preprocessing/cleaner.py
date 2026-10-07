"""
Text cleaning.

Raw text, especially text extracted from PDFs, contains layout artefacts that
hurt every later stage: words split across lines with a hyphen, hard line
breaks in the middle of sentences, ligature characters, page numbers and
irregular whitespace. ``clean_text`` removes these while keeping paragraph
boundaries, which the sentence splitter uses as hard sentence breaks.

The output format is: paragraphs separated by a blank line (``"\\n\\n"``),
with no line breaks inside a paragraph.
"""

import re
import unicodedata

# Control characters except tab (\t) and newline (\n).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏  ﻿]")

# "summari-\nzation" -> "summarization". Only joins when the next line starts
# with a lowercase letter, so genuine hyphenated terms like "COVID-\n19" or
# "Anglo-\nSaxon" are kept as written.
_HYPHENATED_BREAK = re.compile(r"(\w)-\n[ \t]*([a-z])")

# Lines that contain only a page number: "12", "- 12 -", "Page 3", "Page 3 of 10".
# The trailing "\n?" removes the line break too, so no blank line (which would
# look like a paragraph break) is left behind.
_PAGE_NUMBER_LINE = re.compile(
    r"^[ \t]*(?:-[ \t]*)?(?:page[ \t]+)?\d{1,4}(?:[ \t]+of[ \t]+\d{1,4})?(?:[ \t]*-)?[ \t]*$\n?", re.I | re.M
)

# A line that starts a list item: "•", "-", "*", "1.", "a)", "(iv)".
_LIST_ITEM = re.compile(r"^\s*(?:[•●▪◦‣\-\*]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|\(?[ivxlc]{1,6}\))\s+")
_BULLET_GLYPH = re.compile(r"^\s*[•●▪◦‣]\s*")

_MULTI_SPACE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:!?)\]])")
_BLANK_LINES = re.compile(r"\n\s*\n")


def normalize_unicode(text: str) -> str:
    """
    NFKC normalisation turns compatibility characters into their plain form,
    e.g. the ligature "ﬁ" -> "fi" and non-breaking spaces -> normal spaces.
    Curly quotes and dashes are kept (they are valid text, not artefacts).
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ").replace(" ", " ")
    return _CONTROL_CHARS.sub("", text)


def _unwrap_paragraph(block: str) -> list[str]:
    """
    Join the hard-wrapped lines of one paragraph into a single line.

    List items are kept as separate paragraphs because each bullet is usually
    an independent statement rather than part of the previous sentence.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _LIST_ITEM.match(line) and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(_BULLET_GLYPH.sub("", line))
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def clean_text(text: str) -> str:
    """Clean raw document text. Returns "" for empty / whitespace-only input."""
    if not text:
        return ""

    text = normalize_unicode(text)
    text = _HYPHENATED_BREAK.sub(r"\1\2", text)
    text = _PAGE_NUMBER_LINE.sub("", text)

    paragraphs: list[str] = []
    for block in _BLANK_LINES.split(text):
        for para in _unwrap_paragraph(block):
            para = _MULTI_SPACE.sub(" ", para)
            para = _SPACE_BEFORE_PUNCT.sub(r"\1", para).strip()
            if para:
                paragraphs.append(para)

    return "\n\n".join(paragraphs)


def count_words(text: str) -> int:
    """Whitespace word count, used consistently for all statistics."""
    return len(text.split())
