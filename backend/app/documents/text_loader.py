"""Plain-text (.txt) loader with encoding detection."""

from app.documents.base import ExtractedDocument
from app.errors import CorruptedFileError

# Tried in order. UTF-8 (with or without BOM) covers almost all modern files;
# cp1252 covers files saved by older Windows editors. latin-1 can decode any
# byte sequence, so it is the final fallback and never raises.
_FALLBACK_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16 byte-order mark
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise CorruptedFileError("The text file has an invalid UTF-16 encoding.") from exc

    # NUL bytes never occur in real 8-bit text files: this is a binary file
    # (e.g. an image or executable) that was given a .txt name.
    if b"\x00" in data[:8192]:
        raise CorruptedFileError("The file does not appear to be a plain-text document.")

    for encoding in _FALLBACK_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise CorruptedFileError()  # unreachable: latin-1 accepts every byte


def load_text(data: bytes, filename: str = "document.txt") -> ExtractedDocument:
    text = decode_text(data)
    return ExtractedDocument(text=text, file_type="txt", filename=filename, pages=[text])
