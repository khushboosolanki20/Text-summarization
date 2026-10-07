"""
Application-level exceptions.

Every error the user can trigger (bad file, empty text, model failure, ...)
is raised as a subclass of ``IntelliSumError``. Each carries a human-readable
message that is safe to show in the frontend and the HTTP status code the API
should use. ``app.main`` converts them into ``{"detail": message}`` responses,
so Python stack traces are never sent to the client.
"""


class IntelliSumError(Exception):
    status_code: int = 400
    default_message: str = "The request could not be processed."

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


# --- Input / document errors ----------------------------------------------


class EmptyDocumentError(IntelliSumError):
    status_code = 422
    default_message = "The document is empty. Please provide some text to summarize."


class TextTooShortError(IntelliSumError):
    status_code = 422
    default_message = "The text is too short to summarize meaningfully."


class InputTooLargeError(IntelliSumError):
    status_code = 413
    default_message = "The input is too large to process."


class UnsupportedFileTypeError(IntelliSumError):
    status_code = 415
    default_message = "Unsupported file type. Please upload a .txt, .pdf or .docx file."


class CorruptedFileError(IntelliSumError):
    status_code = 422
    default_message = "The file could not be read. It may be corrupted or not a valid document."


class PasswordProtectedError(IntelliSumError):
    status_code = 422
    default_message = "The PDF is password-protected. Please upload an unlocked copy."


class ContextWindowExceededError(IntelliSumError):
    status_code = 413
    default_message = "The text is longer than the summarization model can read in one pass."


# --- Model errors -----------------------------------------------------------


class ModelLoadError(IntelliSumError):
    status_code = 503
    default_message = (
        "The abstractive summarization model could not be loaded. "
        "Extractive methods (TF-IDF, TextRank) are still available."
    )


class InferenceError(IntelliSumError):
    status_code = 500
    default_message = "The summarization model failed to generate a summary. Please try again."


class OCRRequiredError(IntelliSumError):
    status_code = 422
    default_message = (
        "This PDF appears to contain scanned images rather than selectable text. "
        "Optical character recognition (OCR) is not supported; please upload a text-based PDF."
    )
