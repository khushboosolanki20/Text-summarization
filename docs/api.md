# API Reference

Base URL (development): `http://127.0.0.1:8000`
Interactive docs: `http://127.0.0.1:8000/docs` (Swagger UI) and `/redoc`.

All errors are returned as JSON `{"detail": "<human-readable message>"}`; stack traces are never exposed.

## Implemented

### `GET /api/health`
Liveness check plus hardware info.

```json
{
  "status": "ok",
  "app": "IntelliSum",
  "version": "0.1.0",
  "torch_installed": true,
  "cuda_available": true,
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU"
}
```

## Planned *(Phase 11)*

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/summarize/text` | Summarize pasted text |
| POST | `/api/summarize/file` | Summarize an uploaded TXT / PDF / DOCX |
| POST | `/api/evaluate` | ROUGE between a candidate and a reference summary |

Planned request body for `/api/summarize/text`:

```json
{ "text": "...", "method": "hybrid", "length": "medium", "reference_summary": null }
```

`method` ∈ `tfidf | textrank | bart | hybrid`; `length` ∈ `short | medium | long`.
ROUGE fields are `null` unless a `reference_summary` is supplied. ROUGE needs a human-written
reference to compare against, so it cannot be computed for an arbitrary user document.
