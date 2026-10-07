# API Reference

Base URL (development): `http://127.0.0.1:8000`. Interactive documentation, generated from the code:
**Swagger UI** at `/docs`, **ReDoc** at `/redoc`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness, hardware and model status |
| GET | `/api/config` | Methods, length ratios, limits, supported file types |
| POST | `/api/warmup` | Start loading the BART model in the background |
| POST | `/api/summarize/text` | Summarize pasted text (JSON) |
| POST | `/api/summarize/file` | Summarize an uploaded `.txt` / `.pdf` / `.docx` (multipart) |
| POST | `/api/summarize/text/async` | Same as `/summarize/text`, as a background job |
| POST | `/api/summarize/file/async` | Same as `/summarize/file`, as a background job |
| GET | `/api/jobs/{job_id}` | Job status, progress, result or error |
| POST | `/api/evaluate` | ROUGE of any summary against a reference |

## Errors

Every error is JSON with a **single human-readable message**; stack traces are never returned:

```json
{ "detail": "The text is too short to summarize (10 words, 2 sentences). Please provide at least 40 words in 3 or more sentences." }
```

| Status | When |
|---|---|
| 404 | Unknown or expired job id |
| 413 | Text over 500,000 characters, upload over 10 MB, or abstractive input over 20,000 words |
| 415 | Unsupported file type (including legacy `.doc`) |
| 422 | Malformed request (e.g. `"method: Input should be 'tfidf', 'textrank', 'bart' or 'hybrid'"`), empty or too-short text, corrupted / password-protected / scanned PDF ("OCR is not supported"), empty reference summary |
| 500 | Summarization model failed during generation (friendly message); unexpected server error (generic message, details only in the server log) |
| 503 | Summarization model could not be loaded ("Extractive methods (TF-IDF, TextRank) are still available.") |

## `POST /api/summarize/text`

```json
{
  "text": "…",
  "method": "hybrid",
  "length": "medium",
  "reference_summary": null
}
```

| Field | Values | Default |
|---|---|---|
| `text` | the document | required |
| `method` | `tfidf` · `textrank` · `bart` · `hybrid` | `hybrid` |
| `length` | `short` (12 %) · `medium` (22 %) · `long` (32 %) | `medium` |
| `reference_summary` | human-written summary; **ROUGE is only computed when given** | `null` |

### Response (abridged)

```json
{
  "summary": "Solar power capacity grew faster than any other energy source last year…",
  "method": "textrank",
  "length": "short",
  "original_word_count": 123,
  "summary_word_count": 32,
  "compression_ratio": 73.98,
  "processing_time": 0.041,
  "metrics": {
    "original_word_count": 123, "summary_word_count": 32, "compression_ratio": 73.98,
    "num_sentences": 10, "num_selected_sentences": 2, "processing_time": 0.041,
    "rouge1": null, "rouge2": null, "rougeL": null, "rouge": null,
    "rouge_note": "ROUGE needs a human-written reference summary to compare against. No reference was provided, so ROUGE was not computed."
  },
  "strategy": "extractive",
  "sentences": [
    { "index": 0, "text": "Solar power capacity grew…", "page": null, "score": 0.1523, "selected": true }
  ],
  "faithfulness": { "applicable": false, "label": "Potentially unsupported content (experimental)", "reason": "Extractive summaries copy source sentences verbatim." },
  "intermediate_summaries": [],
  "warnings": [],
  "source": { "type": "text" },
  "path": ["preprocess", "extractive_summarize", "postprocess", "evaluate"],
  "timings": { "preprocess": 0.031, "extractive_summarize": 0.008, "postprocess": 0.0, "evaluate": 0.002 },
  "metadata": { "num_sentences": 10, "num_selected": 2, "graph": { "nodes": 10, "edges": 21 }, "top_edges": [] }
}
```

With a `reference_summary`, `metrics.rouge1/rouge2/rougeL` hold the F1 scores (0–1) and `metrics.rouge` holds
precision, recall and F1 for `rouge1`, `rouge2`, `rougeL` and `rougeLsum`.

| Field | Meaning |
|---|---|
| `compression_ratio` | % of the original removed: `100 × (1 − summary words / original words)` |
| `strategy` | `extractive` · `single_pass` (fits BART's window) · `fused` (chunked, then a final pass) · `concatenated` (section-by-section) · `max_levels_reached` |
| `sentences[].selected` | in the summary (TF-IDF/TextRank) or passed to BART (Hybrid) |
| `sentences[].score` | TF-IDF / TextRank importance (null for BART) |
| `sentences[].page` | source page for PDF uploads |
| `faithfulness` | for BART/Hybrid: per summary sentence `coverage`, `similarity`, `closest_source`, `unsupported_numbers`, `unsupported_entities`, `flagged`, `reasons`. **Experimental heuristic, not a guarantee.** |
| `intermediate_summaries` | chunk summaries when a long document was chunked |
| `path`, `timings` | LangGraph nodes visited and seconds spent in each |
| `metadata` | method details: TextRank graph statistics, BART model/device/token budgets, Hybrid `extractive_stage` (input reduction %), `reduction_levels` with `chunk_pages` |

## `POST /api/summarize/file`

`multipart/form-data` with `file` (required) and optional form fields `method`, `length`, `reference_summary`.

```bash
curl -X POST http://127.0.0.1:8000/api/summarize/file -F "file=@report.pdf" -F method=hybrid -F length=short
```

The response is the same as above, plus `source` (`{"type": "file", "filename": …, "file_type": "pdf", "pages": 12,
"title": …}`), a page number for every sentence, and `warnings` such as *"2 of 12 pages appear to be scanned
images and were skipped"*.

## Asynchronous jobs (progress for long documents)

`POST /api/summarize/text/async` and `/api/summarize/file/async` take the same input but return **202** at once:

```json
{ "job_id": "3f2c…", "status": "queued", "progress": null, "result": null, "error": null, "created_at": 1760000000.0, "finished_at": null }
```

Poll `GET /api/jobs/{job_id}`:

```json
{ "job_id": "3f2c…", "status": "running", "progress": { "stage": "level 1: summarizing chunks", "done": 3, "total": 7 }, … }
```

`status` is `queued` → `running` → `completed` (with `result` = the normal response) or `failed` (with a friendly
`error`). Uploads are read and validated before the job is created, so an empty or oversized file fails
immediately. Jobs run one at a time by default (`INTELLISUM_MAX_CONCURRENT_JOBS`) and are kept for one hour after
finishing.

## `POST /api/evaluate`

```json
{ "summary": "…", "reference_summary": "…", "original_text": null }
```

Returns `rouge1`, `rouge2`, `rougeL` (F1), the full `rouge` breakdown, word counts, and, if `original_text` is
given, `compression_ratio` and the `faithfulness` check.

## `GET /api/health`

```json
{
  "status": "ok", "app": "IntelliSum", "version": "0.1.0",
  "torch_installed": true, "cuda_available": false, "device": "cpu",
  "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "gpu_problem": "CUDA error: CUDA-capable device(s) is/are busy or unavailable",
  "abstractive_model": "facebook/bart-large-cnn",
  "abstractive_model_loaded": false
}
```

`abstractive_model_loaded` is false until the first BART/Hybrid request (or `POST /api/warmup`), so the UI can
warn that the first abstractive summary takes longer.

## `GET /api/config`

Methods (id, name, type, description), length ratios, supported file types and size limits, so the frontend
never hard-codes them.
