# Testing

| Suite | Tests | Runtime | Command |
|---|---|---|---|
| Backend, fast | 291 | ≈ 1 min | `pytest -m "not slow"` (from `backend/`) |
| Backend, slow (real `bart-large-cnn`) | 9 | ≈ 12 min on CPU | `pytest -m slow` |
| Frontend | 39 | ≈ 15 s | `npm test` (from `frontend/`) |

Backend line coverage (fast suite, `pytest -m "not slow" --cov=app`): **99 %** (1,735 statements, 22 not
covered). The remaining lines are defensive paths that cannot be triggered without contrived fault injection,
e.g. a PDF page that fails mid-extraction or a malformed UTF-16 byte-order mark.

**Continuous integration** (`.github/workflows/ci.yml`) runs on every push and pull request: the fast backend suite
on Python **3.11** (minimum supported) and **3.14** (development version), plus frontend lint, tests and production
build.

## Strategy: real components, small inputs

- **Algorithms are tested with real libraries.** spaCy, scikit-learn, NetworkX and rouge-score are never mocked;
  tests use small hand-built documents whose correct answer is known (e.g. a 10-sentence solar-energy article with
  two deliberately off-topic sentences that every extractive method must rank last).
- **Reference implementations verify the formulas.** PageRank (`pagerank_reference`) and ROUGE-N / LCS / ROUGE-L
  are re-implemented from scratch and compared with NetworkX and `rouge-score`.
- **Transformer code is tested at three levels:**
  1. a deterministic *stand-in generator* (keeps the first N words) for orchestration logic: chunk loops, length
     planning, routing (seconds);
  2. `hf-internal-testing/tiny-random-bart` (0.5 MB, random weights, 100-token window) for the real model-wrapper
     code: loading, tokenization, batched generation, the no-truncation guard, thread-safe lazy loading, and real
     chunking with a real tokenizer (seconds);
  3. the real `facebook/bart-large-cnn`, marked `slow`: summary quality sanity, determinism, load-once,
     hierarchical summarization, Hybrid being faster than BART, and the end-to-end API integration test.
- **Test documents are generated in code** (PDFs with PyMuPDF, DOCX with python-docx, including scanned, mixed,
  encrypted and corrupted files), so no binary fixtures are committed.
- **Frontend tests use Testing Library**, which finds elements the way a user (or screen reader) does, by role and
  label, so they also check basic accessibility (named radio buttons, alerts, progress bars).

## Backend test inventory (`backend/tests/`)

| File | Tests | What it covers |
|---|---|---|
| `test_cleaner.py` | 11 | Unicode/ligatures, de-hyphenation, page numbers, unwrapping, bullets |
| `test_sentence_splitter.py` | 14 | Abbreviations, decimals, paragraph breaks, fragment filtering, dangling quotes, offsets, segmenter fallback |
| `test_document_loaders.py` | 31 | TXT encodings; PDF pages, headers/footers, scanned (OCR), partially scanned, encrypted, corrupted; DOCX; dispatcher validation (type, size, signature, empty) |
| `test_preprocessing_pipeline.py` | 8 | Validation limits, end-to-end PDF → sentences |
| `test_tfidf.py` | 28 | Tokenizer/stemming, length targets, selection & redundancy, centroid vs mean scoring, edge cases |
| `test_textrank.py` | 18 | Similarity matrix, threshold, k-NN sparsification, PageRank vs formula, ranking, fallbacks |
| `test_extractive_performance.py` | 3 | 2,500-sentence documents within time limits; dense-graph sparsification |
| `test_bart.py` | 21 | Token budgets, output tidying, lazy registry, load errors, input limits; *slow:* real BART |
| `test_models.py` | 9 | Tiny real BART: loading, token counting, batched generation, no truncation, inference errors, real chunking, workflow routing |
| `test_chunker.py` | 12 | Whole sentences, balance, limits, merging, overlap, over-long sentences, real BART tokenizer |
| `test_long_document.py` | 8 | Map-reduce rounds, fused vs concatenated, proportional budgets, termination, progress |
| `test_hybrid.py` | 12 | Selection order and budget, token cap, target relative to the original; *slow:* faster than BART |
| `test_workflow.py` | 17 | Every LangGraph route, loop termination, equivalence with direct calls, errors, progress |
| `test_langchain_integration.py` | 22 | Loaders → Documents, page provenance, cross-page sentences, splitters, recursive vs sentence-aware |
| `test_evaluation.py` | 24 | ROUGE vs reference implementation, clipping, stemming, word order, no-reference behaviour, faithfulness cases |
| `test_api.py` | 44 | Every endpoint and method, readable validation errors, all bad uploads, async jobs; *slow:* text → API → BART integration |
| `test_robustness.py` | 16 | Job crash/expiry, PageRank non-convergence, chunker re-splitting, Hybrid fallback, faithfulness edge cases, warm-up failure, GPU probe (including "CUDA reported but unusable") |
| `test_health.py` | 2 | Health endpoint, 404 JSON |

## Frontend test inventory (`frontend/src/`)

| File | Tests | What it covers |
|---|---|---|
| `utils/utils.test.js` | 13 | Number/score/time formatting, filenames, file validation, local history (cap, delete, broken storage) |
| `services/api.test.js` | 7 | Readable error messages; job polling (progress, completion, failure, cancel) |
| `components/components.test.jsx` | 14 | Form rules (min length, file type, submit payload, model warm-up), keyboard radio groups, results for extractive and abstractive output, inline ROUGE evaluation, faithfulness flags, copy |
| `pages/pages.test.jsx` | 5 | Full page flow with mocked API (progress → result → history; error; cancel), history page |

## Running

```bash
cd backend
pytest -m "not slow"                       # fast suite
pytest -m slow                             # real BART (downloads 1.6 GB on first run)
pytest -m "not slow" --cov=app --cov-report=term-missing
```

```bash
cd frontend
npm test            # once
npm run test:watch  # re-run on save
```
