# Methodology

> Status: skeleton (Phase 1). Sections are written alongside the implementation of each method.

## 1. Preprocessing

Every method works on **sentences**, so the quality of cleaning and segmentation directly affects every summary.

```
file bytes ──► loader (TXT / PDF / DOCX) ──► raw text ──► clean_text ──► split_sentences ──► validation
```

### 1.1 Document extraction (`app/documents/`)

| Format | Library | Notes |
|---|---|---|
| TXT | built-in | Decodes UTF-8 (with/without BOM) and UTF-16 (BOM), then falls back to cp1252 / latin-1. Files containing NUL bytes are rejected as binary. |
| PDF | PyMuPDF | Text extracted **page by page** in reading order (`sort=True`). |
| DOCX | python-docx | Body paragraphs in order; soft line breaks joined. Tables skipped with a warning. |

Before extraction the dispatcher checks, in order: extension → empty file → size limit → **file signature**
(`%PDF-` for PDF, `PK\x03\x04` ZIP header for DOCX), so a renamed file gets a clear error rather than a crash.

**Scanned PDFs.** A scanned page is an image with no text layer. A page with fewer than 20 extracted
characters that contains an image is classified as *image-only*. If every page is image-only, the loader raises
`OCRRequiredError` ("OCR is not supported") instead of silently returning nothing. If only some pages are, the
others are used and the user is warned which pages were skipped.

**Running headers and footers.** For PDFs with ≥ 3 pages, the first/last two lines of each page (at most a third
of the page) are compared across pages after replacing digits with `#` (so "Page 3 of 10" ≡ "Page 4 of 10").
Lines appearing on ≥ 60 % of pages are removed. This matters because frequency-based methods such as TF-IDF would
otherwise treat a journal name repeated on every page as an important term.

### 1.2 Cleaning (`app/preprocessing/cleaner.py`)

1. **Unicode NFKC normalisation**: ligatures (`ﬁ` → `fi`), non-breaking spaces, control characters.
2. **De-hyphenation**: `summari-⏎zation` → `summarization`, only when the next line starts with a lowercase
   letter (so `COVID-⏎19` is kept).
3. **Page-number lines** removed (`12`, `- 4 -`, `Page 3 of 10`).
4. **Unwrapping**: hard line breaks inside a paragraph are joined; blank lines remain paragraph boundaries;
   bullet / numbered list items become separate paragraphs.
5. Whitespace normalisation.

Output format: paragraphs separated by `\n\n`, no line breaks inside a paragraph.

### 1.3 Sentence segmentation (`app/preprocessing/sentence_splitter.py`)

Splitting on `.` fails on *"Dr. Smith met Prof. Jones of M.I.T. at 3 p.m."*. We use spaCy's `en_core_web_sm`:

| Segmenter | How boundaries are found | Speed (measured) | Accuracy |
|---|---|---|---|
| `parser` (default) | Dependency parse: a sentence is a complete syntactic tree | ≈ 3,700 words/s | Best: handled every abbreviation in our tests |
| `senter` | Small dedicated boundary classifier | ≈ 16,000 words/s | Split "M.I.T. They" and "Fig. 2" incorrectly |
| `rule` | Punctuation rules (`sentencizer`) | fastest | Lowest; automatic fallback if the model is missing |

Each paragraph is segmented independently, so a heading without a full stop is never merged with the next
sentence. Fragments with fewer than 3 words or no letters (headings, figure labels, "Yes!") are dropped as
candidates for summarization.

### 1.4 Validation (`app/preprocessing/validation.py`)

Input is rejected with a human-readable message if it is empty, longer than 500,000 characters, or shorter than
40 words / 3 sentences (all configurable in `app/config.py`).

### 1.5 Known limitations

- No OCR: scanned pages are detected and reported, not read.
- Multi-column PDFs and tables in PDFs may still be extracted in an imperfect reading order.
- Header/footer detection cannot tell a genuine running header from body text that is *identical* on every page.
- English only (the spaCy model and later the BART model are English).

## 2. Extractive summarization
Why selecting existing sentences works, and its limitations.

### 2.1 TF-IDF sentence scoring *(Phase 3)*

### 2.2 TextRank *(Phase 4)*
Similarity graph construction, PageRank, and sentence ranking.

## 3. Abstractive summarization
### 3.1 Transformers and attention *(Phase 5)*
### 3.2 BART *(Phase 5)*
What BART is, its denoising pre-training, and why `facebook/bart-large-cnn` suits news-style summarization.

## 4. Long documents and chunking *(Phase 6)*
Why a 1024-token context window requires hierarchical summarization.

## 5. Hybrid TextRank → BART *(Phase 7)*

## 6. Summary length control *(Phase 7/8)*
Short / medium / long presets and how they map to sentence counts and token lengths.

## 7. Faithfulness check (experimental) *(later phase)*
What hallucination means and why this check is only a heuristic.

## 8. Evaluation with ROUGE *(Phase 10)*
ROUGE-1, ROUGE-2, ROUGE-L and their limitations.
