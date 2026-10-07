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

An extractive summary is a **subset of the original sentences**, kept in their original order. It works because
well-written documents state their main points explicitly, in sentences that share vocabulary with the rest of
the text. If we can measure how *central* a sentence is, the top sentences form a summary.

Strengths: every sentence is copied from the source, so the summary cannot state facts the source doesn't
(no hallucination); it is fast, needs no training data or GPU, and every choice can be explained by a score.
Weaknesses: sentences can lose context when taken out of their neighbourhood (dangling "he", "this"), the
summary can feel choppy, and it cannot merge or shorten sentences.

### 2.0 Shared selection procedure (`app/summarizers/base.py`)

TF-IDF and TextRank differ **only** in how they score sentences. Everything else is shared:

1. **Target size:** `k = ceil(n × ratio)`, clamped to `1 … n−1`, where `ratio` is 0.12 / 0.22 / 0.32 for
   short / medium / long (configurable via `INTELLISUM_LENGTH_RATIOS`).
2. **Rank** sentences by score (ties → earlier sentence first).
3. **Redundancy control:** walking down the ranking, skip a sentence whose cosine similarity to an
   already-selected sentence exceeds 0.8. Documents often restate a key point, and both versions would score
   highly. If too many are skipped, they back-fill so the summary still has `k` sentences.
4. **Restore document order** so the summary reads naturally.

The result exposes every sentence's score and the selected indices, so the UI can show *why* a sentence was chosen.

### 2.1 TF-IDF sentence scoring (`app/summarizers/tfidf.py`)

**Term weighting.** Each sentence is treated as a tiny document. After tokenizing, removing stop words and
stemming (NLTK Porter stemmer, `app/preprocessing/tokenizer.py`), scikit-learn's `TfidfVectorizer` builds an
`n_sentences × n_terms` matrix:

```
tf(t, s)  = 1 + log(count of t in s)               (sublinear: repetition has diminishing returns)
idf(t)    = ln((1 + n) / (1 + df(t))) + 1          (df = number of sentences containing t)
w(t, s)   = tf(t, s) · idf(t),   each row then L2-normalised
```

A term weighs heavily in a sentence when it appears there but not everywhere, i.e. it is *specific*.

**Sentence importance: centroid method (default).** Average all sentence vectors to obtain the **document
centroid** `c`, a single vector describing what the document is about. Terms that recur across many sentences
dominate it. Each sentence is scored by its cosine similarity to the centroid:

```
score(s) = cos(v_s, c) = (v_s · c) / (‖v_s‖ ‖c‖)
```

so sentences that talk about the document's main topics rank highest (Radev et al., 2004). The highest-weighted
centroid terms are returned as `top_keywords`, mapped from stems back to readable words.

**Alternative: mean term weight (`scoring="mean"`).** Score a sentence by the average un-normalised TF-IDF
weight of its terms. This is the "textbook" approach, but it has a serious flaw: words that occur **only once**
in the document have the maximum IDF, so off-topic sentences built from unusual words score highest. In our unit
test, "The weather was pleasant in Paris when the report was released" was selected by `mean` and correctly
rejected by `centroid`.

**Design check on CNN/DailyMail (200 test articles, `short` length):**

| Method | Slice | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| Lead-3 baseline | first 200 (CNN) | 30.27 | 12.21 | 20.98 |
| TF-IDF centroid | first 200 (CNN) | 26.57 | 9.26 | 18.31 |
| TF-IDF mean | first 200 (CNN) | 17.10 | 2.98 | 11.45 |
| Lead-3 baseline | items 3000–3199 (Daily Mail) | 39.62 | 17.74 | – |
| TF-IDF centroid | items 3000–3199 (Daily Mail) | 35.24 | 14.74 | – |
| TF-IDF mean | items 3000–3199 (Daily Mail) | 19.52 | 3.35 | – |

Centroid scoring is ~9–16 ROUGE-1 points better than mean scoring, so it is the default. Both trail
**Lead-3** (the first three sentences), a known property of news: journalists put the key facts first
(the "inverted pyramid"), which a position-agnostic method like TF-IDF does not exploit. These are preliminary
numbers used for a design decision; the full comparison is in [experiments.md](experiments.md) (Phase 14).

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
