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

### 2.2 TextRank (`app/summarizers/textrank.py`)

**Intuition.** PageRank ranks a web page as important if many important pages link to it. TextRank
(Mihalcea & Tarau, 2004) applies this to sentences: a sentence is important if it is **similar to many other
important sentences**, i.e. it states content the rest of the document keeps returning to.

**1. Vector representation.** The same TF-IDF sentence vectors as §2.1 (stop words removed, Porter-stemmed,
L2-normalised).

**2. Pairwise similarity.** All cosine similarities at once with one sparse matrix product, `S = X·Xᵀ`
(rows are unit vectors, so dot product = cosine). The diagonal is removed: a sentence cannot vote for itself.

**3. Graph construction (NetworkX).** An undirected weighted graph: one node per sentence, an edge `(i, j)` with
weight `S[i, j]` when `S[i, j] > 0.05`. On long documents each sentence keeps only its **50 strongest edges**
(a k-nearest-neighbour graph; an edge survives if either endpoint keeps it).

**4. PageRank.** `networkx.pagerank` iterates until scores change by less than 10⁻⁶:

```
PR(i) = (1 − d) / N  +  d · Σ_{j ∈ neighbours(i)}  [ w(j,i) / Σ_k w(j,k) ] · PR(j)        d = 0.85
```

Each sentence passes its score to its neighbours in proportion to how similar they are. The `(1 − d)/N` term is
a "random jump" (with probability 0.15 the random walker restarts at a random sentence), which guarantees a
unique solution and gives disconnected sentences a small non-zero score. The scores form a probability
distribution (they sum to 1). `pagerank_reference()` implements the same formula from scratch as a power
iteration, and a unit test checks it matches NetworkX.

**5. Selection.** Shared with TF-IDF (§2.0).

**TextRank vs TF-IDF centroid.** The centroid compares each sentence with the *average* document; TextRank uses
the *structure* of sentence-to-sentence links, so a sentence tightly connected to a cluster of central
sentences is promoted even if it overlaps little with the global average. Unlike the original paper, which used
word overlap normalised by sentence length, we use TF-IDF cosine similarity, so common words contribute less.

#### Design decisions (measured)

**Similarity threshold.** Tuned on 300 random **validation** articles. Tuning on the test set would
leak information into reported results.

| Threshold | ROUGE-1 | Isolated sentences | Mean relative position of selected | Articles with no edges |
|---|---|---|---|---|
| 0.00 | 37.16 | 3.5 % | 0.33 | 0 |
| 0.05 | 36.80 | – | – | – |
| 0.10 | 36.92 | 10.5 % | 0.36 | 0 |
| 0.20 | 37.32 | 43.1 % | 0.35 | 3 |
| 0.30 | 37.86 | 68.6 % | 0.29 | 15 |
| 0.50 | 38.01 | 86.8 % | 0.17 | 94 |

ROUGE *rises* with high thresholds, but for the wrong reason: the graph falls apart (87 % of sentences
isolated at 0.5), PageRank becomes nearly uniform, ties are broken by position, and the summary drifts toward the
article's opening sentences. Lead-4 scores 40.9 on the same articles. A high threshold would make "TextRank"
quietly imitate the Lead baseline. Between 0.00 and 0.20 the differences (±0.5) are within noise, so we use
**0.05**, which removes the weakest half of the edges (pairs sharing a single minor word) and halves graph size.

**k-nearest-neighbour limit.** If all sentences share vocabulary (a long single-topic report), the graph becomes
complete: n(n−1)/2 edges, 3.1 million for 2,500 sentences, 13 s to build and rank. Limiting each sentence to
its 50 strongest edges gives identical ROUGE on validation (36.80 / 16.11 / 24.32 with and without, including the
43 articles longer than 51 sentences) and bounds the cost:

| Document | Without limit | With limit (50) |
|---|---|---|
| 4,102 real sentences (~80k words) | 2.0 s, 203 MB, 310k edges | 1.0 s, 85 MB, 126k edges |
| 2,500 fully connected sentences | ≈ 13 s, 3.1 M edges | 0.9 s, 74k edges |

#### Preliminary comparison (300 random test articles, seed 42, `short` length)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-3 | 40.66 | 17.61 | 25.10 |
| TF-IDF centroid | 34.35 | 14.30 | 22.68 |
| TextRank (threshold 0.05) | 34.88 | 14.44 | 22.80 |

TextRank and TF-IDF are close (both are built on the same TF-IDF vectors); both trail Lead-3 for the
inverted-pyramid reason given in §2.1. Final numbers are produced in Phase 14.

## 3. Abstractive summarization

An abstractive summary is **written**, not selected: the model generates new sentences that can merge facts
from different parts of the document, drop subordinate clauses and paraphrase. This produces fluent, compact
summaries, but because the text is generated it can also state things the source doesn't support
(**hallucination**, see §7).

### 3.1 Transformers and attention

Earlier neural summarizers (RNNs/LSTMs) read text one word at a time and squeezed everything into a fixed-size
memory, so information from early in a long input faded. The **Transformer** (Vaswani et al., 2017) replaces
recurrence with **attention**: every token computes a weighted combination of *all* other tokens,

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
```

where each token's *query* `Q` is compared with every token's *key* `K` and the resulting weights mix their
*values* `V`. Several "heads" do this in parallel, each free to learn a different relation (coreference,
subject-verb, topic). For summarization this means:

- the representation of "she" in sentence 9 can draw directly on "Sara Mellado" in sentence 2;
- when the decoder writes each summary word, **cross-attention** lets it look back at whichever source
  positions are most relevant at that moment, which is how content is "selected" in an abstractive model.

The cost of attention grows with the square of the input length, which is why models have a fixed maximum input
(the context window, §4).

### 3.2 BART (`app/summarizers/bart.py`, `app/summarizers/models.py`)

**Architecture.** BART (Lewis et al., 2019) is an **encoder-decoder** Transformer: a bidirectional encoder (like
BERT) reads the whole document, and an autoregressive decoder (like GPT) generates the summary token by token.
`bart-large` has 12 encoder + 12 decoder layers, 16 attention heads and ~400 M parameters.

**Pre-training.** BART is a *denoising autoencoder*: text is corrupted (spans replaced by a single mask, sentence
order shuffled) and the model learns to reconstruct the original. Reconstruction requires both understanding
the input and generating fluent text, exactly the two skills summarization needs.

**Why `facebook/bart-large-cnn`.** This checkpoint is fine-tuned on ~287,000 CNN/DailyMail article-summary pairs,
so it has learned what a news summary looks like, matches our evaluation dataset, runs locally (no API), and fits
on a laptop GPU or CPU (1.6 GB). Its main constraint is the **1,024-token context window** (≈ 750 words).

**Decoding.**

| Setting | Value | Why |
|---|---|---|
| Beam search | 4 beams | Keep the 4 best partial summaries at each step instead of greedily taking the single most likely token |
| Sampling | off | Same input → same summary (reproducible, testable) |
| `no_repeat_ngram_size` | 3 | Forbid repeating any 3-word sequence (prevents loops like "the report said the report said") |
| `length_penalty` | 2.0 | Counteracts beam search's bias toward short outputs |
| Length | `ratio × input words × 1.3` tokens, ±25 % | 1.3 ≈ BPE tokens per English word; the range lets the model stop at a sentence boundary |

If generation hits the length limit mid-sentence, the incomplete trailing fragment is removed (when at least one
complete sentence remains).

**Engineering.**

- **Loaded once, lazily, thread-safe.** The first abstractive request loads the model (≈ 3 s from a warm disk
  cache, ≈ 36 s on the very first cold read); later requests reuse the same objects. The server starts
  instantly and the extractive methods never wait for BART. `/api/health` reports whether it is loaded.
- **Model-agnostic.** `models.py` wraps any Hugging Face seq2seq model. T5 (`summarize:` prefix) and PEGASUS are
  registered and loaded only if used, so adding a model is one registry entry.
- **GPU if it actually works.** `torch.cuda.is_available()` can return `True` even when every GPU operation fails
  (it happened on our development laptop: driver 537.70 supports CUDA 12.2, PyTorch was built for 12.6). A tiny
  probe operation decides; otherwise the model runs on CPU with a logged warning.
- **No silent truncation.** Input is tokenized with `truncation=False`; if it exceeds 1,024 tokens a
  `ContextWindowExceededError` is raised. Long documents are handled by chunking (§4).
- **Failures are reported, not crashed.** Load failures (no network, missing PyTorch, out of memory) raise
  `ModelLoadError` ("extractive methods are still available"); GPU out-of-memory and other generation errors
  raise `InferenceError`.

## 4. Long documents and chunking

### 4.1 Why chunking is necessary

Self-attention compares every token with every other token, so its cost grows with the **square** of the input
length; models are therefore trained with a fixed maximum input. For BART it is **1,024 tokens (≈ 750 words)**.
A 10-page report has ~5,000 words. The naive fix, letting the tokenizer truncate, silently discards everything
after the first ~750 words. For news that might go unnoticed (key facts come first), but for a report the
conclusions are usually at the end. IntelliSum **never truncates**: input is tokenized with `truncation=False`,
and the low-level `generate()` refuses over-long input (a test verifies this).

### 4.2 Sentence-aware, balanced chunking (`app/preprocessing/chunker.py`)

1. Every sentence is measured with the **model's own tokenizer** (BART BPE), not estimated.
2. Sentences are grouped greedily into chunks of at most **900 tokens** (configurable `chunk_max_tokens`; below
   1,024 to leave headroom for special tokens).
3. **Balanced sizes:** with `T` total tokens and limit `L`, `n = ⌈T/L⌉` chunks are needed, so each aims for `T/n`
   tokens. Filling every chunk to the limit would often leave a tiny last chunk (e.g. 900 + 900 + 60), whose
   summary is mostly noise. A remaining tiny final chunk (< 30 % of the target) is merged into its predecessor
   when that fits.
4. **Whole sentences only.** The one exception is a single "sentence" longer than a whole chunk (e.g. a table
   flattened into one line), which is split at word boundaries and flagged `contains_split_sentence`.
5. **Verification:** after grouping, the *joined* text of each chunk is re-measured (joining can change BPE
   tokenization slightly); a chunk that is over the limit is split again.
6. **Optional overlap** (`chunk_overlap_sentences`, default 0): each chunk starts with the previous chunk's last
   sentence(s) to carry context across the boundary. The overlap is reserved *during* grouping. (A first version
   added it afterwards, but greedy chunks are full by then, so the overlap silently never happened; a unit test
   caught this.) It is off by default because repeated sentences tend to be repeated in the combined summary.

### 4.3 Hierarchical (map-reduce) summarization (`app/summarizers/long_document.py`)

```
document ─► chunks ─► MAP: summarize each chunk ─► COMBINE ─► fits window? ─yes─► REDUCE: final pass ─► summary
                                                               │no
                                                               └─► treat combined text as a new document, repeat
                                                                   (at most 3 rounds)
```

**Length planning.** Let `T = ratio × document words` be the requested length (§2.0).

| Case | Plan | Strategy |
|---|---|---|
| `T` fits one generation pass (`T × 1.3 ≤ 400` tokens, i.e. `T ≤ ~307` words) | Chunk summaries together aim for **2T** words (capped so their combination fits the window), giving the final pass more material than needed so it can **choose and fuse** across sections. The final pass writes `T` words. | `fused` |
| `T` is longer than one pass can write (e.g. 12 % of a 10,000-word report = 1,200 words) | A final pass would have to compress *below* the requested length, so chunk summaries together aim for `T` words and are returned **in document order**, i.e. a section-by-section summary. | `concatenated` |

Each chunk's share of the budget is **proportional to its share of the document's words**, and every round
must at least halve the text, which guarantees termination. After `max_reduction_levels` rounds the current
result is returned and flagged (`max_levels_reached`).

**Guarantees** (each covered by a unit test): every sentence is read exactly once in the first round (nothing
truncated, duplicated or reordered); no chunk exceeds the limit; longer settings give longer summaries; progress
is reported per chunk (for the UI progress bar); the loop terminates even if the model fails to compress.

**Cost.** One model pass per chunk plus the final pass. On CPU that is several seconds per chunk, so abstractive
input is capped at **20,000 words** (configurable). Longer input is rejected with a suggestion to use **Hybrid**,
which first shrinks the document extractively (§5). Nothing is silently cut.

**Limitations.** Chunks are summarized independently, so a chunk loses context from earlier sections (e.g. who
"the company" is); the final fusion pass partly repairs this. The `concatenated` strategy is coherent within each
section but has no transitions between sections.

## 5. Hybrid TextRank → BART (`app/summarizers/hybrid.py`)

### 5.1 Motivation

| | TextRank | BART |
|---|---|---|
| Reads | the whole document, any length | 1,024 tokens per pass |
| Strength | finds central content reliably and quickly | fluent, compressed, rephrased output |
| Weakness | copied sentences: choppy, redundant | long input = many slow passes, each chunk summarized without knowing what matters globally |

Hybrid uses TextRank as a **content selector** and BART as a **rewriter**.

### 5.2 Pipeline

```
full document ─► TextRank scores every sentence (graph centrality, §2.2)
             ─► take sentences best-first, skipping near-duplicates (cosine > 0.8),
                until ≈ 3 × the requested summary length            (word budget)
                and, if the summary fits one BART pass, ≤ 900 BART tokens  (token cap)
             ─► restore original order
             ─► BART (one pass if it fits, otherwise hierarchical, §4)
                with the length target computed from the ORIGINAL document
```

**Why ~3× the summary length?** BART needs more material than the final summary so it can still choose,
merge and rephrase. With exactly the summary length it could only paraphrase TextRank's choice. The factor
(`hybrid_expansion`) is a design choice, to be tuned on validation data in Phase 14.

**Why the token cap?** If the requested summary is short enough for a single generation pass (≤ ~307 words),
the selection is also limited to one BART window. BART then needs **no chunking at all**: one pass instead of one
per chunk plus a fusion pass. When a sentence would overflow the cap, shorter lower-ranked sentences are still
considered, so the window is filled as fully as possible.

**Length relative to the original.** BART normally sizes its summary as a ratio of *its* input. In Hybrid that
input is the (much shorter) selection, so `HybridSummarizer` passes an explicit target, `ratio × original words`,
via `summarize_to_target()`. A unit test checks the summary length matches the original document, not 12 % of the
selection.

### 5.3 Behaviour by document length

| Document vs summary | Behaviour |
|---|---|
| Short text, long setting (e.g. 150 words, 32 %) | 3 × 32 % ≈ 96 % of the text is selected: Hybrid ≈ BART. Filtering only matters when the document is much longer than the summary. |
| Medium/long document, summary fits one pass | Selection capped to one window: **one BART pass** instead of chunking. |
| Very long document, long summary | No token cap; BART summarizes ~3 × T words hierarchically instead of the whole document. This also allows documents above the 20,000-word abstractive limit, as long as the selection is below it. |

### 5.4 Trade-offs

- **Faster and more focused:** BART processes only the most central part of the document. Measured on a
  ~2,200-word test document (`short`, CPU, laptop on battery): **BART 320 s** (6 chunks + fusion pass) vs
  **Hybrid 107 s** (TextRank + a single BART pass), about 3× faster. Quality is compared with ROUGE in Phase 14.
- **Error propagation:** anything TextRank considers peripheral can never reach the summary, and TextRank's biases
  (favouring sentences that share vocabulary with many others) carry over.
- **Coherence:** selected sentences are not contiguous, so BART may see a pronoun whose antecedent was not
  selected.

The result exposes both stages: which sentences were passed to BART (`selected_indices`), their TextRank scores,
the input reduction (% of the document BART did not have to read), and BART's metadata.

## 6. Summary length control

One setting, three presets, configurable via `INTELLISUM_LENGTH_RATIOS`:

| Preset | Ratio | Extractive (TF-IDF, TextRank) | Abstractive (BART) | Hybrid |
|---|---|---|---|---|
| short | 12 % | `k = ⌈0.12 · n⌉` sentences | target `T = 0.12 · words`, generated as 0.75–1.25 · T · 1.3 tokens | TextRank selects ≈ 3T words, BART writes T |
| medium | 22 % | `k = ⌈0.22 · n⌉` | `T = 0.22 · words` | ″ |
| long | 32 % | `k = ⌈0.32 · n⌉` | `T = 0.32 · words` | ″ |

The ratios sit inside the ranges in the project brief (10–15 %, 20–25 %, 30–35 %). Two properties follow:

- **Extractive ratios count sentences, not words**, so the achieved compression depends on which sentences are
  chosen. The reported compression ratio is always measured on the actual output.
- **Abstractive lengths are targets, not guarantees.** The model may stop anywhere in the token range (so it can
  end on a sentence boundary), may end sooner, and generation is bounded by 20–400 tokens per pass. For very long
  documents the summary is section-by-section (§4.3). The reported word counts are always measured, never
  assumed.

## 7. Faithfulness check (experimental) (`app/evaluation/faithfulness.py`)

### 7.1 What hallucination means

An abstractive model generates text token by token from learned probabilities. It can therefore produce
fluent statements the source never made: a wrong number, a name that does not occur, an invented cause. This
is called **hallucination** (or unfaithfulness). Extractive summaries cannot hallucinate (they copy sentences),
although they can still mislead by omission or lost context.

### 7.2 The heuristic

For each sentence of a BART or Hybrid summary (labelled **"Potentially unsupported content (experimental)"**):

| Signal | How | Catches |
|---|---|---|
| Content-word coverage | share of the sentence's stemmed, non-stop words found anywhere in the source | new content |
| Unsupported numbers | numbers in the sentence that never occur in the source | wrong figures, years, amounts |
| Unsupported names | spaCy NER entities (people, organisations, places, …) not found in the source | invented names |
| Closest source passage | best TF-IDF cosine match among source sentences *and adjacent pairs* (abstractive sentences often fuse two) | shown so the reader can verify |

A sentence is flagged if coverage < 60 % or it contains an unsupported number or name. The response lists the
reasons in plain language, e.g. *"number(s) not found in the source: 15"*.

**Worked example** (unit-tested). Source: the solar-energy article ("installations rose by nearly 50 percent").

| Summary sentence | Coverage | Similarity | Flagged because |
|---|---|---|---|
| Solar power capacity grew faster than any other energy source last year. | 100 % | 0.87 | (not flagged) |
| Solar installations rose by nearly **15** percent globally. | 100 % | 0.89 | number not in source: 15 |
| The **World Bank** said **Germany** led the growth. | 33 % | 0.47 | low coverage; names not in source |
| The football team won the championship… | 0 % | 0.00 | low coverage |

The second row is the important one: word overlap and similarity are almost perfect, so a purely
similarity-based check would miss the wrong number.

**On real BART output** (5 random CNN/DailyMail validation articles, `short`): 0 of 16 summary sentences were
flagged, and every sentence had 100 % content-word coverage. `bart-large-cnn` is known to be largely
*extractive in style*: it mostly copies and compresses source phrases. That means no false alarms on these
articles, but the sample is far too small to say anything about how many real errors it would catch.

### 7.3 Limitations, stated honestly

- **Lexical, not semantic.** A faithful paraphrase with different words can be flagged (false positive); a
  sentence recombining source words into a false claim ("China accounted for half of *wind* capacity") is not
  flagged (false negative).
- Entity matching is by substring, and numbers written as words ("fifty") are not compared with digits.
- Better approaches (future work): sentence embeddings for semantic similarity, or a natural-language-inference
  model that checks whether the source *entails* each summary sentence. Both need another model download and were
  left out deliberately.
- It is a **review aid, not a guarantee**: the API and UI say so, and it can be disabled with
  `INTELLISUM_FAITHFULNESS_CHECK=false`.

## 8. Evaluation with ROUGE (`app/evaluation/rouge.py`, `app/evaluation/metrics.py`)

### 8.1 How ROUGE works

ROUGE (Lin, 2004) compares a candidate summary with a human-written **reference** by counting shared units, after
lowercasing, removing punctuation and Porter-stemming:

| Metric | Unit | Measures |
|---|---|---|
| ROUGE-1 | single words | content coverage |
| ROUGE-2 | word pairs (bigrams) | phrasing / local fluency |
| ROUGE-L | longest common subsequence (in order, gaps allowed) | sentence-level word order |
| ROUGE-Lsum | LCS per sentence, combined over the summary | the variant most CNN/DailyMail papers report |

```
precision = overlap / units in candidate        recall = overlap / units in reference
F1        = 2 · P · R / (P + R)
```

Overlap is **clipped**: a word counts at most as many times as it appears in the other text. (A unit test shows it:
"the model summarizes documents" vs "the models summarized *the* document" gives recall 4/5, because the reference
has "the" twice.) The LCS is computed by dynamic programming: `table[i][j]` = LCS of the first *i* candidate and
first *j* reference words.

Scores come from Google's `rouge-score` package (the reference implementation). `rouge_n_reference()`,
`lcs_length()` and `rouge_l_reference()` re-implement the formulas in ~30 readable lines, and tests confirm they
give identical precision, recall and F1.

### 8.2 When ROUGE is (not) computed

ROUGE needs a reference summary. For a user's own document there usually is none, so the API returns
`rouge1 = rouge2 = rougeL = null` with `rouge_note` explaining why, and **never** an estimated score. When the
user supplies a reference (or in experiments, where CNN/DailyMail provides one), all variants are returned with
precision, recall and F1.

Other statistics, always available: original and summary word counts, compression ratio, number of sentences,
number of sentences selected (extractive/hybrid), and processing time.

### 8.3 Limitations of ROUGE

- **Surface overlap, not meaning:** "the firm's profits fell" vs "the company lost money" scores near zero; a
  summary that copies reference words but states something false can score high.
- **Favours extractive output:** copying source sentences reuses the reference's vocabulary, which partly explains
  strong extractive baselines such as Lead-3.
- **One reference** captures one person's notion of what matters; other valid summaries are penalised.
- **Length-sensitive:** longer candidates gain recall and lose precision, so F1 is compared at similar lengths.
- It says nothing about fluency, coherence or factual correctness (hence the faithfulness check in §7).
