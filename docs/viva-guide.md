# Viva guide

Short answers to the questions this project should be able to answer, each with the evidence from IntelliSum
and a pointer to where it is implemented or measured. Longer explanations: [methodology.md](methodology.md),
[architecture.md](architecture.md), [experiments.md](experiments.md).

---

### 1. Why does extractive summarization work?
Well-written documents state their main points explicitly, in sentences that share vocabulary with the rest of the
text. If you can measure how **central** a sentence is, the most central sentences form a summary. Because the
sentences are copied, an extractive summary cannot state anything the source doesn't (no hallucination), and every
choice is explainable by a score. Downsides: copied sentences can be long, redundant or lose context
("he", "this").
→ `app/summarizers/base.py` (shared selection), methodology §2.

### 2. How is TF-IDF used?
Each sentence is treated as a small document and turned into a vector of TF-IDF weights:
`tf = 1 + log(count)`, `idf = ln((1+n)/(1+df)) + 1`, rows L2-normalised, after removing stop words and stemming
(NLTK Porter). A word weighs a lot in a sentence when it occurs there but not everywhere. The **average** of all
sentence vectors (the centroid) describes the whole document; sentences are ranked by **cosine similarity to the
centroid**. The textbook alternative (average term weight) rewards rare, off-topic words: in our tests it picked
"The weather was pleasant in Paris…" and scored 9–16 ROUGE-1 points lower.
→ `app/summarizers/tfidf.py`, methodology §2.1.

### 3. How does TextRank work, and how is the similarity graph built?
1. Sentences → TF-IDF vectors (as above).
2. All pairwise cosine similarities at once: `S = X·Xᵀ` (unit vectors, so dot product = cosine); diagonal removed.
3. Graph (NetworkX): one node per sentence, an undirected edge weighted `S[i,j]` when `S[i,j] > 0.05`; on long
   documents each sentence keeps only its 50 strongest edges.
4. PageRank scores the nodes; the top-scored sentences, in original order, form the summary.
The 0.05 threshold was chosen on validation data. Higher thresholds raised ROUGE only because the graph fell apart
and the summary drifted to the first sentences (TextRank turning into Lead-3), so they were rejected.
→ `app/summarizers/textrank.py`, methodology §2.2.

### 4. How does PageRank rank sentences?
`PR(i) = (1−d)/N + d · Σ_j [w(j,i) / Σ_k w(j,k)] · PR(j)`, with damping `d = 0.85`, iterated until the scores stop
changing. Each sentence passes its score to its neighbours in proportion to similarity, so a sentence is important
if it is similar to many important sentences. The `(1−d)/N` "random jump" guarantees a unique solution and gives
isolated sentences a small score. The scores sum to 1 (a probability distribution: where a random walker over the
sentence graph spends its time). `pagerank_reference()` implements the formula from scratch, and a test confirms it
matches NetworkX.

### 5. How does Transformer attention help summarization?
Attention lets every token compute a weighted mix of all other tokens:
`softmax(Q·Kᵀ/√d)·V`. In the encoder, "she" in sentence 9 can draw directly on "Sara Mellado" in sentence 2, with
no fading memory as in RNNs. In the decoder, **cross-attention** lets each generated word look at the most relevant
source positions; that is how an abstractive model "selects" content while writing. The cost grows with the square
of the input length, hence a fixed maximum input (1,024 tokens for BART).
→ methodology §3.1.

### 6. What is BART?
A Transformer **encoder-decoder** (Lewis et al., 2019): a bidirectional encoder like BERT plus an autoregressive
decoder like GPT, about 400 M parameters (12 + 12 layers). It is pre-trained as a **denoising autoencoder**: text is
corrupted (masked spans, shuffled sentences) and the model learns to reconstruct it, which teaches both
understanding and fluent generation.

### 7. Why is BART suitable here?
`facebook/bart-large-cnn` is fine-tuned on ~287,000 CNN/DailyMail article-summary pairs (the same domain as our
benchmark), runs **locally** (no API, 1.6 GB, works on CPU), and was the best method in our experiments:
**42.3 ROUGE-1**, significantly better than every other method including Lead-3. We decode with deterministic
4-beam search and `no_repeat_ngram_size=3`.
→ `app/summarizers/bart.py`, `app/summarizers/models.py`; experiments §3.

### 8. Why do long documents need chunking?
BART reads at most 1,024 tokens (~750 words). Truncating silently loses everything after that, often the
conclusions. IntelliSum splits the text into **balanced chunks of whole sentences** (≤ 900 BART tokens, measured with
BART's own tokenizer), summarizes each chunk, and fuses the partial summaries in a final pass, repeating if needed
(at most 3 rounds). Nothing is truncated; a test checks that `generate()` refuses over-long input. In the
experiments, 32 of 100 test articles needed chunking.
→ `app/preprocessing/chunker.py`, `app/summarizers/long_document.py`, methodology §4.

### 9. What does LangChain contribute?
The **document-processing layer**, not the algorithm:
- `BaseLoader` → `Document` objects, one per PDF page with page metadata;
- page provenance (each sentence knows its page; chunks report page ranges);
- chunking as a pluggable `TextSplitter`. Our sentence-aware splitter is one; LangChain's
  `RecursiveCharacterTextSplitter` can be swapped in for comparison (it may cut sentences in half).
LangChain is confined to two adapter files; the summarizers never import it.
→ `app/documents/langchain_loaders.py`, `app/preprocessing/langchain_splitter.py`, architecture §7.

### 10. What does LangGraph contribute?
**Explicit control flow**: one typed state graph routes each request (extractive / hybrid / BART; fits the window
or not) and runs the chunk → summarize → combine **loop** with exit conditions. The state records the path taken,
per-node timings, chunks and intermediate summaries (shown in the UI). Nodes contain no algorithms; tests confirm the
graph gives identical results to calling the summarizers directly.
→ `app/graph/summarization_graph.py`, architecture §4.

### 11. How does the hybrid approach work?
TextRank reads the whole document and selects its most central sentences, about **3× the requested summary length**,
skipping near-duplicates. If the summary fits one BART pass, the selection is also capped to one BART window. BART
then rewrites the selection, with the target length computed from the **original** document. Result: 40.0 ROUGE-1
(ties Lead-3, +4.7 over TextRank) at **2.2× BART's speed**, and it never needed chunking. The expansion factor is a
quality/speed dial (1.5× → 39.1, 3× → 42.7, 4× → 45.1 ROUGE-1 on validation, at +40 % time for 4×).
→ `app/summarizers/hybrid.py`, methodology §5, experiments §4.1.

### 12. How does ROUGE work?
It counts overlap between the candidate summary and a human reference, after lowercasing and stemming:
ROUGE-1 = words, ROUGE-2 = word pairs, ROUGE-L = longest common subsequence (in-order words, gaps allowed;
dynamic programming). `precision = overlap / candidate units`, `recall = overlap / reference units`,
F1 = their harmonic mean. Overlap is **clipped**: a word counts at most as often as it appears in both texts.
→ `app/evaluation/rouge.py` (with a from-scratch reference implementation, tested against `rouge-score`),
methodology §8.

### 13. Why does ROUGE have limitations?
- It measures **word overlap, not meaning**: a correct paraphrase can score low.
- It ignores **factual accuracy**: a fluent summary with a wrong number can score high.
- It **favours copying**, which partly explains why Lead-3 is hard to beat on news.
- It is **length-sensitive**: in our validation experiment, BART's own trained length scored +2.7 ROUGE-1 over our
  12 % setting purely because it wrote slightly longer summaries.
- It needs a **human reference**, so IntelliSum shows ROUGE only when one is provided (never estimated).

### 14. What does hallucination mean?
Generated text that is fluent but **not supported by the source**: a wrong figure, a name that doesn't occur, an
invented cause. Only abstractive methods can do it. IntelliSum's experimental check flags summary sentences whose
content words, numbers or named entities are missing from the source, and shows the closest source passage. On
the test set it flagged 13 of 100 BART and 10 of 100 Hybrid summaries. It is a **heuristic**: it caught a
"15 percent" vs "50 percent" error that word overlap alone missed, but it cannot detect a false claim made of source
words. NLI or embedding models would do better (future work).
→ `app/evaluation/faithfulness.py`, methodology §7.

### 15. What are the system's limitations?
- English only; no OCR for scanned PDFs (detected and reported, not read).
- BART on CPU is slow (≈ 57 s per news article; long reports take minutes). The GPU path exists, but on the
  development laptop the NVIDIA driver was too old for CUDA 12.6, so it fell back to CPU automatically.
- Extractive methods ignore sentence position, which costs them on news (Lead-3 wins by ~3.5 ROUGE-1 even at equal length).
- Chunks are summarized independently, so cross-chunk context (who "the company" is) can be lost.
- The faithfulness check is lexical, ROUGE is lexical, and both were evaluated on news only.
- History is per-browser (localStorage) and jobs are in memory: fine for a single-user app, not for a multi-user
  deployment.

### 16. What did the experiments show? (one-minute answer)
On 100 random CNN/DailyMail test articles: **BART 42.3 > Hybrid 40.0 ≈ Lead-3 39.8 > TextRank 35.4 ≈ TF-IDF 34.8**
(ROUGE-1). BART is significantly best; Hybrid gives most of BART's quality at less than half its time; TF-IDF and
TextRank are statistically indistinguishable (500 articles). Lead-3 reproduced the published 40.3, which validates
the evaluation pipeline. All design choices were tuned on the validation split.

---

## Likely follow-up questions

**Why not just call an LLM API?** The brief asked for a genuinely implemented NLP system that runs locally, with no
paid API. A local model keeps documents private, makes results reproducible (deterministic decoding) and makes the
pipeline explainable end to end.

**Why spaCy for sentence splitting instead of splitting on full stops?** Abbreviations and decimals
("Dr.", "M.I.T.", "92.5"). We measured spaCy's parser-based segmentation as the most accurate option (it handled
every abbreviation in our tests), at ~3,700 words/s.

**How do you know the results aren't cherry-picked?** Random seeded samples across the whole split, the same
articles for every method, bootstrap confidence intervals, significance only claimed with ≥ 30 paired articles,
design choices tuned on validation only, and every number regenerated by scripts committed to the repository.

**What would you do with more time?** See "Future work" in the README: a GPU run on the full test set, T5 and
PEGASUS comparisons, an NLI-based faithfulness check, OCR, and position-aware extractive scoring.
