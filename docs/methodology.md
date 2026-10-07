# Methodology

> Status: skeleton (Phase 1). Sections are written alongside the implementation of each method.

## 1. Preprocessing *(Phase 2)*
Text cleaning, sentence segmentation (spaCy), handling of PDF/DOCX artefacts.

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
