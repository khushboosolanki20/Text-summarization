# Experiments

> Status: skeleton (Phase 1). Filled in during Phase 14 with real, reproducible results.

## 1. Goal
Compare TF-IDF, TextRank, BART and Hybrid summarization (and optionally T5 / PEGASUS) on a public benchmark.

## 2. Dataset
CNN/DailyMail (`abisee/cnn_dailymail`, version 3.0.0) via Hugging Face Datasets, using a small, configurable
test subset for local runs. The full dataset is never downloaded during development.

## 3. Metrics
- ROUGE-1, ROUGE-2, ROUGE-L (F1)
- Average inference time per document
- Compression ratio

## 4. Procedure

**Sampling must be random.** The CNN/DailyMail test split is ordered: the first ~1,100 articles are from CNN and
the rest from the Daily Mail, and the two sources behave very differently (Lead-3 ROUGE-1 was 30.3 on the first
200 articles versus 39.6 on items 3000–3199 in a Phase 3 check). Taking "the first N rows" would therefore
measure CNN only. The evaluation subset is drawn with a fixed random seed across the whole split.

## 5. Results
*No results yet. Results will only be reported once actually produced by `experiments/scripts`.*

## 6. Discussion
