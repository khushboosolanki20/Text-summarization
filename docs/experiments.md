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

**Load only the needed split files.** `load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")` downloads
*all* splits (≈ 800 MB, plus ≈ 1.3 GB of processed cache) even though only the test split is returned. Load the
parquet file of the needed split directly instead (`load_dataset("parquet", data_files=...)`), roughly 30 MB.

**Planned: BART length strategy.** `bart-large-cnn` was fine-tuned to produce 56–142-token summaries; IntelliSum
instead sizes summaries by the short/medium/long ratio. Phase 14 compares both on validation articles that fit
in 1,024 tokens. A Phase 5 attempt on CPU was abandoned: the laptop was on battery and throttled
(≈ 0.6 cores used), so 40 articles × 3 configurations did not finish in 3 hours. Run experiments on mains power,
ideally on the GPU.

**Tune on validation, report on test.** Hyper-parameters (e.g. the TextRank similarity threshold) are chosen on
the validation split so the reported test numbers are not optimistically biased.

## 5. Results
*No results yet. Results will only be reported once actually produced by `experiments/scripts`.*

## 6. Discussion
