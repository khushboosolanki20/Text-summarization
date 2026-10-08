# Experiments

Reproducible comparison of IntelliSum's summarization methods on CNN/DailyMail. Results and their discussion:
[docs/experiments.md](../docs/experiments.md).

```
experiments/
├── scripts/
│   ├── common.py               data loading (only the needed parquet file), seeded sampling, resumable CSV, bootstrap CIs
│   ├── run_comparison.py       main comparison on the TEST split (Lead-3, TF-IDF, TextRank, BART, Hybrid, …)
│   ├── tune_on_validation.py   design experiments on the VALIDATION split
│   └── summarize_results.py    per-article CSVs -> summary tables, paired differences, charts
├── notebooks/
│   └── analysis.ipynb          deeper analysis: CNN vs Daily Mail, length vs ROUGE, speed, faithfulness flags, examples
└── results/
    ├── test_short_seed42/      main results (per_article.csv, summary.*, paired.csv, rouge_f1.png, time.png, run.log)
    └── validation_*/           design-experiment results
```

## Setup

From the repository root, with the backend virtual environment active:

```bash
pip install -r experiments/requirements.txt
```

The CNN/DailyMail **test** split (~30 MB) and, for tuning, the **validation** split are downloaded from the Hugging
Face Hub on first use. Only those parquet files are fetched, never the whole dataset.

## Reproduce the results

```bash
# Main comparison (test split, seed 42, "short" = 12 %)
python experiments/scripts/run_comparison.py --methods lead3 tfidf textrank tfidf_top3 textrank_top3 --n 500
python experiments/scripts/run_comparison.py --methods hybrid bart --n 100

# Design experiments (validation split)
python experiments/scripts/tune_on_validation.py hybrid_expansion --n 40
python experiments/scripts/tune_on_validation.py bart_length --n 40
python experiments/scripts/tune_on_validation.py chunking --n 25

# Tables, paired comparisons and charts for every results folder
python experiments/scripts/summarize_results.py
```

- **Same articles for every method:** the split is shuffled once with the seed and the first *N* articles are taken,
  so the 100 BART/Hybrid articles are the first 100 of the 500 extractive ones, and methods can be compared
  pairwise on identical inputs.
- **Resumable:** each (article, method) result is appended to `per_article.csv` immediately. Re-running a command
  skips finished work, so a long CPU run can be stopped and continued.
- **Optional models:** `--methods t5 pegasus` runs T5-base / PEGASUS-cnn_dailymail through the same
  chunking/length code (≈ 0.9 / 2.3 GB downloads; slow on CPU).

## Runtime (laptop CPU, no GPU)

| Method | Time per article |
|---|---|
| Lead-3, TF-IDF, TextRank | ≈ 0.1–0.2 s |
| Hybrid | ≈ 20 s |
| BART | ≈ 60 s (most articles exceed 1,024 tokens and are chunked) |

Run long experiments on mains power: on battery the CPU is throttled and BART becomes several times slower.

## Output columns (`per_article.csv`)

`id`, `source` (cnn/dailymail), `method`, `length`, `status`/`error`, `original_words`, `summary_words`,
`reference_words`, ROUGE-1/2/L/Lsum precision/recall/F1 (`rouge1_p` … `rougeLsum_f`, 0–1), `compression_ratio`,
`seconds`, `strategy` (single_pass / fused / concatenated / …), `chunks`, `input_tokens`, `flagged_sentences`
(faithfulness check), `summary`.
