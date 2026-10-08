"""
Shared helpers for the IntelliSum experiments.

* Data: CNN/DailyMail 3.0.0 from the Hugging Face Hub. Only the parquet file
  of the requested split is downloaded (test ~ 30 MB), never the whole
  dataset (~ 800 MB + cache).
* Sampling: a seeded random sample across the *whole* split. The split is
  ordered (CNN articles first, then Daily Mail), so "the first N rows" would
  only measure CNN (see docs/experiments.md).
* Results: one CSV row per (article, method), appended as soon as it is
  computed, so long runs can be interrupted and resumed.
* Statistics: bootstrap 95 % confidence intervals for means and for paired
  differences between methods.
"""

import csv
import os
import random
import sys
from pathlib import Path

import numpy as np

# Windows consoles default to a legacy code page; print results as UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))  # make the `app` package importable
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

RESULTS_DIR = ROOT / "experiments" / "results"
DATASET = "abisee/cnn_dailymail"
SPLIT_FILES = {
    "test": "3.0.0/test-00000-of-00001.parquet",
    "validation": "3.0.0/validation-00000-of-00001.parquet",
}


def load_split(split: str):
    """Download (once) and read one CNN/DailyMail split as a pandas DataFrame."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, SPLIT_FILES[split], repo_type="dataset")
    return pd.read_parquet(path)


def sample_articles(split: str, n: int, seed: int) -> list[dict]:
    """
    A reproducible random sample of n articles: [{id, article, reference, source}].

    The whole split is shuffled once with the seed and the first n are taken,
    so a smaller run is always a subset of a larger one with the same seed
    (e.g. BART on 100 articles = the first 100 of the 500 extractive ones),
    which allows paired comparisons on identical articles.
    """
    df = load_split(split)
    order = list(range(len(df)))
    random.Random(seed).shuffle(order)
    indices = order[: min(n, len(df))]
    return [
        {
            "id": df.iloc[i]["id"],
            "article": df.iloc[i]["article"],
            # Highlights are one per line, tokenized as "word ." in this dataset.
            "reference": df.iloc[i]["highlights"].replace(" .", "."),
            "source": "cnn" if "(CNN)" in df.iloc[i]["article"][:100] else "dailymail",
        }
        for i in indices
    ]


class ResultWriter:
    """Append rows to a CSV and remember which (id, key) pairs are already done."""

    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[tuple[str, str]] = set()
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.done.add((row["id"], row["method"]))
        else:
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fields).writeheader()

    def is_done(self, article_id: str, method: str) -> bool:
        return (article_id, method) in self.done

    def write(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, self.fields, extrasaction="ignore").writerow(row)
        self.done.add((row["id"], row["method"]))


def bootstrap_ci(values, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high): mean with a 95 % percentile-bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def log(message: str) -> None:
    print(message, flush=True)
