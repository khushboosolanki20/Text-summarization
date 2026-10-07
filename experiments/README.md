# Experiments

Offline comparison of the summarization methods on CNN/DailyMail (implemented in Phase 14).

```
experiments/
├── scripts/     Reproducible evaluation scripts (CLI)
├── notebooks/   Exploratory analysis and charts
└── results/     Generated CSV / JSON results (committed so they can be cited)
```

Setup (inside the backend virtual environment):

```bash
pip install -r experiments/requirements.txt
```

Only a small, configurable subset of the dataset is downloaded. See `docs/experiments.md`.
