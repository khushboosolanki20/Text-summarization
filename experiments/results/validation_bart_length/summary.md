| Method | Articles | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Summary words | Compression | Time / article |
|---|---|---|---|---|---|---|---|---|
| bart_default_56_142 | 30 | 49.05 (45.0–53.0) | 24.82 (20.2–29.5) | 34.50 (30.5–38.6) | 45.79 (41.7–49.8) | 59 | 86.1 % | 26.30 s |
| bart_ratio | 30 | 46.34 (42.2–50.5) | 24.09 (19.4–29.3) | 33.39 (28.8–38.2) | 43.40 (39.1–47.6) | 52 | 89.2 % | 25.38 s |

Paired differences (A − B, same articles), ROUGE-1 / ROUGE-2 F1 with 95 % CI:

| A vs B | Articles | ΔROUGE-1 | ΔROUGE-2 |
|---|---|---|---|
| bart_default_56_142 vs bart_ratio | 30 | +2.72 (+0.1 to +5.7) * | +0.73 (-1.6 to +3.1) |

\* 95 % confidence interval excludes zero (only marked with ≥ 30 paired articles).
