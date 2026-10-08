| Method | Articles | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Summary words | Compression | Time / article |
|---|---|---|---|---|---|---|---|---|
| chunk_recursive | 6 | 43.65 (36.3–51.3) | 19.41 (11.4–29.3) | 27.20 (22.0–32.6) | 41.78 (34.2–49.6) | 111 | 90.4 % | 189.21 s |
| chunk_sentence | 6 | 46.09 (40.2–51.3) | 20.32 (12.2–30.4) | 27.86 (22.8–32.5) | 43.83 (37.7–49.6) | 110 | 90.3 % | 182.28 s |

Paired differences (A − B, same articles), ROUGE-1 / ROUGE-2 F1 with 95 % CI:

| A vs B | Articles | ΔROUGE-1 | ΔROUGE-2 |
|---|---|---|---|
| chunk_recursive vs chunk_sentence | 6 | -2.44 (-6.5 to +0.9) | -0.92 (-2.0 to +0.1) |

\* 95 % confidence interval excludes zero (only marked with ≥ 30 paired articles).
