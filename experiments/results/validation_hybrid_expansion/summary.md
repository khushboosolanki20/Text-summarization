| Method | Articles | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Summary words | Compression | Time / article |
|---|---|---|---|---|---|---|---|---|
| hybrid_x1.5 | 30 | 39.06 (36.2–41.9) | 15.58 (12.2–19.2) | 26.20 (23.0–29.4) | 36.85 (34.1–39.6) | 65 | 90.0 % | 24.16 s |
| hybrid_x2 | 30 | 40.53 (37.2–43.8) | 17.18 (13.5–21.2) | 27.40 (23.4–31.4) | 37.73 (34.4–41.1) | 65 | 90.3 % | 25.38 s |
| hybrid_x3 | 30 | 42.74 (37.9–47.7) | 19.66 (14.7–25.0) | 30.02 (25.6–34.9) | 40.23 (35.7–44.7) | 66 | 90.0 % | 28.61 s |
| hybrid_x4 | 30 | 45.07 (40.9–49.1) | 22.76 (18.4–27.1) | 31.66 (27.7–35.8) | 43.11 (39.0–47.2) | 66 | 89.9 % | 40.17 s |

Paired differences (A − B, same articles), ROUGE-1 / ROUGE-2 F1 with 95 % CI:

| A vs B | Articles | ΔROUGE-1 | ΔROUGE-2 |
|---|---|---|---|
| hybrid_x1.5 vs hybrid_x2 | 30 | -1.47 (-3.6 to +0.6) | -1.60 (-4.0 to +0.3) |
| hybrid_x1.5 vs hybrid_x3 | 30 | -3.68 (-7.8 to +0.4) | -4.07 (-9.3 to +0.6) |
| hybrid_x1.5 vs hybrid_x4 | 30 | -6.01 (-9.3 to -2.4) * | -7.18 (-11.0 to -3.4) * |
| hybrid_x2 vs hybrid_x3 | 30 | -2.21 (-6.3 to +1.5) | -2.47 (-7.3 to +1.7) |
| hybrid_x2 vs hybrid_x4 | 30 | -4.54 (-7.9 to -1.4) * | -5.58 (-9.1 to -2.3) * |
| hybrid_x3 vs hybrid_x4 | 30 | -2.33 (-5.2 to +0.3) | -3.11 (-6.2 to -0.3) * |

\* 95 % confidence interval excludes zero (only marked with ≥ 30 paired articles).
