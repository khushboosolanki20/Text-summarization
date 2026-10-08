"""
Aggregate per-article results into tables and charts.

For every results folder under experiments/results/ containing per_article.csv:
  summary.csv / summary.json   per-method means with 95 % bootstrap CIs
  paired.csv                   paired differences between methods on the same articles
  rouge_f1.png, time.png       charts
  summary.md                   markdown tables (pasted into docs/experiments.md)

Usage:  python experiments/scripts/summarize_results.py [folder ...]
"""

import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from common import RESULTS_DIR, bootstrap_ci, log

KNOWN_ORDER = ["lead3", "tfidf", "textrank", "tfidf_top3", "textrank_top3", "bart", "hybrid", "t5", "pegasus"]
NAMES = {"lead3": "Lead-3", "tfidf": "TF-IDF", "textrank": "TextRank", "tfidf_top3": "TF-IDF top-3",
         "textrank_top3": "TextRank top-3", "bart": "BART", "hybrid": "Hybrid", "t5": "T5", "pegasus": "PEGASUS"}


def method_order(methods) -> list[str]:
    """Known methods in a fixed order, then experiment variants (e.g. hybrid_x2) alphabetically."""
    methods = set(methods)
    return [m for m in KNOWN_ORDER if m in methods] + sorted(methods - set(KNOWN_ORDER))


def name(method: str) -> str:
    return NAMES.get(method, method)
METRICS = [("rouge1_f", "ROUGE-1"), ("rouge2_f", "ROUGE-2"), ("rougeL_f", "ROUGE-L"), ("rougeLsum_f", "ROUGE-Lsum")]
# Validated palette (see frontend/src/utils/chartTheme.js): one series -> slot 1.
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#898781", "#e1e0d9"
MIN_ARTICLES_FOR_SIGNIFICANCE = 30


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in method_order(df.method):
        sub = df[(df.method == method) & (df.status == "ok")]
        row = {"method": method, "articles": len(sub), "errors": int(((df.method == method) & (df.status == "error")).sum())}
        for col, _ in METRICS:
            mean, low, high = bootstrap_ci(100 * sub[col])
            row[col], row[f"{col}_ci_low"], row[f"{col}_ci_high"] = round(mean, 2), round(low, 2), round(high, 2)
        row["summary_words"] = round(sub.summary_words.mean(), 1)
        row["compression_ratio"] = round(sub.compression_ratio.mean(), 1)
        row["seconds_mean"] = round(sub.seconds.mean(), 3)
        row["seconds_median"] = round(sub.seconds.median(), 3)
        flagged = pd.to_numeric(sub.get("flagged_sentences"), errors="coerce")
        row["articles_with_flags"] = int((flagged > 0).sum()) if flagged.notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ROUGE-1/2 F1 difference (A - B) on articles both methods completed, with 95 % CI."""
    ok = df[df.status == "ok"]
    wide = {col: ok.pivot_table(index="id", columns="method", values=col) for col in ("rouge1_f", "rouge2_f")}
    methods = method_order(ok.method)
    rows = []
    for a, b in combinations(methods, 2):
        row = {"a": a, "b": b}
        for col, table in wide.items():
            both = table[[a, b]].dropna()
            mean, low, high = bootstrap_ci(100 * (both[a] - both[b]))
            row["articles"] = len(both)
            row[f"{col}_diff"], row[f"{col}_ci_low"], row[f"{col}_ci_high"] = round(mean, 2), round(low, 2), round(high, 2)
            # "Significant" at 95 % if the CI excludes zero; bootstrap CIs are
            # unreliable for tiny samples, so require at least 30 articles.
            row[f"{col}_significant"] = bool((low > 0 or high < 0) and len(both) >= MIN_ARTICLES_FOR_SIGNIFICANCE)
        rows.append(row)
    return pd.DataFrame(rows)


def charts(summary: pd.DataFrame, folder) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [name(m) for m in summary.method]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})

    # ROUGE F1: one small chart per metric (one axis each), means with 95 % CIs.
    width = max(3.2, 0.55 * len(labels) + 1)  # grow with the number of methods
    fig, axes = plt.subplots(1, len(METRICS), figsize=(width * len(METRICS), 3.8), sharey=True)
    for ax, (col, title) in zip(axes, METRICS):
        means = summary[col].to_numpy()
        err = np.vstack([means - summary[f"{col}_ci_low"], summary[f"{col}_ci_high"] - means])
        ax.bar(labels, means, width=0.55, color=BLUE, yerr=err, ecolor=INK, capsize=3, error_kw={"linewidth": 1})
        ax.set_title(title, color=INK, fontsize=11)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
        # Value above the top of the error bar: never clipped, never on top of a mark.
        for x, (v, top) in enumerate(zip(means, summary[f"{col}_ci_high"])):
            ax.text(x, top + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_ylim(0, max(summary[f"{c}_ci_high"].max() for c, _ in METRICS) * 1.12)
    axes[0].set_ylabel("F1 (0–100), mean with 95 % CI")
    fig.savefig(folder / "rouge_f1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Time per document (log scale: extractive methods are milliseconds, BART seconds).
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(labels) + 1))
    ax.barh(labels, summary.seconds_mean, color=BLUE, height=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Mean seconds per article (log scale)")
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for y, v in enumerate(summary.seconds_mean):
        ax.text(v * 1.08, y, f"{v:.2f} s" if v >= 0.1 else f"{v * 1000:.0f} ms", va="center", fontsize=8, color=INK)
    fig.savefig(folder / "time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown(summary: pd.DataFrame, pairs: pd.DataFrame, folder) -> str:
    lines = [
        "| Method | Articles | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Summary words | Compression | Time / article |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        cells = [f"{r[c]:.2f} ({r[c + '_ci_low']:.1f}–{r[c + '_ci_high']:.1f})" for c, _ in METRICS]
        t = r.seconds_mean
        lines.append(
            f"| {name(r.method)} | {r.articles} | " + " | ".join(cells)
            + f" | {r.summary_words:.0f} | {r.compression_ratio:.1f} % | {t:.2f} s |"
        )
    lines += ["", "Paired differences (A − B, same articles), ROUGE-1 / ROUGE-2 F1 with 95 % CI:", "",
              "| A vs B | Articles | ΔROUGE-1 | ΔROUGE-2 |", "|---|---|---|---|"]
    for _, p in pairs.iterrows():
        def fmt(col):
            mark = " *" if p[f"{col}_significant"] else ""
            return f"{p[col + '_diff']:+.2f} ({p[col + '_ci_low']:+.1f} to {p[col + '_ci_high']:+.1f}){mark}"
        lines.append(f"| {name(p.a)} vs {name(p.b)} | {p.articles} | {fmt('rouge1_f')} | {fmt('rouge2_f')} |")
    lines.append(f"\n\\* 95 % confidence interval excludes zero (only marked with ≥ {MIN_ARTICLES_FOR_SIGNIFICANCE} paired articles).")
    text = "\n".join(lines) + "\n"
    (folder / "summary.md").write_text(text, encoding="utf-8")
    return text


def common_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Only the articles every method completed, so all means are on identical inputs."""
    ok = df[df.status == "ok"]
    counts = ok.groupby("id").method.nunique()
    keep = counts[counts == ok.method.nunique()].index
    return df[df.id.isin(keep)]


def main() -> None:
    folders = [RESULTS_DIR / f for f in sys.argv[1:]] or sorted(p.parent for p in RESULTS_DIR.glob("*/per_article.csv"))
    for folder in folders:
        df = pd.read_csv(folder / "per_article.csv")
        # Main table: every method on the same articles. Methods with larger
        # samples (e.g. extractive on 500) also get their own full-sample table.
        common = common_articles(df)
        summary = summarize(common)
        pairs = paired(common)
        summary.to_csv(folder / "summary.csv", index=False)
        pairs.to_csv(folder / "paired.csv", index=False)
        full = summarize(df)
        full.to_csv(folder / "summary_all_articles.csv", index=False)
        (folder / "summary.json").write_text(
            json.dumps({"common_articles": summary.to_dict(orient="records"),
                        "all_articles": full.to_dict(orient="records")}, indent=2),
            encoding="utf-8",
        )
        charts(summary, folder)
        text = markdown(summary, pairs, folder)
        if not full[["method", "articles"]].equals(summary[["method", "articles"]]):
            extra = "\n".join(
                f"| {name(r.method)} | {r.articles} | {r.rouge1_f:.2f} | {r.rouge2_f:.2f} | {r.rougeL_f:.2f} | {r.rougeLsum_f:.2f} |"
                for _, r in full.iterrows()
            )
            text += ("\nAll completed articles per method (larger samples for cheaper methods):\n\n"
                     "| Method | Articles | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |\n|---|---|---|---|---|---|\n" + extra + "\n")
            (folder / "summary.md").write_text(text, encoding="utf-8")
        log(f"== {folder.name}\n{text}")


if __name__ == "__main__":
    main()
