"""
Compare summarization methods on CNN/DailyMail with ROUGE.

Methods
-------
  lead3      first three sentences (the standard news baseline)
  tfidf      TF-IDF centroid extractive summarizer
  textrank   TextRank extractive summarizer
  bart       facebook/bart-large-cnn (with chunking for long articles)
  hybrid     TextRank selection -> BART
  tfidf_top3, textrank_top3   top-3 sentences (length-controlled, same budget as Lead-3)
  t5, pegasus  optional extra seq2seq models (large downloads, not run by default)

The four IntelliSum methods run through the same LangGraph workflow as the web
app (``run_summarization``), so the experiment measures the actual product.

Every (article, method) result is appended to ``<out>/per_article.csv`` as
soon as it is computed; re-running the same command skips finished work,
so long CPU runs can be interrupted and resumed.

Examples (from the repository root, backend venv active):
    python experiments/scripts/run_comparison.py --methods lead3 tfidf textrank --n 500
    python experiments/scripts/run_comparison.py --methods bart hybrid --n 100
    python experiments/scripts/summarize_results.py
"""

import argparse
import time

from common import RESULTS_DIR, ResultWriter, log, sample_articles

from app.config import get_settings
from app.core.workflow import run_summarization
from app.errors import IntelliSumError
from app.evaluation.rouge import compute_rouge
from app.preprocessing.cleaner import count_words
from app.preprocessing.pipeline import preprocess
from app.summarizers.bart import PegasusSummarizer, T5Summarizer
from app.summarizers.base import select_top_sentences
from app.summarizers.textrank import TextRankSummarizer
from app.summarizers.tfidf import TFIDFSummarizer

WORKFLOW_METHODS = {"tfidf", "textrank", "bart", "hybrid"}
EXTRA_MODELS = {"t5": T5Summarizer, "pegasus": PegasusSummarizer}
# Length-controlled variants: the method's top 3 sentences (same budget as
# Lead-3), to separate sentence-selection quality from summary length.
TOP3 = {"tfidf_top3": TFIDFSummarizer, "textrank_top3": TextRankSummarizer}
ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")

FIELDS = (
    ["id", "source", "method", "length", "status", "error", "original_words", "summary_words", "reference_words"]
    + [f"{k}_{m}" for k in ROUGE_KEYS for m in ("p", "r", "f")]
    + ["compression_ratio", "seconds", "strategy", "chunks", "input_tokens", "flagged_sentences", "summary"]
)


def summarize(method: str, article: str, length: str, reference: str) -> dict:
    """Run one method on one article; returns the per-method fields."""
    start = time.perf_counter()
    if method == "lead3":
        pre = preprocess(article)
        summary, extra = " ".join(pre.sentences[:3]), {"strategy": "lead3", "original_words": pre.word_count}
    elif method in TOP3:
        pre = preprocess(article)
        scores, similarity, _ = TOP3[method]().score_sentences(pre.sentences)
        chosen = select_top_sentences(scores, min(3, len(scores)), similarity, get_settings().redundancy_threshold)
        summary = " ".join(pre.sentences[i] for i in chosen)
        extra = {"strategy": "extractive_top3", "original_words": pre.word_count}
    elif method in WORKFLOW_METHODS:
        out = run_summarization(article, method, length, reference_summary=reference)
        faith = out.faithfulness
        summary = out.summary
        extra = {
            "strategy": out.strategy,
            "original_words": out.original_word_count,
            "chunks": out.metadata.get("chunks", ""),
            "input_tokens": out.metadata.get("input_tokens", ""),
            "flagged_sentences": faith.get("flagged_count", "") if faith.get("applicable") else "",
        }
    else:  # t5 / pegasus
        pre = preprocess(article)
        result = EXTRA_MODELS[method]().summarize(pre.sentences, length)
        summary = result.summary
        extra = {
            "strategy": result.metadata.get("strategy"),
            "original_words": pre.word_count,
            "chunks": result.metadata.get("chunks", ""),
            "input_tokens": result.metadata.get("input_tokens", ""),
        }
    seconds = time.perf_counter() - start

    rouge = compute_rouge(summary, reference)
    row = {
        **extra,
        "summary": summary,
        "summary_words": count_words(summary),
        "seconds": round(seconds, 3),
    }
    row["compression_ratio"] = round(100 * (1 - row["summary_words"] / row["original_words"]), 2)
    for key in ROUGE_KEYS:
        row[f"{key}_p"], row[f"{key}_r"], row[f"{key}_f"] = (rouge[key][m] for m in ("precision", "recall", "f1"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["lead3", "tfidf", "textrank"],
                        choices=["lead3", *sorted(WORKFLOW_METHODS), *TOP3, *EXTRA_MODELS])
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--n", type=int, default=100, help="number of articles (random sample)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length", default="short", choices=["short", "medium", "long"])
    parser.add_argument("--out", default=None, help="results folder (default: results/<split>_<length>_seed<seed>)")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / (args.out or f"{args.split}_{args.length}_seed{args.seed}")
    writer = ResultWriter(out_dir / "per_article.csv", FIELDS)
    articles = sample_articles(args.split, args.n, args.seed)
    log(f"{len(articles)} {args.split} articles (seed {args.seed}) -> {out_dir}")

    for method in args.methods:  # method by method, so cheap methods finish first
        todo = [a for a in articles if not writer.is_done(a["id"], method)]
        log(f"[{method}] {len(articles) - len(todo)} done, {len(todo)} to go")
        started = time.perf_counter()
        for i, a in enumerate(todo, start=1):
            base = {"id": a["id"], "source": a["source"], "method": method, "length": args.length,
                    "reference_words": count_words(a["reference"])}
            try:
                writer.write({**base, "status": "ok", **summarize(method, a["article"], args.length, a["reference"])})
            except IntelliSumError as exc:  # e.g. text too short: recorded, not hidden
                writer.write({**base, "status": "error", "error": exc.message})
            if i % 10 == 0 or i == len(todo):
                rate = (time.perf_counter() - started) / i
                log(f"[{method}] {i}/{len(todo)}  ~{rate:.1f}s/article, ~{rate * (len(todo) - i) / 60:.0f} min left")


if __name__ == "__main__":
    main()
