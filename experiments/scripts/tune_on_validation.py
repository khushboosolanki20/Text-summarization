"""
Design experiments on the VALIDATION split (never on test, so the reported
test results stay unbiased).

  hybrid_expansion  How many times the summary length should TextRank select
                    for BART? (1.5x, 2x, 3x, 4x)
  bart_length       Size BART summaries by the short/medium/long ratio
                    (IntelliSum) or by the 56-142-token range bart-large-cnn
                    was fine-tuned with? (articles that fit one pass)
  chunking          Long articles (> 1,024 tokens): sentence-aware chunks vs
                    LangChain's RecursiveCharacterTextSplitter.

Results go to experiments/results/validation_<experiment>/per_article.csv
(resumable) and are aggregated by summarize_results.py.

Usage:
    python experiments/scripts/tune_on_validation.py hybrid_expansion --n 40
    python experiments/scripts/tune_on_validation.py bart_length --n 40
    python experiments/scripts/tune_on_validation.py chunking --n 25
"""

import argparse
import time

from common import RESULTS_DIR, ResultWriter, log, sample_articles
from run_comparison import FIELDS, ROUGE_KEYS

from app.config import get_settings
from app.core.workflow import run_summarization
from app.errors import IntelliSumError
from app.evaluation.rouge import compute_rouge
from app.preprocessing.cleaner import count_words
from app.preprocessing.pipeline import preprocess
from app.summarizers.bart import BARTSummarizer
from app.summarizers.generation import tidy_generated_text
from app.summarizers.hybrid import HybridSummarizer
from app.summarizers.models import get_model


def scored(summary: str, reference: str, original_words: int, seconds: float, **extra) -> dict:
    rouge = compute_rouge(summary, reference)
    row = {"summary": summary, "summary_words": count_words(summary), "original_words": original_words,
           "seconds": round(seconds, 3), **extra}
    row["compression_ratio"] = round(100 * (1 - row["summary_words"] / original_words), 2)
    for key in ROUGE_KEYS:
        row[f"{key}_p"], row[f"{key}_r"], row[f"{key}_f"] = (rouge[key][m] for m in ("precision", "recall", "f1"))
    return row


def hybrid_expansion(article: dict, pre, length: str):
    bart = BARTSummarizer()
    for factor in (1.5, 2.0, 3.0, 4.0):
        start = time.perf_counter()
        result = HybridSummarizer(abstractive=bart, expansion=factor).summarize(pre.sentences, length)
        yield f"hybrid_x{factor:g}", scored(
            result.summary, article["reference"], pre.word_count, time.perf_counter() - start,
            strategy=result.metadata["abstractive_stage"].get("strategy"),
            input_tokens=result.metadata["abstractive_stage"].get("input_tokens"),
        )


def bart_length(article: dict, pre, length: str):
    bart = BARTSummarizer()
    text = " ".join(pre.sentences)
    if not bart.model.fits(text):
        return  # only single-pass articles: the comparison is about one generation
    start = time.perf_counter()
    result = bart.summarize(pre.sentences, length)
    yield "bart_ratio", scored(result.summary, article["reference"], pre.word_count, time.perf_counter() - start)
    start = time.perf_counter()
    summary, _ = tidy_generated_text(get_model("bart").generate([text], 56, 142)[0])
    yield "bart_default_56_142", scored(summary, article["reference"], pre.word_count, time.perf_counter() - start)


def chunking(article: dict, pre, length: str):
    if BARTSummarizer().model.fits(" ".join(pre.sentences)):
        return  # only articles that need chunking
    settings = get_settings()
    original = settings.chunking_strategy
    try:
        for strategy in ("sentence", "recursive"):
            settings.chunking_strategy = strategy
            start = time.perf_counter()
            out = run_summarization(article["article"], "bart", length)
            yield f"chunk_{strategy}", scored(
                out.summary, article["reference"], out.original_word_count, time.perf_counter() - start,
                strategy=out.strategy, chunks=out.metadata.get("chunks"),
            )
    finally:
        settings.chunking_strategy = original


EXPERIMENTS = {"hybrid_expansion": hybrid_expansion, "bart_length": bart_length, "chunking": chunking}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--n", type=int, default=40, help="articles to try (filters may use fewer)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--length", default="short", choices=["short", "medium", "long"])
    args = parser.parse_args()

    out_dir = RESULTS_DIR / f"validation_{args.experiment}"
    writer = ResultWriter(out_dir / "per_article.csv", FIELDS)
    articles = sample_articles("validation", args.n, args.seed)
    log(f"{args.experiment}: {len(articles)} validation articles (seed {args.seed}) -> {out_dir}")

    for i, article in enumerate(articles, start=1):
        try:
            pre = preprocess(article["article"])
        except IntelliSumError:
            continue
        for method, row in EXPERIMENTS[args.experiment](article, pre, args.length):
            if writer.is_done(article["id"], method):
                continue
            writer.write({"id": article["id"], "source": article["source"], "method": method, "length": args.length,
                          "status": "ok", "reference_words": count_words(article["reference"]), **row})
        log(f"{args.experiment}: {i}/{len(articles)}")


if __name__ == "__main__":
    main()
