"""
ROUGE evaluation (Recall-Oriented Understudy for Gisting Evaluation, Lin 2004).

ROUGE compares a generated ("candidate") summary with a human-written
**reference** summary by counting overlapping units:

* **ROUGE-1**: overlap of single words (unigrams), i.e. how much of the
  reference *content* the candidate covers.
* **ROUGE-2**: overlap of word pairs (bigrams), which rewards matching
  *phrasing* and local word order.
* **ROUGE-L**: the Longest Common Subsequence (LCS): the longest sequence of
  words appearing in both texts in the same order, not necessarily
  contiguously. It rewards sentence-level word order without requiring exact
  phrases.
* **ROUGE-Lsum**: ROUGE-L computed per sentence and combined ("summary-level"
  LCS). This is the variant most published CNN/DailyMail results report, so it
  is included for comparability.

For each, with overlap counted with clipping (a word counts at most as often
as it appears in the other text):

    precision = overlap / units in candidate   (how much of the candidate is relevant)
    recall    = overlap / units in reference   (how much of the reference is covered)
    F1        = 2 * P * R / (P + R)

Text is lowercased, punctuation removed and words Porter-stemmed before
matching ("summarizes" == "summarized").

Limitations: ROUGE measures *word overlap*, not meaning. A correct paraphrase
using different words scores low; a fluent summary that copies reference
words but states something false can score high. It needs a human reference,
so it **cannot be computed for an arbitrary user document**: without a
reference this module returns ``None`` rather than a made-up score.

The scores themselves are computed with Google's ``rouge-score`` package (the
reference implementation used in research). ``rouge_n_reference`` and
``rouge_l_reference`` below re-implement the formulas from scratch so they
can be read and checked; tests verify they match the package.
"""

from collections import Counter

from rouge_score import rouge_scorer
from rouge_score.tokenizers import DefaultTokenizer

from app.errors import InvalidReferenceError
from app.preprocessing.sentence_splitter import split_sentences

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")

_scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
_tokenizer = DefaultTokenizer(use_stemmer=True)


def _sentence_lines(text: str) -> str:
    """ROUGE-Lsum expects one sentence per line."""
    return "\n".join(split_sentences(text, min_words=1)) or text


def compute_rouge(candidate: str, reference: str) -> dict[str, dict[str, float]]:
    """
    ROUGE-1/2/L/Lsum of ``candidate`` against ``reference``.

    Returns ``{"rouge1": {"precision": p, "recall": r, "f1": f}, ...}`` with
    values in [0, 1].
    """
    if not reference or not reference.strip():
        raise InvalidReferenceError()
    scores = _scorer.score(_sentence_lines(reference), _sentence_lines(candidate or ""))
    return {
        name: {
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "f1": round(score.fmeasure, 4),
        }
        for name, score in scores.items()
    }


def rouge_or_none(candidate: str, reference: str | None) -> dict[str, dict[str, float]] | None:
    """ROUGE if a reference is available, otherwise ``None`` (never an invented score)."""
    if reference is None or not reference.strip():
        return None
    return compute_rouge(candidate, reference)


# ---------------------------------------------------------------------------
# From-scratch reference implementations (for explanation and verification).
# ---------------------------------------------------------------------------


def _prf(overlap: int, candidate_total: int, reference_total: int) -> tuple[float, float, float]:
    precision = overlap / candidate_total if candidate_total else 0.0
    recall = overlap / reference_total if reference_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def rouge_n_reference(candidate: str, reference: str, n: int) -> tuple[float, float, float]:
    """ROUGE-N: clipped n-gram overlap. Returns (precision, recall, F1)."""
    cand, ref = _tokenizer.tokenize(candidate), _tokenizer.tokenize(reference)
    cand_ngrams = Counter(tuple(cand[i : i + n]) for i in range(len(cand) - n + 1))
    ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
    # Clipping: an n-gram counts at most min(count in candidate, count in reference) times.
    overlap = sum((cand_ngrams & ref_ngrams).values())
    return _prf(overlap, sum(cand_ngrams.values()), sum(ref_ngrams.values()))


def lcs_length(a: list[str], b: list[str]) -> int:
    """
    Length of the longest common subsequence, by dynamic programming:
    table[i][j] = LCS of a[:i] and b[:j]; if a[i-1] == b[j-1] the LCS grows by
    one, otherwise it is the better of skipping a word in either sequence.
    """
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            current[j] = previous[j - 1] + 1 if a[i - 1] == b[j - 1] else max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l_reference(candidate: str, reference: str) -> tuple[float, float, float]:
    """ROUGE-L: LCS-based precision, recall and F1."""
    cand, ref = _tokenizer.tokenize(candidate), _tokenizer.tokenize(reference)
    return _prf(lcs_length(cand, ref), len(cand), len(ref))
