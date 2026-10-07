"""
Guards against performance regressions in the extractive methods.

An earlier version compared sentences pairwise in Python during selection and
took ~160 s on a 4,000-sentence document; these limits are generous
(measured: ~0.5 s / ~3 s at 4,100 sentences) but catch that kind of mistake.
"""

import random
import time

import pytest

from app.summarizers.textrank import TextRankSummarizer
from app.summarizers.tfidf import TFIDFSummarizer

def _word(rng: random.Random) -> str:
    # Purely alphabetic: the tokenizer ignores digits, so "term123" would
    # collapse every word to "term".
    return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(4, 9)))


VOCAB = sorted({_word(random.Random(i)) for i in range(3000)})


def test_dense_graph_is_sparsified():
    # Worst case: every sentence shares vocabulary with every other sentence,
    # which would give a complete graph of n(n-1)/2 edges.
    sentences = [f"solar energy grid capacity report number {chr(97 + i % 26)}." for i in range(300)]
    result = TextRankSummarizer(max_neighbors=20).summarize(sentences)
    assert result.metadata["graph"]["edges"] <= 300 * 20
    assert result.metadata["graph"]["edges"] < 300 * 299 // 2


@pytest.fixture(scope="module")
def long_document():
    rng = random.Random(0)
    return [" ".join(rng.choices(VOCAB, k=rng.randint(8, 30))) + "." for _ in range(2500)]


@pytest.mark.parametrize("summarizer, limit", [(TFIDFSummarizer(), 5.0), (TextRankSummarizer(), 10.0)], ids=["tfidf", "textrank"])
def test_long_document_is_fast(summarizer, limit, long_document):
    start = time.perf_counter()
    result = summarizer.summarize(long_document, "long")
    elapsed = time.perf_counter() - start
    assert len(result.selected_indices) == 800  # ceil(2500 * 0.32)
    assert elapsed < limit, f"{summarizer.name} took {elapsed:.1f}s"
