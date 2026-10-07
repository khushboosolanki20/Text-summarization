"""
Sentence vectorisation shared by the TF-IDF and TextRank summarizers.

Each sentence is represented as a TF-IDF vector over stemmed content words
(see ``app.preprocessing.tokenizer``). With L2-normalised rows, the dot product
of two sentence vectors is their **cosine similarity**:

    cos(a, b) = (a . b) / (|a| |b|) = a . b      when |a| = |b| = 1

which ranges from 0 (no shared terms) to 1 (same terms in the same proportions).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.preprocessing.tokenizer import tokenize


def make_tfidf_vectorizer(norm: str | None = "l2") -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=tokenize,  # stop-word removal + Porter stemming
        token_pattern=None,  # silence warning: we pass our own tokenizer
        lowercase=False,  # tokenize() already lowercases
        sublinear_tf=True,  # tf = 1 + log(count): repetition has diminishing returns
        norm=norm,
    )


def cosine_similarity_fn(matrix):
    """
    Return similarity(i) -> cosine similarities of sentence i to every
    sentence, for an L2-normalised sparse sentence matrix. One sparse
    matrix-vector product per call (see ``app.summarizers.base.SimilarityFn``).
    """
    matrix = matrix.tocsr()

    def similarity(i: int) -> np.ndarray:
        # sparse matrix x dense vector is far faster than sparse x sparse
        return matrix @ matrix[i].toarray().ravel()

    return similarity
