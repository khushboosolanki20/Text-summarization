"""
TF-IDF extractive summarization.

Idea
----
Treat every sentence as a small "document" and describe it as a vector of
TF-IDF term weights:

    tf(t, s)  = how often term t occurs in sentence s   (we use 1 + log(count))
    idf(t)    = log((1 + n) / (1 + df(t))) + 1,  df(t) = number of sentences containing t
    w(t, s)   = tf(t, s) * idf(t),  then each sentence vector is L2-normalised

A term gets a high weight in a sentence when it occurs there but is *not*
spread across every sentence, i.e. it is specific and informative.

Sentence importance (two strategies, chosen with ``scoring``)
-------------------------------------------------------------
* ``centroid`` (default): average all sentence vectors to get the document
  centroid, a single vector describing what the whole document is about.
  Terms that recur across many sentences dominate it. A sentence's score is
  its cosine similarity to the centroid: sentences that talk about the
  document's main topics score highest (Radev et al., 2004).

* ``mean``: a sentence's score is the average weight of the terms it
  contains, computed with *un-normalised* weights. This rewards sentences
  that are dense in distinctive terms, but can over-reward a short sentence
  containing one rare word.

Pipeline
--------
sentences -> tokenize/stem -> TF-IDF matrix -> sentence scores -> rank ->
top-k with redundancy control -> restore original order -> summary
"""

import numpy as np

from app.preprocessing.tokenizer import StemDisplayMap
from app.preprocessing.vectorizer import cosine_similarity_fn, make_tfidf_vectorizer
from app.summarizers.base import ExtractiveSummarizer, SimilarityFn

SCORING_STRATEGIES = ("centroid", "mean")


class TFIDFSummarizer(ExtractiveSummarizer):
    name = "tfidf"

    def __init__(self, scoring: str = "centroid", top_keywords: int = 10):
        if scoring not in SCORING_STRATEGIES:
            raise ValueError(f"scoring must be one of {SCORING_STRATEGIES}")
        self.scoring = scoring
        self.top_keywords = top_keywords

    def score_sentences(self, sentences: list[str]) -> tuple[list[float], SimilarityFn | None, dict]:
        vectorizer = make_tfidf_vectorizer(norm="l2")
        try:
            # matrix: (n_sentences x n_terms), sparse, rows L2-normalised
            matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # Every sentence consisted only of stop words / numbers: there is
            # no vocabulary to score with. Fall back to equal scores, which
            # makes selection keep the leading sentences.
            return [0.0] * len(sentences), None, {"scoring": self.scoring, "fallback": "empty_vocabulary"}

        terms = vectorizer.get_feature_names_out()
        centroid = np.asarray(matrix.mean(axis=0)).ravel()

        if self.scoring == "centroid":
            # cosine(s, c) = (s . c) / (|s| |c|); |s| = 1 after L2 normalisation
            norm = np.linalg.norm(centroid)
            scores = (matrix @ centroid) / norm if norm > 0 else np.zeros(len(sentences))
        else:  # "mean"
            raw = make_tfidf_vectorizer(norm=None).fit_transform(sentences)
            term_counts = np.diff(raw.indptr)  # non-zero terms per row (CSR format)
            sums = np.asarray(raw.sum(axis=1)).ravel()
            scores = np.divide(sums, term_counts, out=np.zeros(len(sentences)), where=term_counts > 0)

        similarity = cosine_similarity_fn(matrix)
        display = StemDisplayMap(sentences)
        top = np.argsort(-centroid)[: self.top_keywords]
        keywords = [{"term": display.display(terms[i]), "weight": round(float(centroid[i]), 4)} for i in top]

        metadata = {"scoring": self.scoring, "vocabulary_size": len(terms), "top_keywords": keywords}
        return [float(s) for s in scores], similarity, metadata
