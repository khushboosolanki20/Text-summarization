"""
TextRank extractive summarization (Mihalcea & Tarau, 2004).

Idea
----
Google's PageRank ranks a web page as important if many important pages
link to it. TextRank applies the same idea to sentences: a sentence is
important if it is *similar to many other important sentences*, i.e. it
expresses content that the rest of the document keeps coming back to.

Pipeline
--------
1. **Vectorise**: each sentence becomes an L2-normalised TF-IDF vector
   (stop words removed, words stemmed).
2. **Pairwise similarity**: cosine similarity between every pair of
   sentences, computed as one sparse matrix product  S = X Xᵀ.
3. **Graph construction**: an undirected, weighted graph where each sentence
   is a node and two sentences are joined by an edge of weight
   ``S[i, j]`` if their similarity exceeds a small threshold. Self-loops are
   excluded (a sentence cannot "vote" for itself). On long documents each
   sentence keeps only its 50 strongest edges (k-nearest-neighbour graph).
4. **PageRank** (NetworkX): iteratively compute

       PR(i) = (1 - d) / N  +  d * Σ_j  [ w(j, i) / Σ_k w(j, k) ] * PR(j)

   where the sum runs over the neighbours j of i, w are edge weights and d is
   the damping factor (0.85). Each sentence distributes its score to its
   neighbours in proportion to how similar they are. The ``(1 - d) / N``
   term models a "random jump" so that scores are well defined even for
   disconnected sentences. Iteration stops when scores change by less
   than the tolerance (the result is the stationary distribution of a random
   walk over the sentence graph).
5. **Rank, select top-k, restore original order**: shared with TF-IDF
   (see ``ExtractiveSummarizer``).

Difference from TF-IDF centroid scoring: the centroid compares each sentence
with the *average* of the document, whereas TextRank uses the *structure* of
sentence-to-sentence links, so a sentence that is strongly connected to a
group of central sentences is promoted even if it shares few words with the
overall average.
"""

import networkx as nx
import numpy as np
from scipy import sparse

from app.config import get_settings
from app.preprocessing.vectorizer import cosine_similarity_fn, make_tfidf_vectorizer
from app.summarizers.base import ExtractiveSummarizer, SimilarityFn


class TextRankSummarizer(ExtractiveSummarizer):
    name = "textrank"

    def __init__(
        self,
        damping: float | None = None,
        similarity_threshold: float | None = None,
        max_neighbors: int | None = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        top_edges: int = 30,
    ):
        settings = get_settings()
        self.damping = settings.textrank_damping if damping is None else damping
        self.similarity_threshold = (
            settings.textrank_similarity_threshold if similarity_threshold is None else similarity_threshold
        )
        self.max_neighbors = settings.textrank_max_neighbors if max_neighbors is None else max_neighbors
        if not 0 < self.damping < 1:
            raise ValueError("damping must be between 0 and 1")
        self.max_iter = max_iter
        self.tol = tol
        self.top_edges = top_edges

    # ------------------------------------------------------------------ graph

    def similarity_matrix(self, matrix) -> sparse.csr_matrix:
        """Step 2: cosine similarity of all sentence pairs, below-threshold pairs removed."""
        sim = (matrix @ matrix.T).tocsr()  # rows are unit vectors -> dot product = cosine
        sim = (sim - sparse.diags(sim.diagonal())).tocsr()  # remove self-similarity (no self-loops)
        sim.data[sim.data <= self.similarity_threshold] = 0
        sim.eliminate_zeros()
        if self.max_neighbors and sim.shape[0] > self.max_neighbors + 1:
            sim = self._keep_nearest_neighbors(sim)
        return sim

    def _keep_nearest_neighbors(self, sim: sparse.csr_matrix) -> sparse.csr_matrix:
        """
        k-nearest-neighbour sparsification: each sentence keeps only its
        ``max_neighbors`` strongest edges. Without this, a long document whose
        sentences all share vocabulary forms a near-complete graph
        (n^2 / 2 edges: 3 million for 2,500 sentences), which is slow and
        memory-hungry. An edge survives if *either* endpoint keeps it, so the
        graph stays undirected and no sentence loses its best connections.
        """
        k = self.max_neighbors
        rows, cols, vals = [], [], []
        for i in range(sim.shape[0]):
            start, end = sim.indptr[i], sim.indptr[i + 1]
            data, idx = sim.data[start:end], sim.indices[start:end]
            if len(data) > k:
                keep = np.argpartition(-data, k - 1)[:k]  # indices of the k largest
                data, idx = data[keep], idx[keep]
            rows.extend([i] * len(data))
            cols.extend(idx.tolist())
            vals.extend(data.tolist())
        pruned = sparse.csr_matrix((vals, (rows, cols)), shape=sim.shape)
        return pruned.maximum(pruned.T).tocsr()

    def build_graph(self, sim: sparse.csr_matrix) -> nx.Graph:
        """Step 3: weighted undirected sentence graph."""
        graph = nx.Graph()
        graph.add_nodes_from(range(sim.shape[0]))  # isolated sentences are still nodes
        coo = sparse.triu(sim, k=1).tocoo()  # each undirected pair once (i < j)
        graph.add_weighted_edges_from(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()))
        return graph

    def rank(self, graph: nx.Graph) -> tuple[dict[int, float], bool]:
        """Step 4: PageRank scores, plus whether the power iteration converged."""
        try:
            return nx.pagerank(graph, alpha=self.damping, weight="weight", max_iter=self.max_iter, tol=self.tol), True
        except nx.PowerIterationFailedConvergence:
            # Practically never happens with d < 1, but if it does, fall back to
            # weighted degree centrality: the sum of a sentence's similarities.
            degree = dict(graph.degree(weight="weight"))
            total = sum(degree.values()) or 1.0
            return {node: value / total for node, value in degree.items()}, False

    # --------------------------------------------------------------- scoring

    def score_sentences(self, sentences: list[str]) -> tuple[list[float], SimilarityFn | None, dict]:
        n = len(sentences)
        try:
            matrix = make_tfidf_vectorizer(norm="l2").fit_transform(sentences)  # step 1
        except ValueError:
            # Only stop words / numbers: no vocabulary, so no edges can exist.
            return [1.0 / n] * n, None, {"fallback": "empty_vocabulary"}

        sim = self.similarity_matrix(matrix)
        graph = self.build_graph(sim)
        pagerank, converged = self.rank(graph)
        scores = [float(pagerank[i]) for i in range(n)]

        edges = graph.number_of_edges()
        strongest = sorted(graph.edges(data="weight"), key=lambda e: -e[2])[: self.top_edges]
        metadata = {
            "graph": {
                "nodes": n,
                "edges": edges,
                "density": round(nx.density(graph), 4) if n > 1 else 0.0,
                "average_degree": round(2 * edges / n, 2),
                "isolated_sentences": nx.number_of_isolates(graph),
                "similarity_threshold": self.similarity_threshold,
                "max_neighbors": self.max_neighbors,
                "damping": self.damping,
                "converged": converged,
            },
            # The strongest links, e.g. for drawing the sentence graph in the UI.
            "top_edges": [{"source": i, "target": j, "weight": round(w, 4)} for i, j, w in strongest],
        }
        if edges == 0:
            # No sentence shares vocabulary with another: PageRank is uniform,
            # so selection falls back to the leading sentences.
            metadata["fallback"] = "no_edges"
        return scores, cosine_similarity_fn(matrix), metadata


def pagerank_reference(weights: np.ndarray, damping: float = 0.85, iterations: int = 200) -> np.ndarray:
    """
    A from-scratch power-iteration PageRank on a dense weight matrix.

    Not used by the summarizer (NetworkX is used there); kept so the formula
    above can be checked step by step and is verified against NetworkX in
    the tests.
    """
    n = weights.shape[0]
    out_strength = weights.sum(axis=1)
    # Row-normalise: transition[j, i] = probability of moving from j to i.
    # Sentences with no edges ("dangling") jump uniformly to any sentence.
    transition = np.where(out_strength[:, None] > 0, weights / np.where(out_strength == 0, 1, out_strength)[:, None], 1.0 / n)
    scores = np.full(n, 1.0 / n)
    for _ in range(iterations):
        scores = (1 - damping) / n + damping * transition.T @ scores
    return scores / scores.sum()
