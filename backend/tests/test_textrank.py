import networkx as nx
import numpy as np
import pytest

from app.preprocessing.vectorizer import make_tfidf_vectorizer
from app.summarizers.textrank import TextRankSummarizer, pagerank_reference
from tests.test_tfidf import OFF_TOPIC, SOLAR


@pytest.fixture
def textrank():
    return TextRankSummarizer()


# ---------------------------------------------------------------- graph


def test_similarity_matrix_is_symmetric_without_self_loops(textrank):
    matrix = make_tfidf_vectorizer().fit_transform(SOLAR)
    sim = textrank.similarity_matrix(matrix).toarray()
    assert np.allclose(sim, sim.T)
    assert np.all(np.diag(sim) == 0)
    assert sim.max() <= 1.0 + 1e-9


def test_threshold_removes_weak_edges():
    matrix = make_tfidf_vectorizer().fit_transform(SOLAR)
    loose = TextRankSummarizer(similarity_threshold=0.0).similarity_matrix(matrix)
    strict = TextRankSummarizer(similarity_threshold=0.15).similarity_matrix(matrix)
    assert 0 < strict.nnz < loose.nnz
    assert strict.data.min() > 0.15


def test_nearest_neighbor_limit_keeps_strongest_edges():
    weights = np.array(
        [[0, 0.9, 0.5, 0.1], [0.9, 0, 0.2, 0.3], [0.5, 0.2, 0, 0.4], [0.1, 0.3, 0.4, 0]], dtype=float
    )
    from scipy import sparse

    pruned = TextRankSummarizer(max_neighbors=1)._keep_nearest_neighbors(sparse.csr_matrix(weights)).toarray()
    assert np.allclose(pruned, pruned.T)  # still undirected
    # Strongest neighbours: 0->1, 1->0, 2->0, 3->2. Union of kept edges:
    assert pruned[0, 1] == 0.9 and pruned[0, 2] == 0.5 and pruned[2, 3] == 0.4
    assert pruned[0, 3] == 0 and pruned[1, 2] == 0 and pruned[1, 3] == 0


def test_short_documents_are_not_sparsified():
    matrix = make_tfidf_vectorizer().fit_transform(SOLAR)
    full = TextRankSummarizer(max_neighbors=0).similarity_matrix(matrix)
    limited = TextRankSummarizer(max_neighbors=50).similarity_matrix(matrix)
    assert (full != limited).nnz == 0


def test_graph_has_one_node_per_sentence(textrank):
    matrix = make_tfidf_vectorizer().fit_transform(SOLAR)
    graph = textrank.build_graph(textrank.similarity_matrix(matrix))
    assert graph.number_of_nodes() == len(SOLAR)
    assert not any(u == v for u, v in graph.edges())  # no self-loops
    assert all(w > 0 for _, _, w in graph.edges(data="weight"))


def test_off_topic_sentences_are_weakly_connected(textrank):
    # "Several journalists attended..." shares no content words: isolated.
    # "The weather was pleasant ... when the report was released" shares only
    # "report", so it has edges, but the weakest total connection strength.
    matrix = make_tfidf_vectorizer().fit_transform(SOLAR)
    graph = textrank.build_graph(textrank.similarity_matrix(matrix))
    assert graph.degree(7) == 0
    strength = dict(graph.degree(weight="weight"))
    assert set(sorted(strength, key=strength.get)[:2]) == OFF_TOPIC


# ------------------------------------------------------------- PageRank


def test_networkx_pagerank_matches_formula():
    # Verifies the PageRank formula documented in textrank.py against NetworkX,
    # including a dangling (isolated) node 3.
    weights = np.array([[0, 0.5, 0.2, 0], [0.5, 0, 0.8, 0], [0.2, 0.8, 0, 0], [0, 0, 0, 0]], dtype=float)
    graph = nx.from_numpy_array(weights)
    nx_scores = nx.pagerank(graph, alpha=0.85, weight="weight", tol=1e-10)
    ours = pagerank_reference(weights, damping=0.85)
    assert np.allclose([nx_scores[i] for i in range(4)], ours, atol=1e-6)


def test_most_connected_node_ranks_highest():
    # Node 1 is strongly linked to both 0 and 2; node 3 is isolated.
    weights = np.array([[0, 0.9, 0.1, 0], [0.9, 0, 0.9, 0], [0.1, 0.9, 0, 0], [0, 0, 0, 0]], dtype=float)
    scores = pagerank_reference(weights)
    assert scores.argmax() == 1
    assert scores.argmin() == 3


def test_scores_form_a_probability_distribution(textrank):
    scores = textrank.summarize(SOLAR).sentence_scores
    assert sum(scores) == pytest.approx(1.0, abs=1e-4)
    assert all(s > 0 for s in scores)  # the random-jump term keeps every score positive


# ------------------------------------------------------------ summarizer


def test_off_topic_sentences_not_selected(textrank):
    result = textrank.summarize(SOLAR, "long")
    assert len(result.selected_indices) == 4
    assert not OFF_TOPIC & set(result.selected_indices)


def test_off_topic_sentences_score_lowest(textrank):
    scores = textrank.summarize(SOLAR).sentence_scores
    ranked = sorted(range(len(scores)), key=lambda i: scores[i])
    assert set(ranked[:2]) == OFF_TOPIC


def test_result_structure(textrank):
    result = textrank.summarize(SOLAR, "medium")
    assert result.method == "textrank"
    assert len(result.sentence_scores) == len(SOLAR)
    assert result.selected_indices == sorted(result.selected_indices)
    assert result.summary == " ".join(SOLAR[i] for i in result.selected_indices)
    assert 0 < result.compression_ratio < 100
    graph = result.metadata["graph"]
    assert graph["nodes"] == 10 and graph["edges"] > 0 and graph["converged"]
    assert graph["isolated_sentences"] >= 1
    edges = result.metadata["top_edges"]
    assert edges == sorted(edges, key=lambda e: -e["weight"])


def test_longer_setting_gives_longer_summary(textrank):
    words = [textrank.summarize(SOLAR, length).summary_word_count for length in ("short", "medium", "long")]
    assert words[0] < words[1] < words[2]


def test_deterministic(textrank):
    assert textrank.summarize(SOLAR).sentence_scores == textrank.summarize(SOLAR).sentence_scores


def test_empty_and_single_sentence(textrank):
    assert textrank.summarize([]).summary == ""
    assert textrank.summarize(["Only one sentence about solar power."]).selected_indices == [0]


def test_unrelated_sentences_fall_back_to_leading(textrank):
    sentences = ["Cats sleep often.", "Rockets reach orbit.", "Bread needs yeast.", "Rivers flow downhill."]
    result = textrank.summarize(sentences)
    assert result.metadata["fallback"] == "no_edges"
    assert result.selected_indices == [0]


def test_stop_word_only_sentences(textrank):
    result = textrank.summarize(["It is what it is.", "This was that.", "They are here."])
    assert result.metadata["fallback"] == "empty_vocabulary"
    assert result.selected_indices == [0]


def test_invalid_damping():
    with pytest.raises(ValueError):
        TextRankSummarizer(damping=1.5)
