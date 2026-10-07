import pytest

from app.config import get_settings
from app.core.workflow import run_summarization
from app.errors import InvalidReferenceError
from app.evaluation import faithfulness as faithfulness_module
from app.evaluation.faithfulness import check_faithfulness
from app.evaluation.metrics import ROUGE_UNAVAILABLE, evaluate_summary, rouge_metrics
from app.evaluation.rouge import compute_rouge, lcs_length, rouge_l_reference, rouge_n_reference, rouge_or_none
from app.graph.summarization_graph import build_summarization_graph
from app.summarizers.bart import BARTSummarizer
from tests.test_hybrid import StandInModel
from tests.test_tfidf import SOLAR

PAIRS = [
    ("The cat sat on the mat. It was happy.", "The cat was sitting on a mat. The cat was happy."),
    ("Solar capacity grew by fifty percent.", "Solar power capacity grew fifty percent last year."),
    ("Completely different words here.", "Nothing in common at all."),
    ("the the the cat", "the cat the cat"),  # clipping: repeated words count at most min(count) times
]

# ------------------------------------------------------------------- ROUGE


@pytest.mark.parametrize("reference, candidate", PAIRS)
def test_from_scratch_rouge_matches_package(reference, candidate):
    package = compute_rouge(candidate, reference)
    for n, name in ((1, "rouge1"), (2, "rouge2")):
        p, r, f = rouge_n_reference(candidate, reference, n)
        assert (round(p, 4), round(r, 4), round(f, 4)) == tuple(package[name][k] for k in ("precision", "recall", "f1"))
    p, r, f = rouge_l_reference(candidate, reference)
    assert (round(p, 4), round(r, 4), round(f, 4)) == tuple(package["rougeL"][k] for k in ("precision", "recall", "f1"))


def test_identical_and_disjoint_texts():
    text = "Solar power capacity grew faster than any other source."
    assert all(compute_rouge(text, text)[name]["f1"] == 1.0 for name in ("rouge1", "rouge2", "rougeL", "rougeLsum"))
    assert compute_rouge("alpha beta gamma", "delta epsilon zeta")["rouge1"]["f1"] == 0.0


def test_precision_and_recall_differ_for_short_candidates():
    # A short candidate fully contained in the reference: perfect precision, low recall.
    scores = compute_rouge("solar capacity grew", "solar capacity grew fast last year in china")
    assert scores["rouge1"]["precision"] == 1.0
    assert scores["rouge1"]["recall"] < 0.5


def test_stemming_and_case_are_normalised():
    assert compute_rouge("The model Summarizes document", "the models summarized documents")["rouge1"]["recall"] == 1.0


def test_clipping_counts_repeated_words_at_most_min_times():
    # The reference has "the" twice, the candidate once: 4 of 5 reference words matched.
    assert compute_rouge("The model summarizes documents", "the models summarized the document")["rouge1"]["recall"] == 0.8


def test_lcs_length():
    assert lcs_length("a b c d e".split(), "a c e".split()) == 3
    assert lcs_length("a b c".split(), "c b a".split()) == 1
    assert lcs_length([], ["a"]) == 0


def test_rouge_l_rewards_word_order_but_rouge_1_does_not():
    reference = "police arrested the suspect yesterday"
    reordered = "yesterday the suspect arrested police"
    scores = compute_rouge(reordered, reference)
    assert scores["rouge1"]["f1"] == 1.0
    assert scores["rougeL"]["f1"] < 0.5


def test_no_reference_means_no_score():
    assert rouge_or_none("A summary.", None) is None
    assert rouge_or_none("A summary.", "   ") is None
    with pytest.raises(InvalidReferenceError):
        compute_rouge("A summary.", "")


def test_rouge_metrics_fields():
    with_ref = rouge_metrics("Solar capacity grew.", "Solar capacity grew fast.")
    assert 0 < with_ref["rouge1"] <= 1 and with_ref["rouge"]["rougeLsum"]["f1"] > 0
    assert with_ref["rouge_note"] is None
    without = rouge_metrics("Solar capacity grew.", None)
    assert without["rouge1"] is None and without["rouge2"] is None and without["rougeL"] is None
    assert without["rouge"] is None and without["rouge_note"] == ROUGE_UNAVAILABLE


def test_evaluate_summary_combines_statistics_and_rouge():
    metrics = evaluate_summary("one two three", 12, 4, 1, 0.5, reference_summary="one two three four")
    assert metrics["summary_word_count"] == 3 and metrics["compression_ratio"] == 75.0
    assert metrics["rouge1"] == compute_rouge("one two three", "one two three four")["rouge1"]["f1"]


# ------------------------------------------------------------ faithfulness

SUMMARY = (
    "Solar power capacity grew faster than any other energy source last year. "  # faithful
    "Solar installations rose by nearly 15 percent globally. "  # wrong number (source: 50)
    "The World Bank said Germany led the growth. "  # invented names
    "The football team won the championship after a dramatic penalty shootout."  # unrelated
)


@pytest.fixture(scope="module")
def report():
    return check_faithfulness(SUMMARY, SOLAR)


def test_faithful_sentence_is_not_flagged(report):
    first = report["sentences"][0]
    assert not first["flagged"]
    assert first["coverage"] == 1.0 and first["closest_source"] == [0]


def test_wrong_number_is_flagged_even_with_full_word_overlap(report):
    second = report["sentences"][1]
    assert second["coverage"] == 1.0  # word overlap alone would miss it
    assert second["flagged"] and second["unsupported_numbers"] == ["15"]


def test_invented_names_are_flagged(report):
    third = report["sentences"][2]
    assert third["flagged"]
    entities = " | ".join(third["unsupported_entities"])
    assert "World Bank" in entities and "Germany" in entities


def test_unrelated_sentence_is_flagged(report):
    fourth = report["sentences"][3]
    assert fourth["flagged"] and fourth["coverage"] == 0.0
    assert fourth["closest_source"] == []  # nothing similar in the source


def test_report_is_labelled_experimental(report):
    assert report["flagged_count"] == 3
    assert "experimental" in report["label"].lower()
    assert "not a guarantee" in report["note"]


def test_copied_sentences_are_never_flagged():
    report = check_faithfulness(" ".join(SOLAR[:3]), SOLAR)
    assert report["flagged_count"] == 0


def test_works_without_ner_model(monkeypatch):
    monkeypatch.setattr(faithfulness_module, "_ner_pipeline", lambda: None)
    report = check_faithfulness("The World Bank praised solar capacity growth.", SOLAR)
    assert report["sentences"][0]["unsupported_entities"] == []


# ---------------------------------------------------------- in the workflow

TEXT = " ".join(SOLAR)
REFERENCE = "Solar capacity grew faster than any other source, driven by falling panel prices. China led installations."


@pytest.fixture
def graph():
    return build_summarization_graph(abstractive=BARTSummarizer(model=StandInModel()))


def test_workflow_computes_rouge_with_reference(graph):
    out = run_summarization(TEXT, "textrank", "medium", reference_summary=REFERENCE, graph=graph)
    assert out.metrics["rouge1"] == compute_rouge(out.summary, REFERENCE)["rouge1"]["f1"]
    assert out.metrics["rouge_note"] is None


def test_workflow_returns_null_rouge_without_reference(graph):
    out = run_summarization(TEXT, "textrank", "medium", graph=graph)
    assert out.metrics["rouge1"] is None and out.metrics["rouge_note"] == ROUGE_UNAVAILABLE


def test_faithfulness_only_for_abstractive_methods(graph):
    extractive = run_summarization(TEXT, "tfidf", graph=graph)
    assert extractive.faithfulness["applicable"] is False
    abstractive = run_summarization(TEXT, "bart", graph=graph)
    assert abstractive.faithfulness["applicable"] is True
    assert abstractive.faithfulness["flagged_count"] == 0  # the stand-in copies source words


def test_faithfulness_can_be_disabled(graph, monkeypatch):
    monkeypatch.setattr(get_settings(), "faithfulness_check", False)
    out = run_summarization(TEXT, "bart", graph=graph)
    assert out.faithfulness["applicable"] is False and "disabled" in out.faithfulness["reason"]
