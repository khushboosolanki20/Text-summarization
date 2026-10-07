import numpy as np
import pytest

from app.preprocessing.tokenizer import StemDisplayMap, tokenize
from app.summarizers.base import (
    SummaryLength,
    compression_ratio,
    select_top_sentences,
    target_sentence_count,
)
from app.summarizers.tfidf import TFIDFSummarizer

# A document about solar energy with two clearly off-topic sentences (1-based: 5 and 8).
SOLAR = [
    "Solar power capacity grew faster than any other energy source last year, according to a new energy report.",
    "The agency said solar installations rose by nearly 50 percent globally.",
    "China accounted for more than half of all new solar capacity.",
    "Analysts said falling solar panel prices were the main driver of solar growth.",
    "The weather was pleasant in Paris when the report was released.",
    "Wind power also expanded, but at a slower pace than solar power.",
    "The report warned that grid infrastructure must improve to absorb rising solar output.",
    "Several journalists attended the launch event downtown.",
    "Solar panel prices have fallen by about 80 percent over the past decade.",
    "The agency expects solar power to become the largest source of electricity by 2030.",
]
OFF_TOPIC = {4, 7}


# ------------------------------------------------------------ tokenizer


def test_tokenize_removes_stop_words_and_stems():
    assert tokenize("The models are summarizing the documents") == ["model", "summar", "document"]


def test_tokenize_conflates_word_forms():
    assert len(set(tokenize("summarize summarizes summarization summarized"))) == 1


def test_tokenize_ignores_numbers_and_punctuation():
    assert tokenize("In 2020, 92.5% -- wow!") == ["wow"]


def test_stem_display_map_returns_readable_word():
    display = StemDisplayMap(["Summarization helps.", "Summarization is useful.", "We summarize."])
    assert display.display("summar") == "summarization"
    assert display.display("unknown") == "unknown"


# ------------------------------------------------------- shared helpers


@pytest.mark.parametrize(
    "n, length, expected",
    [(10, "short", 2), (10, "medium", 3), (10, "long", 4), (100, "short", 12), (1, "long", 1), (0, "short", 0)],
)
def test_target_sentence_count(n, length, expected):
    assert target_sentence_count(n, length) == expected


def test_target_count_is_always_smaller_than_input():
    for n in range(2, 8):
        for length in SummaryLength:
            assert 1 <= target_sentence_count(n, length) < n


def test_lengths_are_monotonic():
    counts = [target_sentence_count(50, length) for length in ("short", "medium", "long")]
    assert counts == sorted(counts) and len(set(counts)) == 3


def test_compression_ratio():
    assert compression_ratio(1000, 250) == 75.0
    assert compression_ratio(0, 0) == 0.0


def test_selection_restores_document_order():
    assert select_top_sentences([0.1, 0.9, 0.3, 0.8], k=2) == [1, 3]


def test_selection_ties_prefer_earlier_sentences():
    assert select_top_sentences([0.5, 0.5, 0.5], k=1) == [0]


def test_selection_skips_redundant_sentences():
    # 0 and 1 are near-duplicates; 2 is different.
    sim = np.array([[1.0, 0.95, 0.1], [0.95, 1.0, 0.1], [0.1, 0.1, 1.0]])
    selected = select_top_sentences([0.9, 0.85, 0.5], k=2, similarity=lambda i: sim[i], redundancy_threshold=0.8)
    assert selected == [0, 2]


def test_selection_backfills_when_everything_is_redundant():
    selected = select_top_sentences([0.9, 0.8, 0.7], k=2, similarity=lambda i: np.full(3, 0.99), redundancy_threshold=0.8)
    assert selected == [0, 1]


# ------------------------------------------------------ TF-IDF summarizer


def test_off_topic_sentences_are_not_selected():
    result = TFIDFSummarizer("centroid").summarize(SOLAR, "long")
    assert len(result.selected_indices) == 4
    assert not OFF_TOPIC & set(result.selected_indices)


def test_mean_scoring_rewards_rare_words():
    # Documents a known weakness that motivates the centroid default: the
    # off-topic sentences use words that occur nowhere else (maximum IDF), so
    # averaging term weights ranks them highly.
    result = TFIDFSummarizer("mean").summarize(SOLAR, "long")
    assert OFF_TOPIC <= set(result.selected_indices)


def test_off_topic_sentences_score_lowest():
    scores = TFIDFSummarizer().summarize(SOLAR).sentence_scores
    ranked = sorted(range(len(scores)), key=lambda i: scores[i])
    assert set(ranked[:2]) == OFF_TOPIC


def test_result_structure():
    result = TFIDFSummarizer().summarize(SOLAR, SummaryLength.MEDIUM)
    assert result.method == "tfidf"
    assert len(result.sentence_scores) == len(SOLAR)
    assert result.selected_indices == sorted(result.selected_indices)
    assert result.summary == " ".join(SOLAR[i] for i in result.selected_indices)
    assert result.original_word_count == sum(len(s.split()) for s in SOLAR)
    assert result.summary_word_count == len(result.summary.split())
    assert 0 < result.compression_ratio < 100
    assert result.metadata["num_sentences"] == 10
    assert result.metadata["num_selected"] == 3
    assert result.metadata["top_keywords"][0]["term"] == "solar"


def test_longer_setting_gives_longer_summary():
    words = [TFIDFSummarizer().summarize(SOLAR, length).summary_word_count for length in ("short", "medium", "long")]
    assert words[0] < words[1] < words[2]


def test_deterministic():
    a = TFIDFSummarizer().summarize(SOLAR)
    b = TFIDFSummarizer().summarize(SOLAR)
    assert a.summary == b.summary and a.sentence_scores == b.sentence_scores


def test_empty_input():
    result = TFIDFSummarizer().summarize([])
    assert result.summary == "" and result.selected_indices == []


def test_single_sentence_is_returned_as_is():
    result = TFIDFSummarizer().summarize(["Only one sentence about solar power here."])
    assert result.selected_indices == [0]


def test_stop_word_only_sentences_fall_back_to_leading_sentences():
    result = TFIDFSummarizer().summarize(["It is what it is.", "This was that.", "They are here.", "We were there."])
    assert result.metadata["fallback"] == "empty_vocabulary"
    assert result.selected_indices == [0]


def test_near_duplicate_sentences_not_both_selected():
    sentences = SOLAR + [SOLAR[0].replace("last year", "this year")]
    result = TFIDFSummarizer().summarize(sentences, "long")
    assert not {0, len(sentences) - 1} <= set(result.selected_indices)


def test_invalid_scoring_rejected():
    with pytest.raises(ValueError):
        TFIDFSummarizer(scoring="bogus")
