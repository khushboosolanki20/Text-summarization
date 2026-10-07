import pytest

from app.config import get_settings
from app.preprocessing.sentence_splitter import get_nlp, split_sentences


def test_splits_simple_sentences(sample_text):
    sentences = split_sentences(sample_text)
    assert len(sentences) == 5
    assert sentences[0].startswith("Automatic text summarization")
    assert sentences[-1].endswith("at once.")


def test_abbreviations_do_not_end_sentences():
    text = "Dr. Smith met Prof. Jones of M.I.T. on Monday. They discussed the U.S. economy in detail."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("Dr. Smith met Prof. Jones")


def test_decimal_numbers_do_not_split():
    sentences = split_sentences("The model scored 92.5 percent on the benchmark test. It was released in 2020 publicly.")
    assert len(sentences) == 2
    assert "92.5 percent" in sentences[0]


def test_paragraph_boundary_is_a_hard_break():
    # The heading has no final punctuation; without paragraph handling it would
    # be glued onto the first sentence of the next paragraph.
    text = "Introduction to the topic\n\nSummarization is a useful technique for readers."
    sentences = split_sentences(text)
    assert sentences == ["Introduction to the topic", "Summarization is a useful technique for readers."]


def test_short_fragments_are_dropped():
    text = "Results\n\nThe proposed method clearly outperforms the baseline. Yes! See 2.1."
    assert split_sentences(text) == ["The proposed method clearly outperforms the baseline."]


def test_opening_quote_stays_with_its_sentence():
    # spaCy attaches the opening quote of the quoted sentence to the previous one.
    text = (
        "The bill was issued to the businessman Garry John Donoghue. 'Garry, if you don't pay me, "
        "I will give the office everything I have on you,' his adviser said."
    )
    sentences = split_sentences(text)
    assert sentences[0] == "The bill was issued to the businessman Garry John Donoghue."
    assert sentences[1].startswith("'Garry, if you")


def test_offsets_point_at_sentence_starts():
    from app.preprocessing.sentence_splitter import split_sentences_with_offsets

    text = "First sentence is right here. 'Quoted second sentence,' he said loudly.\n\nNew paragraph starts here now."
    for sentence, start in split_sentences_with_offsets(text):
        assert text[start : start + len(sentence)] == sentence


def test_min_words_is_configurable():
    assert split_sentences("Yes! It works well here.", min_words=1) == ["Yes!", "It works well here."]


def test_empty_text_gives_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("\n\n") == []


def test_order_is_preserved():
    text = " ".join(f"This is sentence number {i} in the text." for i in range(20))
    sentences = split_sentences(text)
    assert [s.split()[4] for s in sentences] == [str(i) for i in range(20)]


@pytest.mark.parametrize("segmenter", ["parser", "senter", "rule"])
def test_all_segmenters_load(segmenter):
    nlp = get_nlp(get_settings().spacy_model, segmenter)
    assert len(list(nlp("One sentence here. Another sentence there.").sents)) == 2


def test_missing_model_falls_back_to_rules():
    nlp = get_nlp("model_that_does_not_exist", "parser")
    assert "sentencizer" in nlp.pipe_names
