import pytest
from transformers import AutoTokenizer

from app.preprocessing.chunker import approximate_token_count, chunk_sentences


def words(text: str) -> int:
    return len(text.split())


def make_sentences(n: int, length: int = 10) -> list[str]:
    return [" ".join([f"w{i}"] * (length - 1)) + " end." for i in range(n)]


def flatten(chunks):
    return [s for c in chunks for s in c.sentences]


def test_empty_input():
    assert chunk_sentences([], 100, words) == []
    assert chunk_sentences(["   ", ""], 100, words) == []


def test_small_document_is_a_single_chunk():
    sentences = make_sentences(3)
    chunks = chunk_sentences(sentences, 100, words)
    assert len(chunks) == 1
    assert chunks[0].sentences == sentences
    assert chunks[0].start_sentence == 0


def test_every_sentence_kept_once_in_order():
    sentences = make_sentences(57)
    chunks = chunk_sentences(sentences, 100, words)
    assert flatten(chunks) == sentences  # nothing lost, duplicated or reordered


def test_chunks_respect_limit_and_are_not_empty():
    chunks = chunk_sentences(make_sentences(57), 100, words)
    assert all(0 < c.token_count <= 100 for c in chunks)
    assert all(c.sentences for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_start_sentence_indices():
    chunks = chunk_sentences(make_sentences(30), 100, words)
    expected = 0
    for chunk in chunks:
        assert chunk.start_sentence == expected
        expected += len(chunk.sentences)


def test_chunks_are_balanced():
    # 110 sentences x 10 words = 1,100 words with a 500 limit -> 3 chunks of
    # ~367 words, not 500 + 500 + 100.
    chunks = chunk_sentences(make_sentences(110), 500, words)
    sizes = [c.token_count for c in chunks]
    assert len(chunks) == 3
    assert max(sizes) - min(sizes) <= 20


def test_tiny_last_chunk_is_merged():
    # Sizes 60, 45, 50, 4 with limit 100: grouping gives [60] [45, 50] [4]; the
    # 4-word leftover is merged into the previous chunk (95 + 4 <= 100).
    sentences = [" ".join(["x"] * n) for n in (60, 45, 50, 4)]
    chunks = chunk_sentences(sentences, 100, words)
    assert [c.token_count for c in chunks] == [60, 99]
    assert flatten(chunks) == sentences


def test_sentence_longer_than_limit_is_split_and_flagged():
    giant = " ".join(f"word{i}" for i in range(250)) + "."
    sentences = ["A normal opening sentence here.", giant, "A normal closing sentence here."]
    chunks = chunk_sentences(sentences, 100, words)
    assert all(c.token_count <= 100 for c in chunks)
    assert any(c.contains_split_sentence for c in chunks)
    assert " ".join(flatten(chunks)).split() == " ".join(sentences).split()  # no word lost


def test_overlap_repeats_sentences_but_respects_limit():
    sentences = make_sentences(40)
    chunks = chunk_sentences(sentences, 100, words, overlap_sentences=1)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.sentences[0] == previous.sentences[-1]
    assert all(c.token_count <= 100 for c in chunks)


def test_invalid_limit():
    with pytest.raises(ValueError):
        chunk_sentences(["Some text."], 0)


def test_default_counter_is_approximate_bpe():
    assert approximate_token_count("one two three four five six seven eight nine ten") == 13
    chunks = chunk_sentences(make_sentences(50), 130)  # no counter passed
    assert all(c.token_count <= 130 for c in chunks)


def test_real_bart_tokenizer_counts():
    # Only the tokenizer is loaded (fast); the joined text of every chunk must
    # really fit, as measured by the same tokenizer BART uses.
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    sentences = [
        f"In {2000 + i}, researchers at the Indian Institute of Technology reported a {i}.5% improvement "
        f"in summarization quality using transformer-based models." for i in range(120)
    ]
    chunks = chunk_sentences(sentences, 900, count)
    assert len(chunks) >= 2
    assert all(count(c.text) <= 900 for c in chunks)
    assert flatten(chunks) == sentences
