"""
Tests for the hierarchical (map-reduce) orchestration.

The real BART model takes seconds per chunk on CPU, so these tests use a tiny
deterministic stand-in generator that "summarizes" by keeping the first words
of its input. This checks the orchestration logic (chunking, length planning,
rounds, final pass, progress, no truncation); BART itself is exercised by the
slow tests in test_bart.py.
"""

import math

from app.summarizers.long_document import hierarchical_summarize


class KeepFirstWords:
    """Stand-in model: 1.3 tokens/word, 1,024-token window, records every call."""

    max_input_tokens = 1024

    def __init__(self, compress: bool = True):
        self.compress = compress
        self.calls: list[tuple[str, int, int]] = []

    def count_tokens(self, text: str, special_tokens: bool = True) -> int:
        return math.ceil(len(text.split()) * 1.3) + (2 if special_tokens else 0)

    @property
    def content_token_limit(self) -> int:
        return self.max_input_tokens - 2

    def fits(self, text: str) -> bool:
        return self.count_tokens(text) <= self.max_input_tokens

    def generate(self, texts, min_tokens, max_tokens):
        assert all(self.fits(t) for t in texts), "input exceeded the context window"
        self.calls.append((texts[0], min_tokens, max_tokens))
        if not self.compress:
            return list(texts)
        keep = max(1, int((min_tokens + max_tokens) / 2 / 1.3))  # middle of the allowed range
        return [" ".join(t.split()[:keep]).rstrip(".") + "." for t in texts]


def document(n_sentences: int, words_per_sentence: int = 15) -> list[str]:
    return [f"Sentence {i} " + " ".join(["content"] * (words_per_sentence - 3)) + " here." for i in range(n_sentences)]


def test_medium_document_is_fused_in_a_final_pass():
    gen = KeepFirstWords()
    doc = document(100)  # 1,500 words, ~1,950 tokens: needs chunking
    result = hierarchical_summarize(gen, doc, "short")  # T = 180 words: one pass can produce it
    assert result.strategy == "fused"
    assert len(result.levels) == 1
    assert len(result.levels[0].chunks) >= 3
    assert len(gen.calls) == len(result.levels[0].chunks) + 1  # every chunk + final pass
    final_input = gen.calls[-1][0]
    assert final_input == " ".join(result.intermediate_summaries)


def test_every_sentence_is_read_exactly_once():
    gen = KeepFirstWords()
    doc = document(100)
    result = hierarchical_summarize(gen, doc, "short")
    chunk_sentences = [s for c in result.levels[0].chunks for s in c.sentences]
    assert chunk_sentences == doc  # nothing truncated, duplicated or reordered


def test_chunks_fit_configured_limit():
    gen = KeepFirstWords()
    result = hierarchical_summarize(gen, document(200), "short")
    assert all(c.token_count <= 900 for level in result.levels for c in level.chunks)


def test_long_target_returns_section_summaries_in_order():
    gen = KeepFirstWords()
    doc = document(400)  # 6,000 words; short = 720 words > one pass (~307 words)
    result = hierarchical_summarize(gen, doc, "short")
    assert result.strategy == "concatenated"
    words = len(result.summary.split())
    assert 0.5 * result.target_words <= words <= 1.25 * result.target_words
    # Section summaries keep document order: chunk i's summary starts with its first sentence.
    first_numbers = [int(s.split()[1]) for s in result.intermediate_summaries]
    assert first_numbers == sorted(first_numbers)


def test_chunk_budgets_are_proportional_to_chunk_size():
    gen = KeepFirstWords()
    doc = document(60, 15) + document(20, 60)  # first part: short sentences, second: long ones
    result = hierarchical_summarize(gen, doc, "short")
    level = result.levels[0]
    ratios = [len(s.split()) / c.word_count for s, c in zip(level.summaries, level.chunks)]
    assert max(ratios) - min(ratios) < 0.1


def test_rounds_are_bounded_even_if_the_model_does_not_compress():
    # A model that returns its input unchanged never makes the text fit; the
    # loop must still stop after max_reduction_levels rounds.
    stubborn = KeepFirstWords(compress=False)
    result = hierarchical_summarize(stubborn, document(100), "short")
    assert result.strategy == "max_levels_reached"
    assert len(result.levels) == 3


def test_progress_is_reported():
    gen = KeepFirstWords()
    events = []
    hierarchical_summarize(gen, document(100), "short", on_progress=lambda *e: events.append(e))
    chunk_events = [e for e in events if e[0].startswith("level 1")]
    assert [done for _, done, _ in chunk_events] == list(range(1, len(chunk_events) + 1))
    assert events[-1] == ("final summarization pass", 1, 1)


def test_longer_settings_produce_longer_summaries():
    lengths = [
        len(hierarchical_summarize(KeepFirstWords(), document(100), size).summary.split())
        for size in ("short", "medium", "long")
    ]
    assert lengths[0] < lengths[1] < lengths[2]
