"""
Sentence-aware chunking for long documents.

Why chunking is needed
----------------------
A Transformer's attention compares every token with every other token, so
its cost grows with the square of the input length. Models are therefore
trained with a fixed maximum input: 1,024 tokens (~750 words) for BART.
A longer document cannot be read in one pass, and simply truncating it would
silently throw away everything after the first ~750 words.

Instead the document is split into **chunks** that each fit the model, and
the chunks are summarized separately (see ``app.summarizers.long_document``).

How chunks are formed
---------------------
* Chunks are built from **whole sentences**, so no sentence is cut in half
  and every chunk is readable on its own.
* Chunks are **balanced**: with T total tokens and a limit L we need
  n = ceil(T / L) chunks, so we aim for T / n tokens per chunk instead of
  filling each chunk to the limit. Greedy filling would often leave a tiny
  last chunk ("...and 2 sentences") whose summary is mostly noise.
* A chunk is never empty. A final chunk that is still very small is merged
  into the previous one when that fits.
* Optional **overlap**: the last k sentences of a chunk can be repeated at
  the start of the next one to preserve context across the boundary (off by
  default because it duplicates content in the combined summary).
* The only exception to "never cut a sentence": a single sentence longer
  than a whole chunk (e.g. a table flattened into one line). It is split at
  word boundaries and the chunk is flagged ``contains_split_sentence``.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from app.preprocessing.cleaner import count_words

TokenCounter = Callable[[str], int]

# A last chunk smaller than this fraction of the target size is merged into
# the previous chunk if the result still fits the limit.
_MIN_LAST_CHUNK_FRACTION = 0.3


@dataclass
class Chunk:
    index: int
    sentences: list[str]
    start_sentence: int  # position of the first sentence in the input list
    token_count: int
    contains_split_sentence: bool = False

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def word_count(self) -> int:
        return count_words(self.text)


def approximate_token_count(text: str) -> int:
    """Fallback counter (~1.3 BPE tokens per word) when no tokenizer is supplied."""
    return math.ceil(count_words(text) * 1.3)


def _split_long_sentence(sentence: str, max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    """Split one over-long sentence into word-boundary pieces that each fit."""
    words = sentence.split()
    pieces: list[str] = []
    start = 0
    while start < len(words):
        # Estimate how many words fit, then shrink until the piece really fits.
        ratio = max_tokens / max(1, count_tokens(" ".join(words[start:])))
        size = max(1, min(len(words) - start, int((len(words) - start) * ratio * 0.95)))
        while size > 1 and count_tokens(" ".join(words[start : start + size])) > max_tokens:
            size = max(1, int(size * 0.9))
        pieces.append(" ".join(words[start : start + size]))
        start += size
    return pieces


def chunk_sentences(
    sentences: list[str],
    max_tokens: int,
    count_tokens: TokenCounter | None = None,
    overlap_sentences: int = 0,
) -> list[Chunk]:
    """
    Group consecutive sentences into balanced chunks of at most ``max_tokens``
    tokens (as measured by ``count_tokens`` on the joined chunk text).
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    count_tokens = count_tokens or approximate_token_count
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return []

    # 1. Measure every sentence; split the (rare) ones that can't fit alone.
    units: list[tuple[str, int, int, bool]] = []  # (text, tokens, original index, was_split)
    for i, sentence in enumerate(sentences):
        tokens = count_tokens(sentence)
        if tokens <= max_tokens:
            units.append((sentence, tokens, i, False))
        else:
            for piece in _split_long_sentence(sentence, max_tokens, count_tokens):
                units.append((piece, count_tokens(piece), i, True))

    # 2. Balanced target size.
    total = sum(tokens for _, tokens, _, _ in units)
    target = total / math.ceil(total / max_tokens)

    # 3. Greedy grouping: close a chunk when the next unit would exceed the
    #    limit or the chunk has reached the balanced target. With overlap, a
    #    new chunk starts with the previous chunk's last sentences already
    #    counted, so the repeated context is planned for rather than added
    #    afterwards (when full chunks would have no room left for it).
    groups: list[list[int]] = [[]]
    size = 0
    for u, (_, tokens, _, _) in enumerate(units):
        if groups[-1] and (size + tokens > max_tokens or size >= target):
            carry = groups[-1][-overlap_sentences:] if overlap_sentences > 0 else []
            carry_size = sum(units[c][1] for c in carry)
            if carry_size + tokens > max_tokens or carry_size >= target:
                carry, carry_size = [], 0  # no room: the chunk must make progress
            groups.append(list(carry))
            size = carry_size
        groups[-1].append(u)
        size += tokens

    # 4. Merge a tiny final chunk into its predecessor if it fits (skipping
    #    overlap sentences the predecessor already contains).
    if len(groups) > 1:
        new_units = [u for u in groups[-1] if u not in groups[-2]]
        last = sum(units[u][1] for u in new_units)
        prev = sum(units[u][1] for u in groups[-2])
        if last < target * _MIN_LAST_CHUNK_FRACTION and last + prev <= max_tokens:
            groups.pop()
            groups[-1].extend(new_units)

    # 5. Build chunks and verify the real token count of the joined text
    #    (joining can change tokenization slightly); split again if needed.
    chunks: list[Chunk] = []
    pending = list(groups)
    while pending:
        group = pending.pop(0)
        texts = [units[u][0] for u in group]
        tokens = count_tokens(" ".join(texts))
        if tokens > max_tokens and len(group) > 1:
            middle = len(group) // 2
            pending[:0] = [group[:middle], group[middle:]]
            continue
        chunks.append(
            Chunk(
                index=len(chunks),
                sentences=texts,
                start_sentence=units[group[0]][2],
                token_count=tokens,
                contains_split_sentence=any(units[u][3] for u in group),
            )
        )
    return chunks
