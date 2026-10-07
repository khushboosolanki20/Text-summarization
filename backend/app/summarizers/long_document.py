"""
Hierarchical (map-reduce) summarization for documents longer than the
model's context window.

    long document
        │  sentence segmentation (already done in preprocessing)
        ▼
    chunks of whole sentences, each ≤ chunk limit         (chunk_sentences)
        │  MAP: summarize every chunk independently
        ▼
    intermediate summaries, in document order
        │  COMBINE: join them
        ▼
    combined text ── fits the window? ──yes──► REDUCE: final summarization pass ──► summary
        │ no
        └──► treat the combined text as a new document and repeat
             (at most ``max_reduction_levels`` rounds)

Length planning
---------------
Let T be the requested summary length (ratio x document words).

* If one generation pass can produce T words (T x 1.3 tokens <= 400), the
  chunk summaries together aim for about 2T words (more material than needed,
  so the final pass can *choose* and *fuse*), capped so the combined text
  fits the context window. The final pass then writes a T-word summary.
  Strategy: ``fused``.
* If T is longer than one pass can produce (e.g. 12 % of a 10,000-word
  report = 1,200 words), a final pass would have to compress *below* the
  requested length. Instead the chunk summaries together aim for T words and
  are returned in document order, i.e. a section-by-section summary.
  Strategy: ``concatenated``.

Each chunk's share of the budget is proportional to its share of the
document's words, so long sections get longer summaries.

Nothing is truncated: every sentence of the document is read by the model
exactly once in the first round.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from app.config import get_settings
from app.preprocessing.chunker import Chunk, chunk_sentences
from app.preprocessing.cleaner import count_words
from app.preprocessing.sentence_splitter import split_sentences
from app.summarizers.base import SummaryLength, length_ratio
from app.summarizers.generation import TextGenerator, generate_summary, max_single_pass_words

# Called as on_progress(stage, done, total), e.g. ("level 1: summarizing chunks", 3, 12).
ProgressCallback = Callable[[str, int, int], None]


@dataclass
class ReductionLevel:
    level: int
    input_words: int
    target_words: float  # total words the chunk summaries aim for
    chunks: list[Chunk]
    summaries: list[str]

    @property
    def output_words(self) -> int:
        return sum(count_words(s) for s in self.summaries)

    def describe(self) -> dict:
        return {
            "level": self.level,
            "input_words": self.input_words,
            "target_words": round(self.target_words),
            "output_words": self.output_words,
            "num_chunks": len(self.chunks),
            "chunk_token_counts": [c.token_count for c in self.chunks],
            "contains_split_sentence": any(c.contains_split_sentence for c in self.chunks),
        }


@dataclass
class HierarchicalResult:
    summary: str
    strategy: str  # "fused" | "concatenated" | "max_levels_reached"
    target_words: float
    levels: list[ReductionLevel] = field(default_factory=list)
    final_pass_trimmed: bool = False

    @property
    def intermediate_summaries(self) -> list[str]:
        """The first-round chunk summaries (one per chunk of the original document)."""
        return self.levels[0].summaries if self.levels else []


def summarize_chunks(
    generator: TextGenerator,
    chunks: list[Chunk],
    total_target_words: float,
    on_progress: ProgressCallback | None = None,
    stage: str = "summarizing chunks",
) -> list[str]:
    """MAP step: summarize each chunk, with a length proportional to its size."""
    total_words = sum(c.word_count for c in chunks) or 1
    summaries = []
    for done, chunk in enumerate(chunks, start=1):
        share = total_target_words * chunk.word_count / total_words
        summary, _, _ = generate_summary(generator, chunk.text, share)
        summaries.append(summary)
        if on_progress:
            on_progress(stage, done, len(chunks))
    return summaries


def hierarchical_summarize(
    generator: TextGenerator,
    sentences: list[str],
    length: SummaryLength | str,
    on_progress: ProgressCallback | None = None,
) -> HierarchicalResult:
    settings = get_settings()
    original_words = sum(count_words(s) for s in sentences)
    target = original_words * length_ratio(length)  # T
    fuse = target <= max_single_pass_words()

    chunk_limit = min(settings.chunk_max_tokens, generator.content_token_limit)
    # Words the final pass can read: the window minus 10 % headroom.
    final_input_words = generator.content_token_limit / settings.tokens_per_word * 0.9

    def count(text: str) -> int:
        return generator.count_tokens(text, special_tokens=False)

    levels: list[ReductionLevel] = []
    current = sentences
    for level in range(1, settings.max_reduction_levels + 1):
        input_words = sum(count_words(s) for s in current)
        level_target = min(2 * target, final_input_words) if fuse else target
        # Every round must at least halve the text, which guarantees progress.
        level_target = min(level_target, input_words * 0.5)

        chunks = chunk_sentences(current, chunk_limit, count, settings.chunk_overlap_sentences)
        summaries = summarize_chunks(generator, chunks, level_target, on_progress, f"level {level}: summarizing chunks")
        levels.append(ReductionLevel(level, input_words, level_target, chunks, summaries))
        combined = " ".join(summaries)

        if fuse and generator.fits(combined):
            # REDUCE: one final pass fuses the partial summaries.
            if on_progress:
                on_progress("final summarization pass", 0, 1)
            summary, trimmed, _ = generate_summary(generator, combined, target)
            if on_progress:
                on_progress("final summarization pass", 1, 1)
            return HierarchicalResult(summary, "fused", target, levels, trimmed)

        if not fuse and count_words(combined) <= target * 1.25:
            return HierarchicalResult(combined, "concatenated", target, levels)

        # Still too long: the combined summaries become the next round's input.
        current = split_sentences(combined, min_words=1)

    return HierarchicalResult(" ".join(levels[-1].summaries), "max_levels_reached", target, levels)
