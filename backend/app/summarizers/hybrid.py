"""
Hybrid summarization: TextRank (extractive) -> BART (abstractive).

Motivation
----------
The two families have complementary strengths and weaknesses:

* TextRank is fast, reads the *whole* document regardless of length and
  reliably finds the central sentences, but its output is a list of copied
  sentences that can be choppy and redundant.
* BART writes fluent, compressed summaries, but it reads at most 1,024
  tokens per pass. Long documents need many slow passes (one per chunk), and
  every chunk is summarized without knowing what matters in the document
  as a whole.

Hybrid uses TextRank as a **content selector** and BART as a **rewriter**:

    full document
        │  TextRank: score every sentence by centrality in the similarity graph
        ▼
    key sentences   (best-first, near-duplicates skipped, until ~3x the
        │            requested summary length; capped to one BART window when
        │            the summary itself fits one pass)
        │  restore original order
        ▼
    selected text   (much shorter than the document)
        │  BART: one pass if it fits, hierarchical chunking otherwise
        ▼
    abstractive summary, with length relative to the ORIGINAL document

Effects: for a typical report the whole document is "read" by TextRank, but
BART only processes the most central ~3x-summary-length of it, often in a
single pass instead of many chunks. Less input means less time and fewer
irrelevant details for BART to copy. The cost: information TextRank
considers peripheral can never reach the summary, and TextRank errors
propagate.
"""

import time

from app.config import get_settings
from app.preprocessing.cleaner import count_words
from app.summarizers.bart import BARTSummarizer
from app.summarizers.base import BaseSummarizer, SummaryLength, SummaryResult, length_ratio
from app.summarizers.generation import max_single_pass_words
from app.summarizers.long_document import ProgressCallback
from app.summarizers.textrank import TextRankSummarizer


class HybridSummarizer(BaseSummarizer):
    name = "hybrid"

    def __init__(
        self,
        textrank: TextRankSummarizer | None = None,
        abstractive: BARTSummarizer | None = None,
        expansion: float | None = None,
    ):
        self.textrank = textrank or TextRankSummarizer()
        self.abstractive = abstractive or BARTSummarizer()
        self.expansion = get_settings().hybrid_expansion if expansion is None else expansion
        if self.expansion < 1:
            raise ValueError("expansion must be >= 1 (the selection must be at least as long as the summary)")

    def select_sentences(
        self,
        sentences: list[str],
        word_budget: float,
        token_cap: int | None,
    ) -> tuple[list[int], list[float], dict]:
        """
        Extractive stage. Walk down the TextRank ranking and keep sentences
        until ``word_budget`` words are selected. Near-duplicates of an
        already-selected sentence are skipped (they add length, not content).
        With a ``token_cap``, a sentence that would overflow it is skipped and
        shorter, lower-ranked sentences are still considered, so the selection
        fills one BART window as fully as possible.
        Returns (selected indices in document order, TextRank scores, metadata).
        """
        scores, similarity, textrank_meta = self.textrank.score_sentences(sentences)
        ranking = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
        threshold = get_settings().redundancy_threshold

        def tokens(sentence: str) -> int:
            return self.abstractive.model.count_tokens(sentence, special_tokens=False)

        selected: list[int] = []
        words = token_total = redundant = 0
        for i in ranking:
            if words >= word_budget:
                break
            if similarity is not None and selected and similarity(i)[selected].max() > threshold:
                redundant += 1
                continue
            if token_cap is not None:
                t = tokens(sentences[i])
                if token_total + t > token_cap:
                    continue
                token_total += t
            selected.append(i)
            words += count_words(sentences[i])

        if not selected:  # e.g. the single top sentence is longer than the cap
            selected = [ranking[0]]
        return sorted(selected), scores, {**textrank_meta, "redundant_skipped": redundant}

    def summarize(
        self,
        sentences: list[str],
        length: SummaryLength | str = SummaryLength.MEDIUM,
        on_progress: ProgressCallback | None = None,
    ) -> SummaryResult:
        if not sentences:
            return SummaryResult("", self.name, 0, 0, [], [])

        original_words = sum(count_words(s) for s in sentences)
        target = original_words * length_ratio(length)  # T, relative to the ORIGINAL document
        word_budget = self.expansion * target
        # If the final summary can be written in one pass, also make the
        # selection fit one pass: then BART needs no chunking at all.
        single_pass = target <= max_single_pass_words()
        token_cap = (
            min(get_settings().chunk_max_tokens, self.abstractive.model.content_token_limit) if single_pass else None
        )

        # --- Stage 1: TextRank content selection ---------------------------
        if on_progress:
            on_progress("selecting key sentences (TextRank)", 0, 1)
        start = time.perf_counter()
        selected, scores, textrank_meta = self.select_sentences(sentences, word_budget, token_cap)
        extract_seconds = round(time.perf_counter() - start, 3)
        if on_progress:
            on_progress("selecting key sentences (TextRank)", 1, 1)
        selected_sentences = [sentences[i] for i in selected]  # original document order
        selected_words = sum(count_words(s) for s in selected_sentences)

        # --- Stage 2: BART rewriting ------------------------------------------
        abstract = self.abstractive.summarize_to_target(selected_sentences, target, on_progress)

        return SummaryResult(
            summary=abstract.summary,
            method=self.name,
            original_word_count=original_words,
            summary_word_count=abstract.summary_word_count,
            # For Hybrid these are the sentences that were passed to BART.
            selected_indices=selected,
            sentence_scores=[round(float(s), 6) for s in scores],
            metadata={
                "num_sentences": len(sentences),
                "num_selected": len(selected),
                "extractive_stage": {
                    "method": "textrank",
                    "selected_sentences": len(selected),
                    "selected_words": selected_words,
                    "word_budget": round(word_budget),
                    "token_cap": token_cap,
                    # How much input BART was spared, as a % of the document's words.
                    "input_reduction": round(100 * (1 - selected_words / original_words), 2),
                    "seconds": extract_seconds,
                    **textrank_meta,
                },
                "abstractive_stage": abstract.metadata,
            },
        )
