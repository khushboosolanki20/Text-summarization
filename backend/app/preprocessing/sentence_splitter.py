"""
Sentence segmentation with spaCy.

Sentences are the unit that every IntelliSum method works with: extractive
methods score and select sentences, and the long-document chunker groups
sentences so that no sentence is ever cut in half.

Splitting on "." alone fails on text like "Dr. Smith met Prof. Jones of
M.I.T. at 3 p.m.", so we use spaCy's statistical models. Three back-ends are
available (see ``Settings.sentence_segmenter``):

* ``parser``: boundaries come from the dependency parse (most accurate).
* ``senter``: a small dedicated sentence classifier (faster, slightly less
  accurate around abbreviations).
* ``rule``:   punctuation rules only; used automatically if the spaCy model
  is not installed, so the application still works.

Paragraphs (separated by blank lines after cleaning) are processed
independently, so a heading can never be merged into the next sentence.
"""

import logging
import re
from functools import lru_cache

import spacy
from spacy.language import Language

from app.config import get_settings
from app.preprocessing.cleaner import count_words

logger = logging.getLogger(__name__)

# Components we never need for segmentation; excluding them makes loading and
# processing considerably faster.
_ALWAYS_EXCLUDED = ["ner", "lemmatizer"]


def _rule_based_pipeline(max_length: int) -> Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.max_length = max_length
    return nlp


@lru_cache(maxsize=4)
def get_nlp(model: str, segmenter: str) -> Language:
    """Load (once per process) and cache the spaCy pipeline for segmentation."""
    max_length = get_settings().max_input_chars + 1000
    if segmenter == "rule":
        return _rule_based_pipeline(max_length)

    try:
        if segmenter == "senter":
            nlp = spacy.load(model, exclude=_ALWAYS_EXCLUDED + ["parser", "tagger", "attribute_ruler"])
            nlp.enable_pipe("senter")
        else:  # "parser"
            nlp = spacy.load(model, exclude=_ALWAYS_EXCLUDED)
    except OSError:
        logger.warning(
            "spaCy model '%s' is not installed; falling back to rule-based sentence splitting. "
            "Install it with: python -m spacy download %s",
            model,
            model,
        )
        return _rule_based_pipeline(max_length)

    nlp.max_length = max_length
    return nlp


def split_sentences(text: str, min_words: int | None = None) -> list[str]:
    """
    Split cleaned text into sentences, in document order.

    Fragments with fewer than ``min_words`` words or with no letters at all
    (headings, figure labels, stray numbers) are dropped: they carry no
    summarizable content and would distort sentence scoring.
    """
    return [sentence for sentence, _ in split_sentences_with_offsets(text, min_words)]


_PARAGRAPH = re.compile(r"[^\n]+")
# spaCy sometimes attaches the opening quote of the next sentence to the end of
# the previous one: ["...Donoghue. '", "Garry, if you..."]. Detect a lone
# opening quote at the end of a sentence (after whitespace).
_DANGLING_OPEN_QUOTE = re.compile(r"\s+(['\"‘“])$")


def _fix_dangling_quotes(spans: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Move a dangling opening quote from the end of one sentence to the start of the next."""
    fixed = [list(span) for span in spans]
    for i in range(len(fixed) - 1):
        match = _DANGLING_OPEN_QUOTE.search(fixed[i][0])
        if match:
            quote = match.group(1)
            fixed[i][0] = fixed[i][0][: match.start()]
            fixed[i + 1][0] = quote + fixed[i + 1][0]
            fixed[i + 1][1] -= len(quote)  # the next sentence now starts at the quote
    return [(text, start) for text, start in fixed]


def split_sentences_with_offsets(text: str, min_words: int | None = None) -> list[tuple[str, int]]:
    """
    Like ``split_sentences`` but also returns each sentence's start position
    (character offset) in ``text``, used to map sentences back to the page
    they came from.
    """
    settings = get_settings()
    if min_words is None:
        min_words = settings.min_sentence_words

    # Paragraphs are runs of text between newlines; remember where each starts.
    paragraphs = [(m.group(), m.start()) for m in _PARAGRAPH.finditer(text) if m.group().strip()]
    if not paragraphs:
        return []

    nlp = get_nlp(settings.spacy_model, settings.sentence_segmenter)
    sentences: list[tuple[str, int]] = []
    # nlp.pipe batches the paragraphs, which is much faster than calling nlp()
    # on each one.
    docs = nlp.pipe((p for p, _ in paragraphs), batch_size=64)
    for doc, (_, paragraph_start) in zip(docs, paragraphs):
        spans = []
        for span in doc.sents:
            leading = len(span.text) - len(span.text.lstrip())
            spans.append((span.text.strip(), paragraph_start + span.start_char + leading))
        for sentence, start in _fix_dangling_quotes(spans):
            if count_words(sentence) >= min_words and any(ch.isalpha() for ch in sentence):
                sentences.append((sentence, start))
    return sentences
