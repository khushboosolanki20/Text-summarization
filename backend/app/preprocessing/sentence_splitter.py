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
    settings = get_settings()
    if min_words is None:
        min_words = settings.min_sentence_words

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    nlp = get_nlp(settings.spacy_model, settings.sentence_segmenter)
    sentences: list[str] = []
    # nlp.pipe batches the paragraphs, which is much faster than calling nlp()
    # on each one.
    for doc in nlp.pipe(paragraphs, batch_size=64):
        for span in doc.sents:
            sentence = span.text.strip()
            if count_words(sentence) >= min_words and any(ch.isalpha() for ch in sentence):
                sentences.append(sentence)
    return sentences
