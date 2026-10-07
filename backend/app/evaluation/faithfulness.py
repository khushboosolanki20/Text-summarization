"""
Experimental faithfulness check: "Potentially unsupported content".

What hallucination means here
-----------------------------
An abstractive model *generates* text, so it can produce statements the
source never made: a wrong number ("rose by 15 percent" when the source says
50), a name or place that does not occur, or a plausible-sounding claim with
no basis. Extractive summaries cannot do this (they copy source sentences),
so the check only applies to BART and Hybrid output.

What this check does
--------------------
For every sentence of the generated summary:

1. **Content-word coverage**: the share of the sentence's content words
   (stop words removed, Porter-stemmed) that occur anywhere in the source.
   A sentence assembled from source material has high coverage; a sentence
   introducing new content does not.
2. **Unsupported details**: numbers, and named entities (people,
   organisations, places, ... found by spaCy's NER), that never appear in the
   source. These are the most common and most damaging hallucinations.
3. **Closest source passage**: the source sentence (or pair of adjacent
   sentences, since abstractive sentences often fuse two) with the highest
   TF-IDF cosine similarity, shown so a reader can verify the claim.

A sentence is flagged as *potentially unsupported* if its coverage is below
the threshold (default 60 %) or it contains an unsupported number or entity.

What it is NOT
--------------
This is a lexical heuristic, **not** a hallucination detector:

* a faithful paraphrase using different words can be flagged (false positive);
* a sentence that recombines source words into a false claim ("China accounted
  for half of wind capacity" when it was solar) is not flagged (false negative);
* sentence-embedding or entailment (NLI) models would capture meaning better
  but add another model download; they are listed as future work.

The result is therefore labelled "experimental" and can be disabled with
``INTELLISUM_FAITHFULNESS_CHECK=false``.
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from app.config import get_settings
from app.preprocessing.sentence_splitter import split_sentences
from app.preprocessing.tokenizer import tokenize
from app.preprocessing.vectorizer import make_tfidf_vectorizer

logger = logging.getLogger(__name__)

LABEL = "Potentially unsupported content (experimental)"
NOTE = (
    "Heuristic check: sentences whose words, numbers or names are not found in the source are flagged for review. "
    "It can miss errors and flag faithful paraphrases; it is not a guarantee of correctness."
)

# Entity types whose absence from the source is a meaningful warning sign.
_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"}
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class SentenceSupport:
    sentence: str
    coverage: float  # share of content words found in the source
    similarity: float  # cosine similarity to the closest source passage
    closest_source: list[int]  # index/indices of the closest source sentence(s)
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_entities: list[str] = field(default_factory=list)
    flagged: bool = False
    reasons: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _ner_pipeline():
    """spaCy pipeline with named-entity recognition (loaded once); None if unavailable."""
    import spacy

    try:
        return spacy.load(get_settings().spacy_model, exclude=["parser", "lemmatizer"])
    except OSError:
        logger.warning("spaCy model not installed: faithfulness check runs without entity checks.")
        return None


def _normalise_number(token: str) -> str:
    return token.replace(",", "")


def _numbers(text: str) -> set[str]:
    return {_normalise_number(n) for n in _NUMBER.findall(text)}


def _entities(text: str) -> list[str]:
    nlp = _ner_pipeline()
    if nlp is None:
        return []
    return [ent.text for ent in nlp(text).ents if ent.label_ in _ENTITY_LABELS]


def check_faithfulness(summary: str, source_sentences: list[str], min_coverage: float | None = None) -> dict:
    """Score each summary sentence for support in the source (see module docstring)."""
    min_coverage = get_settings().faithfulness_min_coverage if min_coverage is None else min_coverage
    summary_sentences = split_sentences(summary, min_words=1)
    if not summary_sentences or not source_sentences:
        return {"applicable": True, "label": LABEL, "note": NOTE, "flagged_count": 0, "sentences": []}

    source_text = " ".join(source_sentences)
    source_lower = source_text.lower()
    source_terms = set(tokenize(source_text))
    source_numbers = _numbers(source_text)

    # Candidate source passages: each sentence and each pair of adjacent sentences.
    passages = [[i] for i in range(len(source_sentences))]
    passages += [[i, i + 1] for i in range(len(source_sentences) - 1)]
    vectorizer = make_tfidf_vectorizer()
    try:
        passage_matrix = vectorizer.fit_transform([" ".join(source_sentences[i] for i in p) for p in passages])
        similarities = (vectorizer.transform(summary_sentences) @ passage_matrix.T).toarray()
    except ValueError:  # no vocabulary (e.g. only stop words)
        similarities = None

    results: list[SentenceSupport] = []
    for row, sentence in enumerate(summary_sentences):
        terms = tokenize(sentence)
        coverage = sum(t in source_terms for t in terms) / len(terms) if terms else 1.0
        similarity, closest = 0.0, []
        if similarities is not None and similarities[row].max() > 0:
            best = int(similarities[row].argmax())
            similarity, closest = float(similarities[row][best]), passages[best]

        missing_numbers = sorted(_numbers(sentence) - source_numbers)
        missing_entities = [e for e in dict.fromkeys(_entities(sentence)) if e.lower() not in source_lower]

        reasons = []
        if coverage < min_coverage:
            reasons.append(f"only {coverage:.0%} of its content words appear in the source")
        if missing_numbers:
            reasons.append(f"number(s) not found in the source: {', '.join(missing_numbers)}")
        if missing_entities:
            reasons.append(f"name(s) not found in the source: {', '.join(missing_entities)}")

        results.append(
            SentenceSupport(
                sentence=sentence,
                coverage=round(coverage, 3),
                similarity=round(similarity, 3),
                closest_source=closest,
                unsupported_numbers=missing_numbers,
                unsupported_entities=missing_entities,
                flagged=bool(reasons),
                reasons=reasons,
            )
        )

    return {
        "applicable": True,
        "label": LABEL,
        "note": NOTE,
        "flagged_count": sum(r.flagged for r in results),
        "sentences": [asdict(r) for r in results],
    }


def not_applicable(reason: str) -> dict:
    return {"applicable": False, "label": LABEL, "reason": reason}
