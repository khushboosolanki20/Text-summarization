"""
Word-level tokenization for the vector-space methods (TF-IDF, TextRank).

Turning a sentence into "terms" involves three steps:

1. **Tokenize + lowercase**: keep alphabetic words (numbers and punctuation
   carry little topical meaning on their own).
2. **Stop-word removal**: drop very common function words ("the", "is",
   "of") that appear in every sentence and say nothing about its topic.
   We use scikit-learn's built-in English list (318 words), so no data
   download is needed.
3. **Stemming** (NLTK Porter stemmer): reduce inflected forms to a common
   stem, so "summarize", "summarizes" and "summarization" all count as the
   same term "summar". Without this, related words would be treated as
   unrelated dimensions and similarity between sentences would be
   under-estimated.
"""

import re
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"[a-z][a-z'-]*[a-z]|[a-z]")
_stemmer = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """Sentence -> list of stemmed content-word terms."""
    return [_stemmer.stem(w) for w in _WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]


class StemDisplayMap:
    """
    Remembers which original word each stem came from, so that stems like
    "summar" can be shown to users as a readable word ("summarization").
    """

    def __init__(self, sentences: list[str]):
        self._words: dict[str, Counter[str]] = defaultdict(Counter)
        for sentence in sentences:
            for word in _WORD.findall(sentence.lower()):
                if word not in ENGLISH_STOP_WORDS:
                    self._words[_stemmer.stem(word)][word] += 1

    def display(self, stem: str) -> str:
        """The most frequent original word for this stem."""
        forms = self._words.get(stem)
        return forms.most_common(1)[0][0] if forms else stem
