"""
Chunking strategies exposed as LangChain ``TextSplitter`` objects.

LangChain's ``TextSplitter`` is the standard interface for "cut text into
pieces of at most N units": ``split_text(str) -> list[str]`` and
``split_documents(list[Document]) -> list[Document]`` (metadata such as the
page number is copied onto every chunk). Implementing our chunker behind
this interface makes the chunking strategy **pluggable**:

* ``SentenceAwareTextSplitter`` (default): our balanced, whole-sentence
  chunker (``app.preprocessing.chunker``) wrapped as a LangChain splitter.
* ``RecursiveCharacterTextSplitter`` (LangChain's general-purpose splitter):
  cuts at paragraph, then line, then word boundaries. With sentences joined
  by spaces it falls back to word boundaries and may cut sentences in
  half. Selectable with ``INTELLISUM_CHUNKING_STRATEGY=recursive`` so the
  effect of respecting sentence boundaries can be measured (docs/experiments.md 4.3).

Both are configured with the summarization model's own token counter, so
"chunk size" means model tokens, not characters.

The summarizers never import this module: ``app.summarizers.long_document``
works with the framework-free ``chunk_sentences`` by default, and the
LangGraph workflow plugs a splitter in through ``make_chunker``.
"""

import re
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from app.config import get_settings
from app.preprocessing.chunker import Chunk, chunk_sentences
from app.preprocessing.cleaner import clean_text
from app.preprocessing.sentence_splitter import split_sentences

TokenCounter = Callable[[str], int]
Chunker = Callable[[list[str]], list[Chunk]]

_ENDS_SENTENCE = re.compile(r"[.!?][\"')\]]?$")


class SentenceAwareTextSplitter(TextSplitter):
    """Balanced chunks of whole sentences, each at most ``max_tokens`` model tokens."""

    def __init__(self, max_tokens: int, length_function: TokenCounter, overlap_sentences: int = 0):
        # Overlap is measured in sentences (not characters), so LangChain's
        # character-based chunk_overlap is unused.
        super().__init__(chunk_size=max_tokens, chunk_overlap=0, length_function=length_function)
        self.overlap_sentences = overlap_sentences

    def split_sentence_list(self, sentences: list[str]) -> list[Chunk]:
        """Chunk text that is already segmented into sentences (used by the workflow)."""
        return chunk_sentences(sentences, self._chunk_size, self._length_function, self.overlap_sentences)

    def split_text(self, text: str) -> list[str]:
        """LangChain interface: raw text in, chunk texts out."""
        sentences = split_sentences(clean_text(text), min_words=1)
        return [chunk.text for chunk in self.split_sentence_list(sentences)]

    def create_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Document]:
        """LangChain interface; adds chunk index and token count to each chunk's metadata."""
        documents = super().create_documents(texts, metadatas)
        for index, document in enumerate(documents):
            document.metadata = {
                **document.metadata,
                "chunk_index": index,
                "token_count": self._length_function(document.page_content),
            }
        return documents


def recursive_chunks(sentences: list[str], max_tokens: int, count_tokens: TokenCounter) -> list[Chunk]:
    """Chunk with LangChain's general-purpose RecursiveCharacterTextSplitter (for comparison)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0, length_function=count_tokens)
    texts = splitter.split_text(" ".join(sentences))
    return [
        Chunk(
            index=i,
            sentences=[text],
            start_sentence=-1,  # chunks do not align with sentences, so no sentence index
            token_count=count_tokens(text),
            # A chunk not ending in sentence punctuation cut a sentence in half.
            contains_split_sentence=not _ENDS_SENTENCE.search(text),
        )
        for i, text in enumerate(texts)
    ]


def make_chunker(count_tokens: TokenCounter, max_tokens: int, strategy: str | None = None) -> Chunker:
    """
    The chunking function used by the workflow's ``chunk_document`` node,
    chosen by ``Settings.chunking_strategy`` ("sentence" or "recursive").
    """
    settings = get_settings()
    strategy = strategy or settings.chunking_strategy
    if strategy == "sentence":
        splitter = SentenceAwareTextSplitter(max_tokens, count_tokens, settings.chunk_overlap_sentences)
        return splitter.split_sentence_list
    if strategy == "recursive":
        return lambda sentences: recursive_chunks(sentences, max_tokens, count_tokens)
    raise ValueError(f"Unknown chunking strategy '{strategy}'. Use 'sentence' or 'recursive'.")
