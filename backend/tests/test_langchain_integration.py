"""
LangChain integration: loaders -> Document objects -> preprocessing with page
provenance -> pluggable TextSplitter chunking -> workflow.
"""

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from app.core.workflow import run_summarization
from app.documents.base import TextDocument
from app.documents.langchain_loaders import PastedTextLoader, UploadedFileLoader
from app.errors import CorruptedFileError, OCRRequiredError, UnsupportedFileTypeError
from app.graph.summarization_graph import build_summarization_graph
from app.preprocessing.chunker import chunk_sentences
from app.preprocessing.langchain_splitter import SentenceAwareTextSplitter, make_chunker, recursive_chunks
from app.preprocessing.pipeline import join_pages, preprocess, preprocess_documents
from app.summarizers.bart import BARTSummarizer
from tests.conftest import SAMPLE_TEXT, make_docx, make_pdf, make_scanned_pdf
from tests.test_hybrid import StandInModel, variants
from tests.test_tfidf import SOLAR


def words(text: str) -> int:
    return len(text.split())


PAGES = [
    "Solar power capacity grew faster than any other energy source last year. China accounted for most new capacity.",
    "Analysts said falling panel prices were the main driver of growth. Wind power also expanded more slowly.",
    "The report warned that grid infrastructure must improve quickly. Prices have fallen by eighty percent.",
]


# ------------------------------------------------------------------ loaders


def test_pdf_loader_yields_one_document_per_page():
    loader = UploadedFileLoader(make_pdf(PAGES), "report.pdf")
    docs = loader.load()
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].metadata["source"] == "report.pdf"
    assert docs[0].metadata["file_type"] == "pdf" and docs[0].metadata["total_pages"] == 3
    assert "Analysts" in docs[1].page_content


def test_docx_and_txt_loaders_yield_a_single_document():
    docx_docs = UploadedFileLoader(make_docx([SAMPLE_TEXT]), "notes.docx").load()
    txt_docs = UploadedFileLoader(SAMPLE_TEXT.encode(), "notes.txt").load()
    assert len(docx_docs) == len(txt_docs) == 1
    assert "page" not in docx_docs[0].metadata
    assert txt_docs[0].metadata["file_type"] == "txt"


def test_pasted_text_loader():
    [doc] = PastedTextLoader("Some pasted text.").load()
    assert doc.page_content == "Some pasted text." and doc.metadata["source"] == "pasted text"


def test_loader_reuses_phase2_validation():
    with pytest.raises(UnsupportedFileTypeError):
        UploadedFileLoader(b"data", "image.png").load()
    with pytest.raises(CorruptedFileError):
        UploadedFileLoader(b"not a pdf", "fake.pdf").load()
    with pytest.raises(OCRRequiredError):
        UploadedFileLoader(make_scanned_pdf(2), "scan.pdf").load()


def test_loader_exposes_extraction_warnings():
    loader = UploadedFileLoader(make_scanned_pdf(1, text_pages=[SAMPLE_TEXT]), "mixed.pdf")
    docs = loader.load()
    assert [d.metadata["page"] for d in docs] == [2]  # the scanned page 1 produced no Document
    assert "1 of 2 pages" in loader.warnings[0]


# -------------------------------------------------- preprocessing + provenance


def test_sentences_are_mapped_to_their_pages():
    docs = UploadedFileLoader(make_pdf(PAGES), "report.pdf").load()
    pre = preprocess_documents(docs)
    assert pre.sentence_count == 6
    assert pre.sentence_pages == [1, 1, 2, 2, 3, 3]


def test_sentence_continuing_across_page_break_is_rejoined():
    pages = [
        "The first page ends in the middle of a",
        "sentence that continues here. A new sentence starts afterwards on the second page. "
        "The report then discusses solar capacity and falling panel prices in several regions. "
        "It closes with recommendations for grid investment.",
    ]
    pre = preprocess_documents([TextDocument(p, {"page": i}) for i, p in enumerate(pages, 1)])
    assert pre.sentences[0] == "The first page ends in the middle of a sentence that continues here."
    assert pre.sentence_pages == [1, 2, 2, 2]  # attributed to the page it starts on


@pytest.mark.parametrize(
    "pages, expected",
    [
        (["Ends here.", "Starts here."], "Ends here.\n\nStarts here."),
        (["continues", "on the next page."], "continues on the next page."),
        (["summari-", "zation works."], "summarization works."),
        (["Heading", "Body text."], "Heading\n\nBody text."),  # next page starts with a capital
    ],
)
def test_join_pages(pages, expected):
    text, starts = join_pages(pages)
    assert text == expected
    assert starts[0] == 0 and len(starts) == 2


def test_framework_free_documents_work_too():
    # The pipeline only needs .page_content and .metadata: no LangChain required.
    lc = preprocess_documents([Document(page_content=p, metadata={"page": i}) for i, p in enumerate(PAGES, 1)])
    plain = preprocess_documents([TextDocument(p, {"page": i}) for i, p in enumerate(PAGES, 1)])
    assert lc == plain


def test_plain_text_has_no_pages():
    pre = preprocess(" ".join(PAGES))
    assert pre.sentence_pages == [None] * pre.sentence_count


# ------------------------------------------------------------------ splitters


def test_sentence_splitter_is_a_langchain_text_splitter():
    splitter = SentenceAwareTextSplitter(max_tokens=40, length_function=words)
    assert isinstance(splitter, TextSplitter)
    chunks = splitter.split_text(" ".join(SOLAR))
    assert len(chunks) >= 3
    assert all(words(c) <= 40 for c in chunks)
    assert all(c.endswith(".") for c in chunks)  # whole sentences only


def test_sentence_splitter_matches_core_chunker():
    splitter = SentenceAwareTextSplitter(max_tokens=40, length_function=words)
    assert splitter.split_sentence_list(SOLAR) == chunk_sentences(SOLAR, 40, words)


def test_split_documents_keeps_metadata_and_adds_chunk_info():
    splitter = SentenceAwareTextSplitter(max_tokens=20, length_function=words)
    docs = [Document(page_content=p, metadata={"page": i, "source": "r.pdf"}) for i, p in enumerate(PAGES, 1)]
    chunks = splitter.split_documents(docs)
    assert {c.metadata["page"] for c in chunks} == {1, 2, 3}
    assert all(c.metadata["source"] == "r.pdf" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["token_count"] <= 20 for c in chunks)


def test_recursive_splitter_can_cut_sentences_in_half():
    # LangChain's general-purpose splitter falls back to word boundaries.
    chunks = recursive_chunks(SOLAR, 25, words)
    assert all(c.token_count <= 25 for c in chunks)
    assert any(c.contains_split_sentence for c in chunks)
    assert " ".join(c.text for c in chunks).split() == " ".join(SOLAR).split()  # nothing lost
    # ...whereas the sentence-aware splitter never does.
    assert not any(c.contains_split_sentence for c in make_chunker(words, 25, "sentence")(SOLAR))


def test_unknown_strategy():
    with pytest.raises(ValueError):
        make_chunker(words, 100, "magic")


# ------------------------------------------------------------------- workflow


@pytest.fixture
def graph():
    return build_summarization_graph(abstractive=BARTSummarizer(model=StandInModel()))


def test_workflow_accepts_loaded_documents(graph):
    loader = UploadedFileLoader(make_pdf(PAGES * 2), "report.pdf")
    docs = loader.load()
    out = run_summarization(documents=docs, method="textrank", length="medium", warnings=loader.warnings, graph=graph)
    assert out.sentence_pages == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    assert out.metadata["source"] == "report.pdf" and out.metadata["num_documents"] == 6
    selected_pages = [out.sentence_pages[i] for i in out.selected_indices]
    assert all(isinstance(p, int) for p in selected_pages)


def test_long_pdf_chunks_report_their_pages(graph):
    pages = [" ".join(variants(1)) for _ in range(12)]  # ~220 words per page
    docs = UploadedFileLoader(make_pdf(pages), "long.pdf").load()
    out = run_summarization(documents=docs, method="bart", length="short", graph=graph)
    level = out.metadata["reduction_levels"][0]
    chunk_pages = level["chunk_pages"]
    assert len(chunk_pages) == level["num_chunks"] >= 2
    assert chunk_pages[0][0] == 1 and chunk_pages[-1][-1] == 12
    flat = [p for pages_ in chunk_pages for p in pages_]
    assert flat == sorted(flat)  # chunks follow page order


def test_recursive_strategy_runs_through_the_workflow(graph, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "chunking_strategy", "recursive")
    out = run_summarization(" ".join(variants(10)), "bart", "short", graph=graph)
    level = out.metadata["reduction_levels"][0]
    assert level["num_chunks"] >= 2
    assert level["chunk_pages"] is None  # recursive chunks do not align with sentences


def test_text_or_documents_required(graph):
    with pytest.raises(ValueError):
        run_summarization(graph=graph)
    with pytest.raises(ValueError):
        run_summarization("text", documents=[TextDocument("text")], graph=graph)
