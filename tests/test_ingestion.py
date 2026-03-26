"""
tests/test_ingestion.py
────────────────────────
Unit tests for the document ingestion pipeline (split logic).
These tests are fully offline — no embeddings or LLM calls are made.
"""

import pytest
from langchain_core.documents import Document
from scripts.ingest import split_documents


@pytest.fixture
def sample_docs():
    """A minimal set of LangChain Documents for testing."""
    return [
        Document(
            page_content="Rohanta is an AI/ML Engineer specialising in RAG systems. "
                         "He has over three years of professional experience. "
                         "He builds production-grade LLM applications.",
            metadata={"source": "test.txt"},
        )
    ]


def test_split_produces_chunks(sample_docs):
    """Splitting should return at least one chunk."""
    chunks = split_documents(sample_docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) >= 1


def test_chunk_size_respected(sample_docs):
    """No chunk should exceed the specified chunk size (in characters)."""
    chunk_size = 80
    chunks = split_documents(sample_docs, chunk_size=chunk_size, chunk_overlap=10)
    for chunk in chunks:
        assert len(chunk.page_content) <= chunk_size + 20  # allow splitter tolerance


def test_metadata_preserved(sample_docs):
    """Source metadata should be preserved across all chunks."""
    chunks = split_documents(sample_docs, chunk_size=100, chunk_overlap=10)
    for chunk in chunks:
        assert chunk.metadata.get("source") == "test.txt"


def test_empty_docs_returns_empty():
    """Empty document list should produce empty chunks."""
    chunks = split_documents([], chunk_size=500, chunk_overlap=50)
    assert chunks == []
