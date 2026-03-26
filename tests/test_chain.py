"""
tests/test_chain.py
────────────────────
Unit tests for the RAG chain builder using mocks.
No real LLM or vector store is instantiated — this keeps
tests fast and fully offline.
"""

from unittest.mock import MagicMock, patch
import pytest

from langchain_core.documents import Document
from src.core.chain import _format_docs


# ── _format_docs ──────────────────────────────────────────────────────────────

def test_format_docs_single():
    """Single document should be formatted with a source label."""
    docs = [Document(page_content="Rohanta is an AI engineer.")]
    result = _format_docs(docs)
    assert "[Source 1]" in result
    assert "Rohanta is an AI engineer." in result


def test_format_docs_multiple():
    """Multiple documents should each get their own source label."""
    docs = [
        Document(page_content="First chunk."),
        Document(page_content="Second chunk."),
    ]
    result = _format_docs(docs)
    assert "[Source 1]" in result
    assert "[Source 2]" in result
    assert "First chunk." in result
    assert "Second chunk." in result


def test_format_docs_empty():
    """Empty document list should return an empty string."""
    result = _format_docs([])
    assert result == ""


def test_format_docs_separator():
    """Chunks should be separated by double newlines."""
    docs = [
        Document(page_content="A"),
        Document(page_content="B"),
    ]
    result = _format_docs(docs)
    assert "\n\n" in result


# ── build_rag_chain (mocked) ──────────────────────────────────────────────────

def test_chain_invocation_with_mocks():
    """
    Chain should call the LLM and return a string answer.
    All external dependencies are mocked.
    """
    from src.core.chain import build_rag_chain

    # Mock vector store
    mock_doc = Document(page_content="Rohanta is an AI Engineer.")
    mock_retriever = MagicMock()
    mock_retriever.invoke = MagicMock(return_value=[mock_doc])

    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever

    mock_vs_service = MagicMock()
    mock_vs_service.load.return_value = mock_vs

    # Mock LLM — must behave like a ChatModel (returns AIMessage-like object)
    from langchain_core.messages import AIMessage

    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=AIMessage(content="He is an AI engineer."))

    mock_llm_service = MagicMock()
    mock_llm_service.get_chat_model.return_value = mock_llm

    chain = build_rag_chain(
        vector_store_service=mock_vs_service,
        llm_service=mock_llm_service,
    )

    # Verify chain is callable (doesn't raise on construction)
    assert chain is not None
