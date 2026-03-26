"""
src/utils/exceptions.py
───────────────────────
Domain-specific exception hierarchy for the RAG application.
Raising typed exceptions instead of bare Exception makes
error handling explicit and aids debugging in production.
"""


class RAGBaseException(Exception):
    """Base class for all application exceptions."""


class VectorStoreNotFoundError(RAGBaseException):
    """Raised when the FAISS index directory does not exist."""


class IngestionError(RAGBaseException):
    """Raised when document ingestion or chunking fails."""


class EmbeddingError(RAGBaseException):
    """Raised when embedding generation fails."""


class LLMError(RAGBaseException):
    """Raised when the LLM call fails or returns unexpected output."""


class RetrievalError(RAGBaseException):
    """Raised when the retrieval step fails."""


class ConfigurationError(RAGBaseException):
    """Raised when required configuration values are missing."""
