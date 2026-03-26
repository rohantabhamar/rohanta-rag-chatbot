"""
src/services/vector_store_service.py
─────────────────────────────────────
Handles all FAISS vector store operations:
  - Loading an existing index from disk
  - Saving a new index to disk
  - Creating an index from documents

Isolating these operations here means the retriever and
ingestion pipeline never interact with FAISS directly.
"""

import os
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import get_settings
from src.services.embedding_service import get_embedding_service
from src.utils import get_logger, VectorStoreNotFoundError

logger = get_logger(__name__)


class VectorStoreService:
    """Manages FAISS vector store lifecycle."""

    def __init__(self, index_dir: str | None = None) -> None:
        settings = get_settings()
        self._index_dir = index_dir or settings.index_dir
        self._store: FAISS | None = None

    # ── Public Interface ──────────────────────────────────────────────────────

    def load(self) -> FAISS:
        """Load the FAISS index from disk. Raises VectorStoreNotFoundError if missing."""
        if self._store is not None:
            return self._store

        if not os.path.isdir(self._index_dir):
            raise VectorStoreNotFoundError(
                f"FAISS index not found at '{self._index_dir}'. "
                "Run `python scripts/ingest.py` first."
            )

        logger.info("Loading FAISS index from: %s", self._index_dir)
        embeddings = get_embedding_service().get_embeddings()
        self._store = FAISS.load_local(
            self._index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded. Total vectors: %d", self._store.index.ntotal)
        return self._store

    def save(self, store: FAISS) -> None:
        """Persist the FAISS index to disk."""
        logger.info("Saving FAISS index to: %s", self._index_dir)
        store.save_local(self._index_dir)
        self._store = store
        logger.info("FAISS index saved successfully.")

    def build_from_documents(self, documents: list[Document]) -> FAISS:
        """Build a brand-new FAISS index from a list of LangChain Documents."""
        logger.info("Building FAISS index from %d document chunks …", len(documents))
        embeddings = get_embedding_service().get_embeddings()
        store = FAISS.from_documents(documents, embeddings)
        self.save(store)
        return store


@lru_cache(maxsize=1)
def get_vector_store_service() -> VectorStoreService:
    """Return a singleton VectorStoreService instance."""
    return VectorStoreService()
