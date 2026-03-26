"""
src/core/retriever.py
──────────────────────
Builds the LangChain retriever from the loaded FAISS vector store.

Isolating retriever configuration here means search strategy
(similarity, MMR, hybrid) can be swapped without touching the chain.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import get_settings
from src.utils import get_logger

logger = get_logger(__name__)


def build_retriever(vector_store: FAISS) -> VectorStoreRetriever:
    """
    Create a configured retriever from a FAISS vector store.

    Uses MMR (Max Marginal Relevance) by default to balance relevance
    and diversity in retrieved chunks — preventing repetitive context.

    Args:
        vector_store: A loaded FAISS vector store instance.

    Returns:
        A LangChain VectorStoreRetriever ready to use in a chain.
    """
    settings = get_settings()

    logger.info(
        "Building retriever | search_type=%s | top_k=%d | mmr_lambda=%.2f",
        settings.search_type,
        settings.top_k,
        settings.mmr_lambda,
    )

    if settings.search_type == "mmr":
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.top_k,
                "lambda_mult": settings.mmr_lambda,
            },
        )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.top_k},
        )

    return retriever
