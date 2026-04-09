"""
src/core/retriever.py
──────────────────────
Advanced RAG retrieval pipeline:

  Stage 1 — Hybrid retrieval
    · FAISS vector search  (semantic similarity)
    · BM25 keyword search  (exact term match)
    · Reciprocal Rank Fusion to merge both result lists

  Stage 2 — Reranking
    · EmbeddingsRedundantFilter  — remove near-duplicate chunks
    · FlashRank cross-encoder    — precise relevance scoring

  Stage 3 — Ordering
    · LongContextReorder         — best docs at edges for LLM attention
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain_core.documents import Document

from flashrank import Ranker, RerankRequest

from src.config import get_settings
from src.services.embedding_service import get_embedding_service
from src.utils import get_logger

logger = get_logger(__name__)

# Flash rerank model loaded once at module level
_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        logger.info("Loading FlashRank model...")
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
        logger.info("FlashRank model loaded.")
    return _ranker


def _reciprocal_rank_fusion(
    results: dict[str, list[Document]], k: int = 60
) -> list[Document]:
    """
    Merge multiple retriever result lists using Reciprocal Rank Fusion.
    Score = Σ 1 / (k + rank)  — rewards docs appearing across multiple lists.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for _query, docs in results.items():
        for rank, doc in enumerate(docs):
            doc_id = doc.page_content
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[did] for did in sorted_ids]


def advanced_retrieve(
    query: str,
    vector_store: FAISS,
    all_documents: list[Document],
) -> list[Document]:
    """
    Full advanced retrieval pipeline for a given query.

    Args:
        query:         The search query string.
        vector_store:  Loaded FAISS vector store.
        all_documents: All raw document chunks (needed for BM25).

    Returns:
        Final reranked and reordered list of top documents.
    """
    settings = get_settings()
    embeddings = get_embedding_service().get_embeddings()

    # ── Stage 1a: FAISS vector search ────────────────────────────────────────
    faiss_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.top_k * 3,          # fetch wide — reranking narrows
            "lambda_mult": settings.mmr_lambda,
        },
    )

    # ── Stage 1b: BM25 keyword search ────────────────────────────────────────
    bm25_retriever = BM25Retriever.from_documents(
        all_documents, k=settings.top_k * 3
    )

    # ── Stage 1c: Retrieve from both + manual RRF merge ──────────────────────
    faiss_docs = faiss_retriever.invoke(query)
    bm25_docs  = bm25_retriever.invoke(query)
    fused_docs = _reciprocal_rank_fusion(
        {"faiss": faiss_docs, "bm25": bm25_docs}
    )
    logger.info("Stage 1 — Hybrid retrieval: %d docs after RRF", len(fused_docs))

    # ── Stage 2a: Remove near-duplicate chunks ────────────────────────────────
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    deduped_docs = redundant_filter.transform_documents(fused_docs)
    logger.info("Stage 2a — After dedup: %d docs", len(deduped_docs))

    # ── Stage 2b: FlashRank cross-encoder rerank ──────────────────────────────
    ranker = _get_ranker()
    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(deduped_docs)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerank_request)
    top_k_final = settings.top_k
    reranked_docs = [deduped_docs[r["id"]] for r in reranked_results[:top_k_final]]
    logger.info("Stage 2b — After FlashRank: %d docs", len(reranked_docs))

    # ── Stage 3: Lost-in-the-middle reorder ───────────────────────────────────
    reorder = LongContextReorder()
    final_docs = reorder.transform_documents(reranked_docs)
    logger.info("Stage 3 — Final docs after reorder: %d", len(final_docs))

    return final_docs


def build_retriever(vector_store: FAISS) -> FAISS:
    """
    Kept for backward compatibility.
    Returns the raw FAISS retriever — advanced_retrieve() is the main entrypoint.
    """
    settings = get_settings()
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.top_k,
            "lambda_mult": settings.mmr_lambda,
        },
    )
