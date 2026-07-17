"""
src/services/embedding_service.py
──────────────────────────────────
Responsible solely for creating the HuggingFace embeddings object.
Wrapping this in a service class makes it easy to swap embedding
providers (OpenAI, Cohere, local models) without touching other layers.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings
from src.utils import get_logger, EmbeddingError

logger = get_logger(__name__)


class EmbeddingService:
    """Manages creation and caching of embedding models."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._embeddings: HuggingFaceEmbeddings | None = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Return (and lazily initialise) the embedding model."""
        if self._embeddings is None:
            print(f">>> [DEBUG] Loading embedding model: {self._model_name}", flush=True)
            logger.info("Loading embedding model: %s", self._model_name)
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self._model_name
                )
                print(">>> [DEBUG] Embedding model loaded successfully.", flush=True)
                logger.info("Embedding model loaded successfully.")
            except Exception as exc:
                print(f">>> [DEBUG] Embedding model FAILED to load: {exc}", flush=True)
                raise EmbeddingError(
                    f"Failed to load embedding model '{self._model_name}': {exc}"
                ) from exc
        return self._embeddings


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return a singleton EmbeddingService instance."""
    return EmbeddingService()