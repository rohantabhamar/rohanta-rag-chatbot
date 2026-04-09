from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = ""

    # ── LLM ───────────────────────────────────────────────────────────────────
    repo_id: str = "llama-3.1-8b-instant"
    max_new_tokens: int = 1024
    temperature: float = 0.1

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Vector Store ──────────────────────────────────────────────────────────
    index_dir: str = "faiss_index"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 6              # final docs passed to LLM
    mmr_lambda: float = 0.65
    search_type: str = "mmr"
    rerank_top_n: int = 6       # FlashRank top-n (should match top_k)

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 200
    chunk_overlap: int = 50

    # ── App ───────────────────────────────────────────────────────────────────
    app_title: str = "Rohanta's RAG Assistant"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
