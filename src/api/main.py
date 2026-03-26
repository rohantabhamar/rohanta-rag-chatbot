"""
src/api/main.py
────────────────
Optional REST API layer built with FastAPI.
Exposes the RAG chain as an HTTP endpoint for integration with
external clients, dashboards, or microservices.

Endpoints:
    POST /chat       → Ask a question, get an answer
    GET  /health     → Liveness probe for deployment health checks
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.core import build_rag_chain
from src.services import get_vector_store_service, get_llm_service
from src.utils import get_logger, VectorStoreNotFoundError, LLMError

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title=settings.app_title,
    description="Production-grade RAG chatbot API",
    version="1.0.0",
)

# ── Lazy-load chain at startup ────────────────────────────────────────────────
_rag_chain = None


def get_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = build_rag_chain(
            vector_store_service=get_vector_store_service(),
            llm_service=get_llm_service(),
        )
    return _rag_chain


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User question")


class ChatResponse(BaseModel):
    answer: str
    question: str


class HealthResponse(BaseModel):
    status: str
    index_dir: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe — returns 200 if the service is running."""
    return HealthResponse(status="ok", index_dir=settings.index_dir)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer a question using the RAG pipeline.

    Request body:
        { "question": "Tell me about Rohanta's experience." }

    Response:
        { "answer": "...", "question": "..." }
    """
    logger.info("Received question: %s", request.question[:80])
    try:
        chain = get_chain()
        answer: str = chain.invoke(request.question)
    except VectorStoreNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error during inference: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")

    logger.info("Answer generated successfully.")
    return ChatResponse(answer=answer, question=request.question)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
