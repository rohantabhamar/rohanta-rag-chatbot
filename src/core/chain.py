"""
src/core/chain.py
──────────────────
RAG chain with full conversation memory + query expansion.

Features:
  - Full infinite conversation memory (safe truncation at 6000 chars)
  - Query expansion for better FAISS retrieval
  - Condense step for follow-up questions
  - Source-labelled context formatting
"""

from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.core.prompts import RAG_PROMPT_TEMPLATE, CONDENSE_PROMPT_TEMPLATE
from src.core.retriever import build_retriever
from src.services import VectorStoreService, LLMService
from src.utils import get_logger

logger = get_logger(__name__)


# ── Query expansion map ───────────────────────────────────────────────────────
QUERY_EXPANSIONS = {
    "last company"        : "Lyceum Infotech company Rohanta worked",
    "previous company"    : "Lyceum Infotech company Rohanta worked",
    "last job"            : "Lyceum Infotech MLOps engineer job",
    "previous job"        : "Lyceum Infotech job experience",
    "projects"            : "RAG chatbot machine efficiency predictor stroke prediction deployed",
    "his project"         : "RAG chatbot machine efficiency predictor deployed project",
    "your project"        : "RAG chatbot machine efficiency predictor deployed project",
    "work experience"     : "Lyceum Infotech assistant professor experience",
    "companies"           : "Lyceum Infotech Guru Gobind Singh Foundation",
    "where did he work"   : "Lyceum Infotech Guru Gobind Singh Foundation",
    "where he worked"     : "Lyceum Infotech Guru Gobind Singh Foundation",
    "education"           : "SSC HSC bachelor engineering master CGPA",
    "degree"              : "bachelor engineering master CGPA university",
    "qualification"       : "SSC HSC bachelor engineering master CGPA",
    "skills"              : "Python LangChain FAISS Docker FastAPI PyTorch",
    "tech stack"          : "Python LangChain FAISS Docker FastAPI PyTorch AWS",
    "contact"             : "email phone LinkedIn GitHub Frankfurt Germany",
    "location"            : "Frankfurt Germany",
    "current role"        : "self employed AI ML consultant MSc Berlin",
    "current job"         : "self employed AI ML consultant MSc Berlin",
    "what is he doing"    : "self employed AI ML consultant MSc Berlin pursuing",
    "currently"           : "self employed AI ML consultant MSc Berlin pursuing",
    "research"            : "RAG retrieval augmented generation LLM research",
    "certifications"      : "master engineering bachelor CGPA academic",
    "tools"               : "Python LangChain FAISS Docker FastAPI PyTorch AWS",
    "deployment"          : "Docker FastAPI HuggingFace Spaces deployed production",
    "llm"                 : "large language model LLM transformer RAG LangChain",
    "salary"              : "Frankfurt Germany AI ML engineer consultant",
    "experience"          : "Lyceum Infotech assistant professor AI ML engineer years",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_docs(documents: list[Document]) -> str:
    """Format retrieved docs with source labels."""
    if not documents:
        return "No relevant context found."
    return "\n\n".join(
        f"[Source {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )


def _safe_format_history(
    chat_history: list[dict],
    max_chars: int = 6000,
) -> str:
    """
    Format the FULL conversation history.
    If total length exceeds max_chars, truncate oldest messages
    from the front — keeping the most recent messages always.
    This prevents context window overflow on very long conversations.
    """
    if not chat_history:
        return "No previous conversation."

    lines = []
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    full_history = "\n".join(lines)

    if len(full_history) <= max_chars:
        return full_history

    # Truncate from the front — keep most recent messages
    truncated = full_history[-max_chars:]
    first_newline = truncated.find("\n")
    if first_newline > 0:
        truncated = truncated[first_newline + 1:]

    return "[Earlier messages truncated]\n" + truncated


def _expand_query(question: str) -> str:
    """Expand short or vague queries into better FAISS search terms."""
    q_lower = question.lower()
    for pattern, expansion in QUERY_EXPANSIONS.items():
        if pattern in q_lower:
            logger.info(
                "Query expanded: '%s' → '%s'",
                question[:60],
                expansion[:60],
            )
            return expansion
    return question


# ── Chain builder ─────────────────────────────────────────────────────────────

def build_rag_chain(
    vector_store_service: VectorStoreService,
    llm_service: LLMService,
) -> Runnable:
    """
    Assemble the RAG chain with full memory + query expansion.

    Input format:
        {
            "question": "your question here",
            "chat_history": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        }

    Returns:
        A LangChain Runnable that accepts the above dict and returns
        an answer string.
    """
    logger.info("Assembling RAG chain with full memory + query expansion...")

    vector_store = vector_store_service.load()
    retriever    = build_retriever(vector_store)
    chat_model   = llm_service.get_chat_model()
    parser       = StrOutputParser()

    def condense_question(inputs: dict) -> str:
        """
        Rewrite a follow-up question as a standalone question
        using the conversation history.
        If no history exists, return the original question unchanged.
        """
        question    = inputs["question"]
        history     = inputs.get("chat_history", [])

        if not history:
            return question

        history_str = _safe_format_history(history)

        condensed = (
            CONDENSE_PROMPT_TEMPLATE | chat_model | parser
        ).invoke({
            "chat_history" : history_str,
            "question"     : question,
        })

        condensed = condensed.strip()
        logger.info("Condensed question: %s", condensed[:100])
        return condensed

    def build_final_inputs(inputs: dict) -> dict:
        """
        Full pipeline step:
          1. Condense follow-up into standalone question
          2. Expand query for better FAISS retrieval
          3. Retrieve relevant document chunks
          4. Format history (full, safely truncated)
          5. Return dict ready for prompt template
        """
        # Step 1 — condense
        standalone = condense_question(inputs)

        # Step 2 — expand for better retrieval
        search_query = _expand_query(standalone)

        # Step 3 — retrieve + format context
        docs    = retriever.invoke(search_query)
        context = _format_docs(docs)

        # Step 4 — full conversation history (safely truncated)
        history_str = _safe_format_history(
            inputs.get("chat_history", [])
        )

        logger.info(
            "Search query: '%s' | Docs retrieved: %d",
            search_query[:60],
            len(docs),
        )

        return {
            "context"      : context,
            "chat_history" : history_str,
            "question"     : inputs["question"],
        }

    chain: Runnable = (
        RunnableLambda(build_final_inputs)
        | RAG_PROMPT_TEMPLATE
        | chat_model
        | parser
    )

    logger.info("RAG chain assembled successfully.")
    return chain