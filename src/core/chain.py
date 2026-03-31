"""
src/core/chain.py
──────────────────
RAG chain with full conversation memory + query expansion.

Features:
  - Full infinite conversation memory (safe truncation at 6000 chars)
  - Query expansion for better FAISS retrieval (applied to raw + condensed question)
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
    # ── Last / previous job ───────────────────────────────────────────────────
    "last company"             : "Lyceum Infotech company Rohanta worked",
    "previous company"         : "Lyceum Infotech company Rohanta worked",
    "last job"                 : "Lyceum Infotech MLOps engineer job",
    "previous job"             : "Lyceum Infotech job experience",
    "last work"                : "Lyceum Infotech MLOps engineer September 2022 December 2024 achievements",
    "last worked"              : "Lyceum Infotech MLOps engineer September 2022 December 2024 achievements",
    "as an employee"           : "Lyceum Infotech MLOps engineer September 2022 December 2024 achievements",
    "as employee"              : "Lyceum Infotech MLOps engineer September 2022 December 2024 achievements",
    "previous employment"      : "Lyceum Infotech MLOps engineer September 2022 December 2024 achievements",
    "previous role"            : "Lyceum Infotech MLOps engineer September 2022 December 2024 achievements",
    "previous position"        : "Lyceum Infotech MLOps engineer September 2022 December 2024 achievements",
    "achieve"                  : "Lyceum Infotech 40% cost reduction Jenkins CI/CD 15 applications RAG LangChain Terraform Kubernetes Docker",
    "achievement"              : "Lyceum Infotech 40% cost reduction Jenkins CI/CD 15 applications RAG LangChain Terraform Kubernetes Docker",
    "accomplishment"           : "Lyceum Infotech 40% cost reduction Jenkins CI/CD RAG LangChain MLOps",
    "lyceum"                   : "Lyceum Infotech MLOps engineer September 2022 December 2024 40% cost reduction Jenkins CI/CD RAG LangChain Terraform Kubernetes Docker",
    "mlops engineer"           : "Lyceum Infotech MLOps engineer September 2022 December 2024 40% cost reduction Jenkins CI/CD RAG LangChain",
    # ── Projects ──────────────────────────────────────────────────────────────
    "projects"                 : "RAG chatbot machine efficiency predictor stroke prediction deployed HuggingFace Render",
    "his project"              : "RAG chatbot machine efficiency predictor deployed project",
    "your project"             : "RAG chatbot machine efficiency predictor deployed project",
    "deployed"                 : "RAG chatbot machine efficiency predictor stroke prediction HuggingFace Render",
    # ── Work experience ────────────────────────────────────────────────────────
    "work experience"          : "Lyceum Infotech assistant professor experience MLOps engineer",
    "companies"                : "Lyceum Infotech Guru Gobind Singh Foundation",
    "where did he work"        : "Lyceum Infotech Guru Gobind Singh Foundation",
    "where he worked"          : "Lyceum Infotech Guru Gobind Singh Foundation",
    # ── Education ─────────────────────────────────────────────────────────────
    "education"                : "SSC HSC bachelor engineering master CGPA MSc BSBI Berlin ongoing no grade",
    "degree"                   : "bachelor engineering master CGPA university MSc BSBI Berlin ongoing",
    "qualification"            : "SSC HSC bachelor engineering master CGPA MSc BSBI Berlin ongoing",
    "academic"                 : "SSC HSC bachelor engineering master CGPA MSc BSBI Berlin ongoing",
    "grade"                    : "bachelor engineering CGPA 8 first class distinction MSc ongoing no grade",
    "university"               : "Savitribai Phule Pune University BSBI Berlin engineering master bachelor",
    # ── Skills / tech stack ────────────────────────────────────────────────────
    "skills"                   : "Python PyTorch TensorFlow LangChain FAISS Docker FastAPI AWS Kubernetes Terraform Jenkins",
    "tech stack"               : "Python PyTorch TensorFlow LangChain FAISS Docker FastAPI AWS Kubernetes Terraform Jenkins",
    "core tech"                : "Python PyTorch TensorFlow LangChain FAISS Docker FastAPI AWS Kubernetes Terraform Jenkins",
    "technologies"             : "Python PyTorch TensorFlow LangChain FAISS Docker FastAPI AWS Kubernetes Terraform Jenkins",
    "tools"                    : "Python PyTorch TensorFlow LangChain FAISS Docker FastAPI AWS Kubernetes Terraform Jenkins",
    # ── Contact / location ─────────────────────────────────────────────────────
    "contact"                  : "email phone LinkedIn GitHub Frankfurt Germany rohantabhamare22",
    "linkedin"                 : "LinkedIn www.linkedin.com rohanta-bhamare Frankfurt Germany",
    "github"                   : "GitHub github.com rohantabhamar Frankfurt Germany",
    "location"                 : "Frankfurt Germany",
    "email"                    : "rohantabhamare22@gmail.com email Frankfurt Germany",
    # ── Current role ──────────────────────────────────────────────────────────
    "current role"             : "self employed AI ML consultant MSc Berlin Frankfurt",
    "current job"              : "self employed AI ML consultant MSc Berlin Frankfurt",
    "what is he doing"         : "self employed AI ML consultant MSc Berlin pursuing",
    "currently"                : "self employed AI ML consultant MSc Berlin pursuing",
    # ── Chatbot architecture ───────────────────────────────────────────────────
    "llm"                      : "Groq Llama3 llama-3.1-8b-instant large language model RAG LangChain",
    "embedding"                : "sentence-transformers all-MiniLM-L6-v2 HuggingFace embeddings FAISS",
    "embedding model"          : "sentence-transformers all-MiniLM-L6-v2 HuggingFace sentence transformers",
    "vector store"             : "FAISS vector store MMR retrieval sentence-transformers all-MiniLM-L6-v2",
    "this chatbot"             : "Groq Llama3 all-MiniLM-L6-v2 FAISS MMR LangChain Streamlit rohanta_rag",
    "how does this chatbot"    : "Groq Llama3 all-MiniLM-L6-v2 FAISS MMR LangChain Streamlit architecture",
    "powers this"              : "Groq Llama3 llama-3.1-8b-instant all-MiniLM-L6-v2 FAISS",
    "rag architecture"         : "FAISS MMR retrieval all-MiniLM-L6-v2 LangChain Groq Llama3 Streamlit",
    # ── Misc ──────────────────────────────────────────────────────────────────
    "research"                 : "RAG retrieval augmented generation LLM research",
    "certifications"           : "master engineering bachelor CGPA academic",
    "deployment"               : "Docker FastAPI HuggingFace Spaces deployed production Render",
    "salary"                   : "Frankfurt Germany AI ML engineer consultant",
    "experience"               : "Lyceum Infotech assistant professor AI ML engineer years",
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
            "chat_history": [...]
        }
    """
    logger.info("Assembling RAG chain with full memory + query expansion...")

    vector_store = vector_store_service.load()
    retriever    = build_retriever(vector_store)
    chat_model   = llm_service.get_chat_model()
    parser       = StrOutputParser()

    def condense_question(inputs: dict) -> str:
        """Rewrite a follow-up into a standalone question. Skip if no history."""
        question = inputs["question"]
        history  = inputs.get("chat_history", [])

        if not history:
            return question

        history_str = _safe_format_history(history)

        condensed = (
            CONDENSE_PROMPT_TEMPLATE | chat_model | parser
        ).invoke({
            "chat_history": history_str,
            "question":     question,
        })

        condensed = condensed.strip()
        logger.info("Condensed question: %s", condensed[:100])
        return condensed

    def build_final_inputs(inputs: dict) -> dict:
        """
        Full pipeline:
          1. Try expansion on the RAW original question first
          2. If no expansion matched, condense then expand
          3. Retrieve docs with the best search query
          4. Return dict for the prompt template
        """
        raw_question = inputs["question"]

        # Step 1 — try expansion on the raw question immediately
        # This ensures user phrasing always hits the expansion map
        # before the condense step can rewrite it into different words.
        search_query = _expand_query(raw_question)

        if search_query == raw_question:
            # No expansion matched the raw question — run condense then try again
            standalone   = condense_question(inputs)
            search_query = _expand_query(standalone)
        else:
            # Expansion matched raw question — still condense for the prompt
            # but use the expanded query for retrieval
            standalone = condense_question(inputs)

        # Step 2 — retrieve + format context
        docs    = retriever.invoke(search_query)
        context = _format_docs(docs)

        # Step 3 — full conversation history
        history_str = _safe_format_history(inputs.get("chat_history", []))

        logger.info(
            "Search query: '%s' | Docs retrieved: %d",
            search_query[:60],
            len(docs),
        )

        return {
            "context":      context,
            "chat_history": history_str,
            "question":     raw_question,
        }

    chain: Runnable = (
        RunnableLambda(build_final_inputs)
        | RAG_PROMPT_TEMPLATE
        | chat_model
        | parser
    )

    logger.info("RAG chain assembled successfully.")
    return chain
