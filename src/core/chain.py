from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.core.prompts import RAG_PROMPT_TEMPLATE, CONDENSE_PROMPT_TEMPLATE
from src.core.retriever import build_retriever
from src.services import VectorStoreService, LLMService
from src.utils import get_logger

logger = get_logger(__name__)

# Query expansion map — common short queries → better search terms
QUERY_EXPANSIONS = {
    "last company": "Lyceum Infotech company Rohanta worked",
    "previous company": "Lyceum Infotech company Rohanta worked",
    "last job": "Lyceum Infotech MLOps engineer job",
    "previous job": "Lyceum Infotech job experience",
    "projects": "RAG chatbot machine efficiency predictor stroke prediction deployed",
    "his project": "RAG chatbot machine efficiency predictor deployed project",
    "work experience": "Lyceum Infotech assistant professor experience",
    "companies": "Lyceum Infotech Guru Gobind Singh Foundation",
    "where did he work": "Lyceum Infotech Guru Gobind Singh Foundation",
    "education": "SSC HSC bachelor engineering master CGPA",
    "degree": "bachelor engineering master CGPA university",
    "qualification": "SSC HSC bachelor engineering master CGPA",
    "skills": "Python LangChain FAISS Docker FastAPI PyTorch",
    "tech stack": "Python LangChain FAISS Docker FastAPI PyTorch AWS",
    "contact": "email phone LinkedIn GitHub Frankfurt Germany",
    "location": "Frankfurt Germany",
}


def _expand_query(question: str) -> str:
    """Expand short/vague queries into better FAISS search terms."""
    q_lower = question.lower()
    for pattern, expansion in QUERY_EXPANSIONS.items():
        if pattern in q_lower:
            logger.info("Query expanded: '%s' → '%s'", question, expansion)
            return expansion
    return question


def _format_docs(documents: list[Document]) -> str:
    if not documents:
        return "No relevant context found."
    return "\n\n".join(
        f"[Source {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )


def _format_history(chat_history: list[dict]) -> str:
    if not chat_history:
        return "No previous conversation."
    lines = []
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def build_rag_chain(
    vector_store_service: VectorStoreService,
    llm_service: LLMService,
) -> Runnable:

    logger.info("Assembling RAG chain with memory + query expansion...")

    vector_store = vector_store_service.load()
    retriever = build_retriever(vector_store)
    chat_model = llm_service.get_chat_model()
    parser = StrOutputParser()

    def condense_question(inputs: dict) -> str:
        question = inputs["question"]
        history = inputs.get("chat_history", [])
        if not history:
            return question
        history_str = _format_history(history)
        condensed = (
            CONDENSE_PROMPT_TEMPLATE | chat_model | parser
        ).invoke({
            "chat_history": history_str,
            "question": question,
        })
        return condensed.strip()

    def build_final_inputs(inputs: dict) -> dict:
        # Step 1 — condense follow-up questions
        standalone = condense_question(inputs)

        # Step 2 — expand query for better FAISS retrieval
        search_query = _expand_query(standalone)

        # Step 3 — retrieve and format context
        docs = retriever.invoke(search_query)
        context = _format_docs(docs)

        # Step 4 — format history
        history_str = _format_history(inputs.get("chat_history", []))

        logger.info(
            "Search query: '%s' | Docs retrieved: %d",
            search_query[:60],
            len(docs),
        )

        return {
            "context": context,
            "chat_history": history_str,
            "question": inputs["question"],
        }

    chain: Runnable = (
        RunnableLambda(build_final_inputs)
        | RAG_PROMPT_TEMPLATE
        | chat_model
        | parser
    )

    logger.info("RAG chain assembled successfully.")
    return chain