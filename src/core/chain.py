"""
src/core/chain.py
──────────────────
Builds the RAG chain using LangChain's composable LCEL (LangChain
Expression Language) primitives.

Pipeline:
    User Question
         │
         ▼
  ┌─────────────────────────────┐
  │  RunnableParallel           │
  │  ┌───────────┐ ┌──────────┐ │
  │  │ retriever │ │passthrough│ │
  │  │ → format  │ │ question  │ │
  │  └───────────┘ └──────────┘ │
  └─────────────────────────────┘
         │
         ▼
    PromptTemplate
         │
         ▼
      ChatLLM
         │
         ▼
   StrOutputParser
         │
         ▼
      Answer (str)
"""

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from src.core.prompts import RAG_PROMPT_TEMPLATE
from src.core.retriever import build_retriever
from src.services import VectorStoreService, LLMService
from src.utils import get_logger

logger = get_logger(__name__)


def _format_docs(documents: list[Document]) -> str:
    """
    Concatenate retrieved document chunks into a single context string.

    Each chunk is separated by a blank line for readability within the prompt.
    """
    return "\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )


def build_rag_chain(
    vector_store_service: VectorStoreService,
    llm_service: LLMService,
) -> Runnable:
    """
    Assemble and return the complete RAG chain.

    Args:
        vector_store_service: Service exposing the FAISS vector store.
        llm_service: Service exposing the chat-wrapped LLM.

    Returns:
        A composable LangChain Runnable that accepts a question str
        and returns an answer str.
    """
    logger.info("Assembling RAG chain …")

    vector_store = vector_store_service.load()
    retriever = build_retriever(vector_store)
    chat_model = llm_service.get_chat_model()
    parser = StrOutputParser()

    # Run retrieval and pass-through in parallel so both context and
    # question are available when the prompt template is invoked.
    parallel_step = RunnableParallel(
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
    )

    chain = parallel_step | RAG_PROMPT_TEMPLATE | chat_model | parser

    logger.info("RAG chain assembled successfully.")
    return chain
