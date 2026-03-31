from langchain_core.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""You are a helpful assistant with detailed knowledge about Rohanta Bhamare.

Answer the question using ONLY the context provided below.
Be specific and direct — extract exact facts like company names, dates, and project names.
Do not say "I don't have that information" if the context contains relevant details.
Do not mention source numbers.
Maximum 5 sentences.

Context:
{context}

Previous conversation:
{chat_history}

Question: {question}

Answer:""",
    input_variables=["context", "chat_history", "question"],
)

CONDENSE_PROMPT_TEMPLATE = PromptTemplate(
    template="""Rewrite the follow-up question as a complete standalone question.
Expand short questions like "his last company" to "What was Rohanta's last company?".
Return ONLY the rewritten question.

History:
{chat_history}

Follow-up: {question}

Standalone question:""",
    input_variables=["chat_history", "question"],
)