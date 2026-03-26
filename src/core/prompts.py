from langchain_core.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""You are a helpful assistant that answers questions about Rohanta Bhamare.

Answer the question using ONLY the context provided below.
Be direct, specific, and concise — 3 to 5 sentences maximum.
Do not mention source numbers. Do not add extra questions or answers.
If the answer is not in the context, say: "I don't have that information."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

CONDENSE_PROMPT_TEMPLATE = PromptTemplate(
    template="""Rewrite the follow-up question as a standalone question.
Return ONLY the rewritten question, nothing else.

Chat History:
{chat_history}

Follow-up: {question}

Standalone Question:""",
    input_variables=["chat_history", "question"],
)