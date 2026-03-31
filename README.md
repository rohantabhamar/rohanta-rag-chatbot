---
title: Rohanta RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Rohanta's Personal RAG Chatbot

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about Rohanta Bhamare's professional background, skills, experience, and projects.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq API — `llama-3.1-8b-instant` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS with MMR retrieval (k=8) |
| Orchestration | LangChain — RunnableLambda, PromptTemplate |
| Frontend | Streamlit |
| Deployment | HuggingFace Spaces (Docker SDK) |

## Features

- **Full conversation memory** — remembers the entire chat history (safely truncated at 6000 chars)
- **Query expansion** — maps vague queries (e.g. "last job", "tech stack") to rich FAISS search terms
- **Hallucination guard** — strict prompt rules prevent the LLM from inventing facts not in the knowledge base
- **Out-of-scope detection** — politely declines general knowledge questions
- **MMR retrieval** — diversity-aware chunk selection for richer context

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/rohantabhamar/rohanta_rag
cd rohanta_rag

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Build the FAISS index
python scripts/ingest.py --input my_data.txt

# 6. Run the app
streamlit run src/ui/app.py
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
GROQ_API_KEY=gsk_your_key_here
```

For HuggingFace Spaces, set `GROQ_API_KEY` as a **Space Secret** in Settings → Variables and secrets.

## Project Structure

```
rohanta_rag/
├── src/
│   ├── core/
│   │   ├── chain.py        # RAG chain with query expansion + memory
│   │   ├── prompts.py      # Prompt templates with anti-hallucination rules
│   │   └── retriever.py    # FAISS MMR retriever
│   ├── config/
│   │   └── settings.py     # Pydantic settings
│   ├── services/           # LLM + vector store services
│   ├── ui/
│   │   └── app.py          # Streamlit frontend
│   └── utils/              # Logging + error handling
├── scripts/
│   └── ingest.py           # Knowledge base ingestion CLI
├── my_data.txt             # Knowledge base
├── Dockerfile              # HuggingFace Spaces deployment
└── start.sh                # Ingest + launch script
```

## Built by

**Rohanta Bhamare** — AI Engineer | Frankfurt, Germany  
[LinkedIn](https://linkedin.com/in/rohanta-bhamare-380611346) · [GitHub](https://github.com/rohantabhamar)
