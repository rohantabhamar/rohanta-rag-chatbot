# 🤖 Rohanta's Personal RAG Chatbot

> **Production-grade, modular Retrieval-Augmented Generation (RAG) system** built with LangChain, FAISS, HuggingFace, and Streamlit.

---

## 🏗️ Architecture Overview

```
rohanta_rag/
├── src/
│   ├── config/          # Centralized configuration (env vars, constants)
│   ├── core/            # Core RAG engine (retriever, chain builder)
│   ├── services/        # Embedding, LLM, vector store services
│   ├── api/             # FastAPI backend (optional REST API mode)
│   ├── ui/              # Streamlit frontend
│   └── utils/           # Logging, error handling, helpers
├── scripts/
│   └── ingest.py        # CLI tool to build/update FAISS index
├── tests/               # Unit + integration tests
├── .env.example         # Environment variable template
├── pyproject.toml       # Poetry dependency management
└── Dockerfile           # Containerized deployment
```

## ✨ Key Features

- **Modular Architecture** — Each component is independently replaceable
- **MMR Retrieval** — Max Marginal Relevance for diverse, non-redundant context
- **Chat History** — Full multi-turn conversation memory
- **Streaming UI** — Word-by-word streaming output in Streamlit
- **Structured Logging** — JSON-structured logs for production observability
- **Type-Safe** — Full type annotations with Pydantic models
- **Tested** — pytest unit + integration test suite
- **Dockerized** — One-command deployment

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/rohantabhamar/rag-chatbot
cd rag-chatbot
pip install -r requirements.txt   # or: poetry install
```

### 2. Configure Environment

```bash
cp .env.example .env
# Add your HUGGINGFACEHUB_API_TOKEN to .env
```

### 3. Ingest Your Data

```bash
python scripts/ingest.py --input my_data.txt
```

### 4. Run the App

```bash
streamlit run src/ui/app.py
```

### 5. (Optional) Run REST API

```bash
uvicorn src.api.main:app --reload
```

## 🐳 Docker

```bash
docker build -t rohanta-rag .
docker run -p 8501:8501 --env-file .env rohanta-rag
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 🔧 Configuration

All configuration lives in `src/config/settings.py` and `.env`:

| Variable | Default | Description |
|---|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | required | HuggingFace API token |
| `INDEX_DIR` | `faiss_index` | FAISS index directory |
| `REPO_ID` | `mistralai/Mistral-7B-Instruct-v0.2` | LLM model ID |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `500` | Document chunk size |
| `CHUNK_OVERLAP` | `50` | Chunk overlap tokens |
| `TOP_K` | `5` | Retrieved documents per query |
| `MAX_NEW_TOKENS` | `512` | Max generation tokens |

---

*Built by [Rohanta Bhamare](https://linkedin.com/in/rohanta-bhamare-380611346) — AI/ML Engineer specializing in LLM Systems & RAG Architectures*
