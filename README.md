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

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about Rohanta Bhamare's professional background, skills, experience, and projects — built with an advanced multi-stage retrieval pipeline.

🔗 **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/rosneha/rohanta-rag-chatbot)

---

## Advanced RAG Pipeline

This chatbot goes beyond basic vector search. Every query passes through a 4-stage pipeline:

```
User Query
    │
    ▼
Stage 1 — Hybrid Retrieval
    ├── FAISS vector search     (semantic similarity)
    ├── BM25 keyword search     (exact term matching)
    └── Reciprocal Rank Fusion  (merges both result lists → ~50 docs)
    │
    ▼
Stage 2 — Reranking
    ├── EmbeddingsRedundantFilter   (removes near-duplicate chunks)
    └── FlashRank cross-encoder     (precise relevance scoring → top 6 docs)
    │
    ▼
Stage 3 — Ordering
    └── LongContextReorder  (best docs placed at edges for LLM attention)
    │
    ▼
Stage 4 — Generation
    └── Groq Llama3  (answers from clean, ranked, ordered context)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — `llama-3.1-8b-instant` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS with MMR retrieval |
| Keyword Search | BM25 (rank-bm25) |
| Rank Fusion | Reciprocal Rank Fusion (custom implementation) |
| Reranking | FlashRank — `ms-marco-MiniLM-L-12-v2` |
| Deduplication | EmbeddingsRedundantFilter |
| Context Ordering | LongContextReorder |
| Orchestration | LangChain — RunnableLambda, PromptTemplate |
| Frontend | Streamlit |
| Deployment | HuggingFace Spaces (Docker SDK) |

---

## Features

- **Hybrid search** — combines semantic vector search (FAISS) with exact keyword search (BM25) for broader, more accurate retrieval
- **Reciprocal Rank Fusion** — merges results from both retrievers using the RRF formula: score = Σ 1/(k + rank), rewarding docs that appear across multiple queries
- **FlashRank reranking** — cross-encoder model scores each retrieved doc precisely against the query, far more accurate than cosine similarity alone
- **Lost in the Middle ordering** — reorders final docs so the most relevant appear at the edges of the LLM context window, where attention is strongest
- **Full conversation memory** — remembers the entire chat history (safely truncated at 6000 chars for long sessions)
- **Query expansion** — maps vague queries like "last job" or "tech stack" to rich, specific search terms
- **Hallucination guard** — strict prompt rules prevent the LLM from inventing facts not present in the knowledge base
- **Out-of-scope detection** — politely declines general knowledge questions outside Rohanta's background

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/rohantabhamar/rohanta_rag
cd rohanta_rag

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
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

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
GROQ_API_KEY=gsk_your_key_here
```

For HuggingFace Spaces, set `GROQ_API_KEY` as a **Space Secret** in:
`Settings → Variables and secrets → New secret`

---

## Project Structure

```
rohanta_rag/
├── src/
│   ├── core/
│   │   ├── chain.py        # RAG chain — query expansion, memory, pipeline
│   │   ├── prompts.py      # Prompt templates with anti-hallucination rules
│   │   └── retriever.py    # Advanced retrieval — hybrid search + RRF + FlashRank + reorder
│   ├── config/
│   │   └── settings.py     # Pydantic settings (top_k, mmr_lambda, rerank_top_n, etc.)
│   ├── services/
│   │   ├── embedding_service.py    # HuggingFace embeddings
│   │   ├── llm_service.py          # Groq LLM
│   │   └── vector_store_service.py # FAISS index management
│   ├── ui/
│   │   └── app.py          # Streamlit frontend
│   └── utils/              # Logging + custom exceptions
├── scripts/
│   └── ingest.py           # Knowledge base ingestion CLI
├── eval/
│   ├── run_evals.py        # RAGAS evaluation pipeline
│   └── test_questions.json # Evaluation question set
├── tests/                  # Unit tests
├── .streamlit/
│   └── config.toml         # Streamlit config (disables noisy file watcher)
├── my_data.txt             # Knowledge base
├── Dockerfile              # HuggingFace Spaces Docker deployment
├── start.sh                # Entrypoint — runs ingest then launches Streamlit
├── packages.txt            # System packages for HuggingFace
└── requirements.txt        # Python dependencies
```

---

## Retrieval Pipeline — Technical Detail

### Stage 1: Hybrid Search + RRF

Two retrievers run in parallel on every query:

- **FAISS (MMR)** fetches `top_k × 3` docs using Maximum Marginal Relevance — balancing relevance and diversity
- **BM25** fetches `top_k × 3` docs using TF-IDF keyword scoring — catches exact term matches that semantic search misses

Results are merged using **Reciprocal Rank Fusion**:

```
RRF(doc) = Σ  1 / (60 + rank)
```

A doc appearing consistently across both retrievers scores higher than one ranked #1 in only one.

### Stage 2: FlashRank Reranking

After deduplication, every remaining doc is scored against the original query using a **cross-encoder** (`ms-marco-MiniLM-L-12-v2`). Unlike bi-encoders that compare embeddings separately, cross-encoders read the query and document together — dramatically more accurate. Only the top `top_k` docs survive.

### Stage 3: Lost in the Middle

Research shows LLMs lose focus on context placed in the middle of a long prompt. `LongContextReorder` reorders docs so the **most relevant are at the start and end**, and least relevant in the middle.

---

## Evaluation

Run RAGAS evaluation against the test question set:

```bash
python eval/run_evals.py
```

Metrics tracked: Answer Relevance, Faithfulness, Context Recall, Chunk Retrieval Quality.

---

## Built by

**Rohanta Bhamare** — AI/ML Engineer | Frankfurt, Germany

[![LinkedIn](https://img.shields.io/badge/LinkedIn-rohanta--bhamare-blue?logo=linkedin)](https://linkedin.com/in/rohanta-bhamare-380611346)
[![GitHub](https://img.shields.io/badge/GitHub-rohantabhamar-black?logo=github)](https://github.com/rohantabhamar)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-rosneha-yellow?logo=huggingface)](https://huggingface.co/rosneha)
