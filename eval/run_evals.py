import sys, os, math, json

# ── Fix paths ──────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.services import get_vector_store_service, get_llm_service
from src.core import build_rag_chain, build_retriever


def safe_score(val):
    """Handle NaN or list results caused by timeouts."""
    if isinstance(val, list):
        valid = [v for v in val if v is not None and not math.isnan(float(v))]
        return round(float(sum(valid) / len(valid)), 4) if valid else None
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 4)
    except Exception:
        return None


def run_evaluation():
    with open(os.path.join(os.path.dirname(__file__), "test_questions.json")) as f:
        test_cases = json.load(f)

    # ── Boot your existing services ────────────────────────────────────
    vs_service   = get_vector_store_service()
    llm_service  = get_llm_service()
    chain        = build_rag_chain(vs_service, llm_service)
    vector_store = vs_service.load()
    retriever    = build_retriever(vector_store)

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"Running eval on {len(test_cases)} questions...\n")

    for tc in test_cases:
        q = tc["question"]

        answer = chain.invoke({"question": q, "chat_history": []})
        if not isinstance(answer, str):
            answer = str(answer)

        docs    = retriever.invoke(q)
        context = [doc.page_content for doc in docs]

        questions.append(q)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(tc["ground_truth"])

        print(f"  ✓ {q[:70]}...")

    # ── Build RAGAS dataset ────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # ── Wrap LLM + embeddings for RAGAS ───────────────────────────────
    ragas_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # ── Assign LLM to each metric ──────────────────────────────────────
    faithfulness.llm            = ragas_llm
    context_recall.llm          = ragas_llm
    context_precision.llm       = ragas_llm
    answer_relevancy.llm        = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    # ── Run config: longer timeout to avoid Groq timeouts ─────────────
    run_config = RunConfig(
        timeout=180,
        max_retries=5,
        max_wait=90,
        max_workers=1,
    )

    print("\nRunning RAGAS scoring...\n")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        run_config=run_config,
    )

    # ── Safe score extraction (handles NaN / timeouts) ─────────────────
    summary = {
        "faithfulness":      safe_score(results["faithfulness"]),
        "answer_relevancy":  safe_score(results["answer_relevancy"]),
        "context_recall":    safe_score(results["context_recall"]),
        "context_precision": safe_score(results["context_precision"]),
    }

    print("\n=== RAGAS Results ===")
    for k, v in summary.items():
        status = f"{v:.4f}" if v is not None else "TIMEOUT — rerun script"
        print(f"  {k:25s}: {status}")

    # ── Save results ───────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "results", "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({
            "summary":       summary,
            "num_questions": len(test_cases),
        }, f, indent=2)

    print(f"\nSaved → eval/results/eval_results.json")
    return results


if __name__ == "__main__":
    run_evaluation()