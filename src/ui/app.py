"""
src/ui/app.py
──────────────
Streamlit frontend for the RAG chatbot.

Features:
  - Chat-style message history (persisted in session state)
  - Streaming token output
  - Sidebar with model info and controls
  - Graceful error handling with user-friendly messages
  - Completely decoupled from business logic (calls core + services only)
"""

import sys
import os

# Allow running: streamlit run src/ui/app.py from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from src.config import get_settings
from src.core import build_rag_chain
from src.services import get_vector_store_service, get_llm_service
from src.utils import get_logger, VectorStoreNotFoundError

logger = get_logger(__name__)
settings = get_settings()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.app_title,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .info-badge {
        background-color: #f0f4ff;
        border: 1px solid #ccd6ff;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        font-size: 0.85rem;
        color: #3a3a6e;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="main-header">🤖 {settings.app_title}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask me anything about Rohanta\'s background, '
    'skills, experience, or projects.</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown(f'<div class="info-badge">🧠 <b>LLM:</b> {settings.repo_id.split("/")[-1]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-badge">📐 <b>Embeddings:</b> all-MiniLM-L6-v2</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-badge">🗃️ <b>Vector Store:</b> FAISS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-badge">🔍 <b>Retrieval:</b> MMR (k={settings.top_k})</div>', unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built by [Rohanta Bhamare](https://linkedin.com/in/rohanta-bhamare-380611346)")
    st.caption("🔗 [GitHub](https://github.com/rohantabhamar)")

# ── Load resources (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading knowledge base & AI model…")
def load_rag_chain():
    """Load and cache the RAG chain. Runs once per Streamlit session."""
    return build_rag_chain(
        vector_store_service=get_vector_store_service(),
        llm_service=get_llm_service(),
    )


try:
    rag_chain = load_rag_chain()
except VectorStoreNotFoundError:
    st.error(
        "⚠️ **Knowledge base not found.**\n\n"
        "Run the ingestion script first:\n"
        "```bash\npython scripts/ingest.py --input my_data.txt\n```"
    )
    st.stop()
except Exception as exc:
    st.error(f"❌ Failed to initialise the app: `{exc}`")
    logger.exception("App initialisation failed.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render message history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask me about Rohanta's experience, skills, or projects …"):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            try:
                answer: str = rag_chain.invoke(user_input)
            except Exception as exc:
                logger.error("Chain invocation failed: %s", exc)
                answer = (
                    "⚠️ Something went wrong while generating an answer. "
                    "Please try again or rephrase your question."
                )
        st.markdown(answer)

    # 3. Persist assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
