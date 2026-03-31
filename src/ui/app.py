import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from src.config import get_settings
from src.core import build_rag_chain
from src.services import get_vector_store_service, get_llm_service, LLMService
from src.utils import get_logger, VectorStoreNotFoundError

logger = get_logger(__name__)
settings = get_settings()

st.set_page_config(
    page_title=settings.app_title,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-header { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .info-badge { background-color: #f0f4ff; border: 1px solid #ccd6ff; border-radius: 8px;
        padding: 0.45rem 0.75rem; font-size: 0.84rem; color: #3a3a6e; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="main-header">🤖 {settings.app_title}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Powered by Groq · Ask me anything about '
    "Rohanta's background, skills, experience, or projects.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(f'<div class="info-badge">🧠 <b>LLM:</b> {settings.repo_id}</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-badge">📐 <b>Embeddings:</b> all-MiniLM-L6-v2</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-badge">🗃️ <b>Vector Store:</b> FAISS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-badge">🔍 <b>Retrieval:</b> MMR (k={settings.top_k})</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-badge">🧠 <b>Memory:</b> Last 6 messages</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_facts = {}
        st.rerun()

    st.divider()
    st.caption("Built by [Rohanta Bhamare](https://linkedin.com/in/rohanta-bhamare-380611346)")
    st.caption("🔗 [GitHub](https://github.com/rohantabhamar)")


@st.cache_resource(show_spinner="⏳ Loading knowledge base & AI model...")
def load_rag_chain():
    return build_rag_chain(
        vector_store_service=get_vector_store_service(),
        llm_service=get_llm_service(),
    )


try:
    rag_chain = load_rag_chain()
except VectorStoreNotFoundError:
    st.error(
        "⚠️ **Knowledge base not found.**\n\n"
        "Run: `python scripts/ingest.py --input my_data.txt`"
    )
    st.stop()
except Exception as exc:
    st.error(f"❌ Failed to initialise: `{exc}`")
    logger.exception("App init failed.")
    st.stop()


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_facts" not in st.session_state:
    st.session_state.user_facts = {}


def extract_user_facts(text: str) -> dict:
    """Extract personal facts the user mentions about themselves."""
    facts = {}
    text_lower = text.lower()

    # Name detection
    import re
    name_match = re.search(
        r"(?:i am|i'm|my name is|call me)\s+([a-zA-Z]+)", text_lower
    )
    if name_match:
        facts["name"] = name_match.group(1).capitalize()

    return facts


def build_system_context() -> str:
    """Build context string from known user facts."""
    if not st.session_state.user_facts:
        return ""
    lines = ["Information about the user asking:"]
    for key, val in st.session_state.user_facts.items():
        lines.append(f"- User's {key}: {val}")
    return "\n".join(lines)


# ── Render history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input(
    "Ask me about Rohanta's experience, skills, or projects..."
):
    # Extract and store any personal facts the user mentions
    new_facts = extract_user_facts(user_input)
    st.session_state.user_facts.update(new_facts)

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if user is asking about themselves
                user_question_lower = user_input.lower()
                personal_keywords = ["my name", "what is my", "who am i", "tell me my"]

                if any(kw in user_question_lower for kw in personal_keywords):
                    # Answer from stored facts directly
                    if st.session_state.user_facts:
                        facts_str = ", ".join(
                            f"your {k} is {v}"
                            for k, v in st.session_state.user_facts.items()
                        )
                        answer = f"Based on what you told me earlier, {facts_str}."
                    else:
                        answer = "You haven't told me anything about yourself yet!"
                else:
                    # Build system context from user facts
                    system_ctx = build_system_context()

                    # Inject user context into question if relevant
                    augmented_question = user_input
                    if system_ctx and new_facts:
                        augmented_question = f"{user_input}\n\n[Context: {system_ctx}]"

                    answer: str = rag_chain.invoke({
                        "question": augmented_question,
                        "chat_history": st.session_state.messages[:-1],
                    })

                    # Personalise answer if we know the user's name
                    user_name = st.session_state.user_facts.get("name")
                    if user_name and len(answer) > 0:
                        if not answer.startswith(user_name):
                            answer = answer

            except Exception as exc:
                logger.error("Chain invocation failed: %s", exc)
                answer = "Something went wrong. Please try again."

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})