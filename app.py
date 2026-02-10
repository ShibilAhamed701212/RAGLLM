"""
Pro RAG Chatbot — Streamlit interface for chatting with your documents.
"""

import shutil
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.config import (
    DATA_DIR,
    VECTOR_DIR,
    TOP_K,
    DEFAULT_TEMPERATURE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LLM_MODEL,
)
from src.utils import (
    get_embeddings, get_llm, load_faiss_index,
    list_ollama_models, pull_ollama_model, delete_ollama_model,
    get_index_stats,
)
from src.ingestion import ingest_all
from src.core import (
    get_retriever, get_rag_stream_with_scores,
    DEFAULT_SYSTEM_PROMPT,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pro RAG Chatbot", page_icon="🤖", layout="wide")

# ── Premium CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white;
    }
    .main-header h1 { color: white !important; margin: 0; font-size: 2rem; }
    .main-header p { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Model badge */
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        color: white;
        margin-top: 0.5rem;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Stats cards */
    .stats-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1rem;
    }
    .stat-card {
        flex: 1;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 6px;
    }
    .score-high { background: #1a472a; color: #4ade80; }
    .score-mid  { background: #422006; color: #fbbf24; }
    .score-low  { background: #450a0a; color: #f87171; }

    /* Sidebar styling */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }

    /* Chat message animation */
    .stChatMessage {
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1;   transform: translateY(0); }
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #667eea;
    }

    .focus-indicator {
        background: linear-gradient(135deg, #f59e0b22, #f59e0b11);
        border: 1px solid #f59e0b44;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: #f59e0b;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def _cached_embeddings():
    return get_embeddings()


@st.cache_resource
def _cached_llm(temperature: float, model: str):
    return get_llm(temperature=temperature, model=model)


@st.cache_resource
def _cached_vector_db(_embeddings):
    return load_faiss_index(_embeddings)


# ── Session state defaults ────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if "response_count" not in st.session_state:
    st.session_state.response_count = 0


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📁 Document Control")

    # -- Indexed files --
    st.subheader("Currently Indexed Files")
    files = list(DATA_DIR.glob("*.*")) if DATA_DIR.exists() else []
    if files:
        for f in files:
            size_kb = f.stat().st_size / 1024
            st.caption(f"✅ {f.name}  ({size_kb:.0f} KB)")
    else:
        st.info("No files in data folder.")

    st.divider()

    # -- Focus mode --
    st.subheader("🎯 Focus Mode")
    focus_path = None
    if files:
        file_names = [f.name for f in files]
        selected = st.selectbox(
            "Select a document to chat with:",
            ["All Documents"] + file_names,
            help="Narrow the AI's focus to a single document.",
        )
        if selected != "All Documents":
            focus_path = DATA_DIR / selected
    else:
        st.caption("Upload files to enable focus mode.")

    st.divider()

    # -- Upload & ingest --
    uploaded = st.file_uploader(
        "Add New Documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "md"],
    )

    if st.button("🚀 Ingest & Index", use_container_width=True):
        if uploaded:
            DATA_DIR.mkdir(exist_ok=True)
            for f in uploaded:
                (DATA_DIR / f.name).write_bytes(f.getbuffer())

            with st.status("Indexing documents…") as status:
                st.write("Extracting and chunking content…")
                ingest_all(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                status.update(label="Index Ready!", state="complete")

            st.cache_resource.clear()
            st.rerun()
        else:
            st.warning("Please upload files first.")

    # -- Reset --
    if st.button("🗑️ Reset All Data", use_container_width=True, type="secondary"):
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        if VECTOR_DIR.exists():
            shutil.rmtree(VECTOR_DIR)
        st.cache_resource.clear()
        st.session_state.history = []
        st.success("System reset! Upload new files.")
        time.sleep(1)
        st.rerun()

    st.divider()

    # -- Select Model --
    st.subheader("🧠 Select Model")
    ollama_models = list_ollama_models()
    model_names = [m["name"] for m in ollama_models]
    model_labels = [f"{m['name']}  ({m['size_gb']} GB)" for m in ollama_models]

    default_idx = model_names.index(LLM_MODEL) if LLM_MODEL in model_names else 0

    selected_model_idx = st.selectbox(
        "Active Ollama Model",
        range(len(model_labels)),
        index=default_idx,
        format_func=lambda i: model_labels[i],
        help="Switch between locally installed Ollama models.",
    )
    selected_model = model_names[selected_model_idx]

    st.divider()

    # -- Download New Model --
    st.subheader("📥 Download Model")
    st.caption("Enter an Ollama model name (e.g. `gemma3:4b`, `phi4-mini`, `mistral`)")
    new_model = st.text_input("Model name", placeholder="gemma3:4b")
    if st.button("⬇️ Pull Model", use_container_width=True):
        if new_model.strip():
            with st.status(f"Downloading {new_model}…", expanded=True) as status:
                st.write("This may take several minutes depending on model size.")
                success = pull_ollama_model(new_model.strip())
                if success:
                    status.update(label=f"✅ {new_model} ready!", state="complete")
                    time.sleep(1)
                    st.rerun()
                else:
                    status.update(label=f"❌ Failed to pull {new_model}", state="error")
        else:
            st.warning("Enter a model name first.")

    st.divider()

    # -- Settings --
    st.subheader("🛠️ Settings")
    temperature = st.slider("Creativity (Temp)", 0.0, 1.0, DEFAULT_TEMPERATURE)
    top_k = st.slider("Search depth (k)", 1, 10, TOP_K)

    st.divider()

    # -- Custom System Prompt --
    st.subheader("✏️ System Prompt")
    with st.expander("Edit AI Behavior"):
        custom_prompt = st.text_area(
            "System prompt",
            value=st.session_state.system_prompt,
            height=200,
            help="Customize how the AI interprets and responds to your questions.",
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True):
                st.session_state.system_prompt = custom_prompt
                st.success("Saved!")
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
                st.rerun()

    st.divider()

    # -- Export & Clear --
    col_exp, col_clr = st.columns(2)
    with col_exp:
        if st.session_state.history:
            chat_md = _export_chat_markdown(st.session_state.history) if callable(
                globals().get("_export_chat_markdown")
            ) else ""
            # Build export content
            lines = [f"# Pro RAG Chat Export\n_Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]
            for msg in st.session_state.history:
                role = "**You**" if msg["role"] == "user" else "**Assistant**"
                lines.append(f"{role}: {msg['content']}\n")
            export_md = "\n".join(lines)

            st.download_button(
                "📄 Export",
                data=export_md,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
    with col_clr:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.history = []
            st.session_state.response_count = 0
            st.rerun()


# ── Initialise resources ──────────────────────────────────────────────────────

embeddings = _cached_embeddings()
vector_db = _cached_vector_db(embeddings)
llm = _cached_llm(temperature, selected_model)

# ── Main chat area ────────────────────────────────────────────────────────────

# Header
st.markdown(f"""
<div class="main-header">
    <h1>🤖 Pro RAG Chatbot</h1>
    <p>Chat with your documents locally and privately.</p>
    <div class="model-badge">🧠 {selected_model}</div>
</div>
""", unsafe_allow_html=True)

# Analytics Dashboard
if vector_db is not None:
    stats = get_index_stats(vector_db)
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">{stats['unique_sources']}</div>
            <div class="stat-label">Documents</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['total_pages']}</div>
            <div class="stat-label">Pages</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['total_chunks']}</div>
            <div class="stat-label">Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{st.session_state.response_count}</div>
            <div class="stat-label">Queries</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if vector_db is None:
    st.warning("No document index found. Upload and ingest PDFs in the sidebar.")
    st.stop()

# Focus mode indicator
if focus_path:
    st.markdown(
        f'<div class="focus-indicator">🎯 Focus Mode: <strong>{focus_path.name}</strong></div>',
        unsafe_allow_html=True,
    )

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("docs"):
            with st.expander(f"📎 Sources ({len(msg['docs'])} chunks)"):
                for doc in msg["docs"]:
                    source = Path(doc.metadata.get("source", "Unknown")).name
                    page = doc.metadata.get("page", "?")
                    score = doc.metadata.get("score")

                    # Score badge
                    if score is not None:
                        if score >= 0.7:
                            badge = f'<span class="score-badge score-high">{score:.0%}</span>'
                        elif score >= 0.4:
                            badge = f'<span class="score-badge score-mid">{score:.0%}</span>'
                        else:
                            badge = f'<span class="score-badge score-low">{score:.0%}</span>'
                        st.markdown(
                            f"**{source}** (Page {page}) {badge}",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write(f"- **{source}** (Page {page})")

                    st.caption(doc.page_content[:300] + "…" if len(doc.page_content) > 300 else doc.page_content)

# Handle new input
if prompt := st.chat_input("Ask about your documents…"):
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        stream, docs = get_rag_stream_with_scores(
            query=prompt,
            db=vector_db,
            llm=llm,
            top_k=top_k,
            filter_path=focus_path,
            chat_history=st.session_state.history[:-1],  # exclude current query
            system_prompt=st.session_state.system_prompt,
        )

        for chunk in stream:
            full_response += getattr(chunk, "content", str(chunk))
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        st.session_state.response_count += 1
        st.session_state.history.append(
            {"role": "assistant", "content": full_response, "docs": docs}
        )

        # Show sources with scores
        with st.expander(f"📎 Sources ({len(docs)} chunks)"):
            for doc in docs:
                source = Path(doc.metadata.get("source", "Unknown")).name
                page = doc.metadata.get("page", "?")
                score = doc.metadata.get("score")

                if score is not None:
                    if score >= 0.7:
                        badge = f'<span class="score-badge score-high">{score:.0%}</span>'
                    elif score >= 0.4:
                        badge = f'<span class="score-badge score-mid">{score:.0%}</span>'
                    else:
                        badge = f'<span class="score-badge score-low">{score:.0%}</span>'
                    st.markdown(
                        f"**{source}** (Page {page}) {badge}",
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(f"- **{source}** (Page {page})")

                st.caption(doc.page_content[:300] + "…" if len(doc.page_content) > 300 else doc.page_content)
