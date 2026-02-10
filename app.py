"""
Pro RAG Chatbot — Streamlit interface for chatting with your documents.
"""

import json
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
from src.ingestion import ingest_all, ingest_url
from src.core import (
    get_retriever, get_rag_stream_with_scores,
    DEFAULT_SYSTEM_PROMPT,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pro RAG Chatbot", page_icon="🤖", layout="wide")

# ── Theme definitions ─────────────────────────────────────────────────────────

THEMES = {
    "Midnight Purple": {
        "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "sidebar_bg": "linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%)",
        "card_bg": "linear-gradient(135deg, #1a1a2e, #16213e)",
        "accent": "#667eea",
        "accent2": "#764ba2",
        "border": "rgba(102, 126, 234, 0.3)",
    },
    "Ocean Blue": {
        "gradient": "linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%)",
        "sidebar_bg": "linear-gradient(180deg, #0c1222 0%, #0f172a 100%)",
        "card_bg": "linear-gradient(135deg, #0f172a, #1e293b)",
        "accent": "#0ea5e9",
        "accent2": "#2563eb",
        "border": "rgba(14, 165, 233, 0.3)",
    },
    "Emerald": {
        "gradient": "linear-gradient(135deg, #10b981 0%, #059669 100%)",
        "sidebar_bg": "linear-gradient(180deg, #071a12 0%, #0d2818 100%)",
        "card_bg": "linear-gradient(135deg, #0d2818, #14412e)",
        "accent": "#10b981",
        "accent2": "#059669",
        "border": "rgba(16, 185, 129, 0.3)",
    },
    "Sunset": {
        "gradient": "linear-gradient(135deg, #f97316 0%, #ef4444 100%)",
        "sidebar_bg": "linear-gradient(180deg, #1a0f08 0%, #2a1510 100%)",
        "card_bg": "linear-gradient(135deg, #2a1510, #3d1c16)",
        "accent": "#f97316",
        "accent2": "#ef4444",
        "border": "rgba(249, 115, 22, 0.3)",
    },
    "Rose Gold": {
        "gradient": "linear-gradient(135deg, #f43f5e 0%, #ec4899 100%)",
        "sidebar_bg": "linear-gradient(180deg, #1a0a12 0%, #2a1020 100%)",
        "card_bg": "linear-gradient(135deg, #2a1020, #3d1830)",
        "accent": "#f43f5e",
        "accent2": "#ec4899",
        "border": "rgba(244, 63, 94, 0.3)",
    },
}

# ── Session state defaults ────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if "response_count" not in st.session_state:
    st.session_state.response_count = 0
if "saved_sessions" not in st.session_state:
    st.session_state.saved_sessions = {}
if "active_session" not in st.session_state:
    st.session_state.active_session = "Default"
if "theme" not in st.session_state:
    st.session_state.theme = "Midnight Purple"
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0

# Load saved sessions from disk
SESSIONS_FILE = Path(VECTOR_DIR).parent / ".chat_sessions.json"

def _load_sessions():
    if SESSIONS_FILE.exists():
        try:
            data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
            st.session_state.saved_sessions = data
        except Exception:
            pass

def _save_sessions():
    try:
        # Strip 'docs' from history before saving (not JSON serializable)
        clean = {}
        for name, hist in st.session_state.saved_sessions.items():
            clean[name] = [
                {"role": m["role"], "content": m["content"]}
                for m in hist
            ]
        SESSIONS_FILE.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    except Exception:
        pass

_load_sessions()


# ── Dynamic CSS with theme ────────────────────────────────────────────────────

theme = THEMES[st.session_state.theme]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

    /* Main header styling */
    .main-header {{
        background: {theme['gradient']};
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white;
        position: relative;
        overflow: hidden;
    }}
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 200px;
        height: 200px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }}
    .main-header h1 {{ color: white !important; margin: 0; font-size: 2rem; font-weight: 700; }}
    .main-header p {{ color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 0.95rem; }}

    /* Model badge */
    .model-badge {{
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
    }}

    /* Stats cards */
    .stats-row {{
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1rem;
    }}
    .stat-card {{
        flex: 1;
        background: {theme['card_bg']};
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid {theme['border']};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .stat-card .stat-value {{
        font-size: 1.8rem;
        font-weight: 700;
        background: {theme['gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .stat-card .stat-label {{
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }}

    /* Score badge */
    .score-badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 6px;
    }}
    .score-high {{ background: #1a472a; color: #4ade80; }}
    .score-mid  {{ background: #422006; color: #fbbf24; }}
    .score-low  {{ background: #450a0a; color: #f87171; }}

    /* Sidebar styling */
    [data-testid="stSidebar"] > div:first-child {{
        background: {theme['sidebar_bg']};
    }}

    /* Chat message animation */
    .stChatMessage {{
        animation: fadeIn 0.3s ease-in;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1;   transform: translateY(0); }}
    }}

    .focus-indicator {{
        background: linear-gradient(135deg, #f59e0b22, #f59e0b11);
        border: 1px solid #f59e0b44;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: #f59e0b;
        margin-bottom: 0.5rem;
    }}

    /* Quick prompt buttons */
    .quick-prompt {{
        display: inline-block;
        padding: 0.4rem 0.9rem;
        margin: 0.2rem;
        border-radius: 20px;
        background: {theme['card_bg']};
        border: 1px solid {theme['border']};
        color: {theme['accent']};
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .quick-prompt:hover {{
        background: {theme['gradient']};
        color: white;
        border-color: transparent;
    }}

    /* Response metrics */
    .response-metrics {{
        display: flex;
        gap: 1rem;
        padding: 0.4rem 0;
        font-size: 0.75rem;
        color: #666;
    }}
    .metric-item {{
        display: flex;
        align-items: center;
        gap: 4px;
    }}

    /* Session badge */
    .session-badge {{
        background: {theme['card_bg']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        font-size: 0.8rem;
        color: {theme['accent']};
        display: inline-block;
        margin-top: 0.3rem;
    }}
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


# ── Helper: render sources ────────────────────────────────────────────────────

def _render_sources(docs, label="📎 Sources"):
    """Render source documents with score badges."""
    with st.expander(f"{label} ({len(docs)} chunks)"):
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

            content = doc.page_content
            st.caption(content[:300] + "…" if len(content) > 300 else content)


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

    st.divider()

    # -- URL Ingestion --
    st.subheader("🌐 Ingest from URL")
    url_input = st.text_input("Paste a web URL", placeholder="https://example.com/article")
    if st.button("🔗 Fetch & Index URL", use_container_width=True):
        if url_input.strip():
            with st.status("Fetching page…") as status:
                ok, msg = ingest_url(url_input.strip())
                if ok:
                    st.write(msg)
                    st.write("Re-indexing all documents…")
                    ingest_all(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                    status.update(label="✅ URL indexed!", state="complete")
                    st.cache_resource.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    status.update(label=f"❌ Failed: {msg}", state="error")
        else:
            st.warning("Paste a URL first.")

    # -- Reset --
    st.divider()
    if st.button("🗑️ Reset All Data", use_container_width=True, type="secondary"):
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        if VECTOR_DIR.exists():
            shutil.rmtree(VECTOR_DIR)
        st.cache_resource.clear()
        st.session_state.history = []
        st.session_state.response_count = 0
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
    st.caption("e.g. `gemma3:4b`, `phi4-mini`, `mistral`, `deepseek-r1:8b`")
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

    # -- Theme Selector --
    st.subheader("🎨 Theme")
    theme_choice = st.selectbox(
        "Color theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
        label_visibility="collapsed",
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

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

    # -- Chat Sessions --
    st.subheader("💬 Chat Sessions")

    # Save current session
    session_name = st.text_input("Session name", value=st.session_state.active_session)
    col_save, col_new = st.columns(2)
    with col_save:
        if st.button("💾 Save", use_container_width=True, key="save_session"):
            if session_name.strip():
                st.session_state.saved_sessions[session_name] = list(st.session_state.history)
                st.session_state.active_session = session_name
                _save_sessions()
                st.success(f"Saved '{session_name}'")
    with col_new:
        if st.button("🆕 New", use_container_width=True):
            # Save current first
            if st.session_state.history and st.session_state.active_session:
                st.session_state.saved_sessions[st.session_state.active_session] = list(st.session_state.history)
                _save_sessions()
            st.session_state.history = []
            st.session_state.active_session = f"Chat {len(st.session_state.saved_sessions) + 1}"
            st.session_state.response_count = 0
            st.rerun()

    # Load saved session
    if st.session_state.saved_sessions:
        load_session = st.selectbox(
            "Load session",
            ["—"] + list(st.session_state.saved_sessions.keys()),
        )
        if load_session != "—" and st.button("📂 Load", use_container_width=True):
            st.session_state.history = list(st.session_state.saved_sessions[load_session])
            st.session_state.active_session = load_session
            st.rerun()

    st.divider()

    # -- Export & Clear --
    col_exp, col_clr = st.columns(2)
    with col_exp:
        if st.session_state.history:
            lines = [f"# Pro RAG Chat Export\n_Session: {st.session_state.active_session}_\n_Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n_Model: {selected_model}_\n"]
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
            st.session_state.total_tokens = 0
            st.session_state.total_time = 0.0
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
    <div class="session-badge">💬 {st.session_state.active_session}</div>
</div>
""", unsafe_allow_html=True)

# Analytics Dashboard
if vector_db is not None:
    stats = get_index_stats(vector_db)
    avg_time = (st.session_state.total_time / st.session_state.response_count
                if st.session_state.response_count > 0 else 0)
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
        <div class="stat-card">
            <div class="stat-value">{avg_time:.1f}s</div>
            <div class="stat-label">Avg Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{st.session_state.total_tokens}</div>
            <div class="stat-label">Tokens</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if vector_db is None:
    st.warning("No document index found. Upload and ingest documents in the sidebar.")
    st.stop()

# Focus mode indicator
if focus_path:
    st.markdown(
        f'<div class="focus-indicator">🎯 Focus Mode: <strong>{focus_path.name}</strong></div>',
        unsafe_allow_html=True,
    )

# Quick prompt suggestions (only when chat is empty)
if not st.session_state.history:
    st.markdown("#### ⚡ Quick Prompts")
    qp_cols = st.columns(4)
    quick_prompts = [
        ("📋 Summarize", "Provide a detailed summary of the document."),
        ("🔑 Key Points", "What are the key points and main takeaways?"),
        ("❓ What is this?", "What is this document about? Give an overview."),
        ("👤 Author Info", "Who is the author and what are their credentials?"),
        ("📊 Main Topics", "List all the main topics covered in this document."),
        ("💡 Key Insights", "What are the most interesting insights from this document?"),
        ("📖 Chapter List", "List all chapters or sections in this document."),
        ("🎯 Conclusions", "What are the main conclusions or recommendations?"),
    ]
    for i, (label, prompt_text) in enumerate(quick_prompts):
        col = qp_cols[i % 4]
        with col:
            if st.button(label, use_container_width=True, key=f"qp_{i}"):
                st.session_state.history.append({"role": "user", "content": prompt_text})
                st.rerun()

# Display chat history
for idx, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # Response metrics
            resp_time = msg.get("time", 0)
            resp_tokens = msg.get("tokens", 0)
            if resp_time or resp_tokens:
                st.markdown(
                    f'<div class="response-metrics">'
                    f'<span class="metric-item">⏱️ {resp_time:.1f}s</span>'
                    f'<span class="metric-item">📝 {resp_tokens} tokens</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Text-to-speech button
            if msg["content"]:
                tts_text = msg["content"].replace("'", "\\'").replace("\n", " ")[:500]
                st.markdown(
                    f"""<button onclick="
                        if (window.speechSynthesis.speaking) {{
                            window.speechSynthesis.cancel();
                        }} else {{
                            const u = new SpeechSynthesisUtterance('{tts_text}');
                            u.rate = 1.0;
                            window.speechSynthesis.speak(u);
                        }}
                    " style="
                        background: none; border: 1px solid #555; border-radius: 6px;
                        padding: 4px 10px; color: #aaa; cursor: pointer; font-size: 0.75rem;
                        transition: all 0.2s;
                    " onmouseover="this.style.borderColor='{theme['accent']}'; this.style.color='{theme['accent']}';"
                       onmouseout="this.style.borderColor='#555'; this.style.color='#aaa';">
                        🔊 Read Aloud
                    </button>""",
                    unsafe_allow_html=True,
                )

        if msg.get("docs"):
            _render_sources(msg["docs"])

# Handle new input
if prompt := st.chat_input("Ask about your documents…"):
    # Avoid double-adding if triggered by quick prompts
    if not st.session_state.history or st.session_state.history[-1].get("content") != prompt:
        st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        token_count = 0

        start_time = time.time()

        stream, docs = get_rag_stream_with_scores(
            query=prompt,
            db=vector_db,
            llm=llm,
            top_k=top_k,
            filter_path=focus_path,
            chat_history=st.session_state.history[:-1],
            system_prompt=st.session_state.system_prompt,
        )

        for chunk in stream:
            content = getattr(chunk, "content", str(chunk))
            full_response += content
            token_count += 1
            placeholder.markdown(full_response + "▌")

        elapsed = time.time() - start_time
        placeholder.markdown(full_response)

        # Show metrics
        st.markdown(
            f'<div class="response-metrics">'
            f'<span class="metric-item">⏱️ {elapsed:.1f}s</span>'
            f'<span class="metric-item">📝 {token_count} tokens</span>'
            f'<span class="metric-item">⚡ {token_count/elapsed:.1f} tok/s</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Update session state
        st.session_state.response_count += 1
        st.session_state.total_tokens += token_count
        st.session_state.total_time += elapsed

        st.session_state.history.append({
            "role": "assistant",
            "content": full_response,
            "docs": docs,
            "time": elapsed,
            "tokens": token_count,
        })

        # TTS button for new response
        tts_text = full_response.replace("'", "\\'").replace("\n", " ")[:500]
        st.markdown(
            f"""<button onclick="
                if (window.speechSynthesis.speaking) {{
                    window.speechSynthesis.cancel();
                }} else {{
                    const u = new SpeechSynthesisUtterance('{tts_text}');
                    u.rate = 1.0;
                    window.speechSynthesis.speak(u);
                }}
            " style="
                background: none; border: 1px solid #555; border-radius: 6px;
                padding: 4px 10px; color: #aaa; cursor: pointer; font-size: 0.75rem;
                transition: all 0.2s;
            " onmouseover="this.style.borderColor='{theme['accent']}'; this.style.color='{theme['accent']}';"
               onmouseout="this.style.borderColor='#555'; this.style.color='#aaa';">
                🔊 Read Aloud
            </button>""",
            unsafe_allow_html=True,
        )

        _render_sources(docs)
