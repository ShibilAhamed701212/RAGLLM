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
    get_index_stats, semantic_search,
)
from src.ingestion import ingest_all, ingest_url
from src.core import (
    get_retriever, get_rag_stream_with_scores,
    DEFAULT_SYSTEM_PROMPT, PERSONAS, generate_followups,
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

# ── Multi-language ────────────────────────────────────────────────────────────

LANGUAGES = {
    "🇬🇧 English": "",
    "🇪🇸 Spanish": "Respond entirely in Spanish.",
    "🇫🇷 French": "Respond entirely in French.",
    "🇩🇪 German": "Respond entirely in German.",
    "🇯🇵 Japanese": "Respond entirely in Japanese.",
    "🇨🇳 Chinese": "Respond entirely in Simplified Chinese.",
    "🇮🇳 Hindi": "Respond entirely in Hindi.",
    "🇸🇦 Arabic": "Respond entirely in Arabic.",
    "🇧🇷 Portuguese": "Respond entirely in Portuguese.",
    "🇰🇷 Korean": "Respond entirely in Korean.",
}


# ── Session state defaults ────────────────────────────────────────────────────

DEFAULTS = {
    "history": [],
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "response_count": 0,
    "saved_sessions": {},
    "active_session": "Default",
    "theme": "Midnight Purple",
    "total_tokens": 0,
    "total_time": 0.0,
    "persona": "📚 Default",
    "language": "🇬🇧 English",
    "followups": [],
    "show_shortcuts": False,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

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
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

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
    .model-badge {{
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.15); backdrop-filter: blur(10px);
        padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem;
        color: white; margin-top: 0.5rem; border: 1px solid rgba(255,255,255,0.2);
    }}
    .stats-row {{ display: flex; gap: 0.8rem; margin-bottom: 1rem; }}
    .stat-card {{
        flex: 1;
        background: {theme['card_bg']};
        padding: 1rem; border-radius: 10px; text-align: center;
        border: 1px solid {theme['border']};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stat-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
    .stat-card .stat-value {{
        font-size: 1.8rem; font-weight: 700;
        background: {theme['gradient']};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .stat-card .stat-label {{
        font-size: 0.75rem; color: #888;
        text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem;
    }}
    .score-badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: 6px; }}
    .score-high {{ background: #1a472a; color: #4ade80; }}
    .score-mid  {{ background: #422006; color: #fbbf24; }}
    .score-low  {{ background: #450a0a; color: #f87171; }}
    [data-testid="stSidebar"] > div:first-child {{ background: {theme['sidebar_bg']}; }}
    .stChatMessage {{ animation: fadeIn 0.3s ease-in; }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .focus-indicator {{
        background: linear-gradient(135deg, #f59e0b22, #f59e0b11);
        border: 1px solid #f59e0b44; border-radius: 8px;
        padding: 0.5rem 1rem; font-size: 0.85rem; color: #f59e0b; margin-bottom: 0.5rem;
    }}
    .response-metrics {{
        display: flex; gap: 1rem; padding: 0.4rem 0; font-size: 0.75rem; color: #666;
    }}
    .metric-item {{ display: flex; align-items: center; gap: 4px; }}
    .session-badge {{
        background: {theme['card_bg']}; border: 1px solid {theme['border']};
        border-radius: 8px; padding: 0.3rem 0.7rem; font-size: 0.8rem;
        color: {theme['accent']}; display: inline-block; margin-top: 0.3rem;
    }}
    .followup-btn {{
        display: inline-block; padding: 0.35rem 0.85rem; margin: 0.2rem;
        border-radius: 20px; background: {theme['card_bg']};
        border: 1px solid {theme['border']}; color: {theme['accent']};
        font-size: 0.8rem; cursor: pointer; transition: all 0.2s ease;
        text-decoration: none;
    }}
    .followup-btn:hover {{ background: {theme['gradient']}; color: white; border-color: transparent; }}
    .search-result {{
        background: {theme['card_bg']}; border: 1px solid {theme['border']};
        border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem;
        transition: transform 0.2s ease;
    }}
    .search-result:hover {{ transform: translateY(-2px); }}
    .search-result-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }}
    .copy-btn {{
        background: none; border: 1px solid #555; border-radius: 6px;
        padding: 4px 10px; color: #aaa; cursor: pointer; font-size: 0.75rem;
        transition: all 0.2s;
    }}
    .copy-btn:hover {{ border-color: {theme['accent']}; color: {theme['accent']}; }}
    .action-row {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.3rem; }}
    .shortcut-modal {{
        background: {theme['card_bg']}; border: 1px solid {theme['border']};
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }}
    .shortcut-modal h3 {{ color: {theme['accent']}; margin: 0 0 1rem 0; }}
    .shortcut-row {{ display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
    .shortcut-key {{
        background: rgba(255,255,255,0.08); padding: 2px 8px; border-radius: 4px;
        font-family: monospace; font-size: 0.85rem; color: {theme['accent']};
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
    with st.expander(f"{label} ({len(docs)} chunks)"):
        for doc in docs:
            source = Path(doc.metadata.get("source", "Unknown")).name
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("score")
            if score is not None:
                cls = "score-high" if score >= 0.7 else ("score-mid" if score >= 0.4 else "score-low")
                st.markdown(
                    f'**{source}** (Page {page}) <span class="score-badge {cls}">{score:.0%}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.write(f"- **{source}** (Page {page})")
            content = doc.page_content
            st.caption(content[:300] + "…" if len(content) > 300 else content)


def _build_effective_prompt():
    """Build the effective system prompt from persona + language + custom edits."""
    base = PERSONAS.get(st.session_state.persona, DEFAULT_SYSTEM_PROMPT)
    lang_instruction = LANGUAGES.get(st.session_state.language, "")
    if lang_instruction:
        base += f"\n\nIMPORTANT: {lang_instruction}"
    return base


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
        selected_doc = st.selectbox(
            "Select a document to chat with:",
            ["All Documents"] + file_names,
            help="Narrow the AI's focus to a single document.",
        )
        if selected_doc != "All Documents":
            focus_path = DATA_DIR / selected_doc
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
        st.success("System reset!")
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

    # -- AI Persona --
    st.subheader("🎭 AI Persona")
    persona_choice = st.selectbox(
        "Response style",
        list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(st.session_state.persona),
        label_visibility="collapsed",
        help="Changes how the AI frames its responses — academic, creative, concise, etc.",
    )
    if persona_choice != st.session_state.persona:
        st.session_state.persona = persona_choice
        st.session_state.system_prompt = PERSONAS[persona_choice]

    st.divider()

    # -- Language --
    st.subheader("🌍 Response Language")
    lang_choice = st.selectbox(
        "Language",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.language),
        label_visibility="collapsed",
    )
    if lang_choice != st.session_state.language:
        st.session_state.language = lang_choice

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
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save", use_container_width=True, key="save_prompt"):
                st.session_state.system_prompt = custom_prompt
                st.success("Saved!")
        with c2:
            if st.button("🔄 Reset", use_container_width=True, key="reset_prompt"):
                st.session_state.system_prompt = PERSONAS[st.session_state.persona]
                st.rerun()

    st.divider()

    # -- Chat Sessions --
    st.subheader("💬 Chat Sessions")
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
            if st.session_state.history and st.session_state.active_session:
                st.session_state.saved_sessions[st.session_state.active_session] = list(st.session_state.history)
                _save_sessions()
            st.session_state.history = []
            st.session_state.active_session = f"Chat {len(st.session_state.saved_sessions) + 1}"
            st.session_state.response_count = 0
            st.session_state.followups = []
            st.rerun()

    if st.session_state.saved_sessions:
        load_session = st.selectbox("Load session", ["—"] + list(st.session_state.saved_sessions.keys()))
        if load_session != "—" and st.button("📂 Load", use_container_width=True):
            st.session_state.history = list(st.session_state.saved_sessions[load_session])
            st.session_state.active_session = load_session
            st.session_state.followups = []
            st.rerun()

    st.divider()

    # -- Export & Clear --
    col_exp, col_clr = st.columns(2)
    with col_exp:
        if st.session_state.history:
            lines = [
                f"# Pro RAG Chat Export",
                f"_Session: {st.session_state.active_session}_",
                f"_Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
                f"_Model: {selected_model}_",
                f"_Persona: {st.session_state.persona}_\n",
            ]
            for msg in st.session_state.history:
                role = "**You**" if msg["role"] == "user" else "**Assistant**"
                lines.append(f"{role}: {msg['content']}\n")
            st.download_button(
                "📄 Export",
                data="\n".join(lines),
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
            st.session_state.followups = []
            st.rerun()

    st.divider()

    # -- Keyboard shortcuts --
    if st.button("⌨️ Keyboard Shortcuts", use_container_width=True):
        st.session_state.show_shortcuts = not st.session_state.show_shortcuts


# ── Initialise resources ──────────────────────────────────────────────────────

embeddings = _cached_embeddings()
vector_db = _cached_vector_db(embeddings)
llm = _cached_llm(temperature, selected_model)

# ── Main area — Tabs ──────────────────────────────────────────────────────────

tab_chat, tab_search, tab_summary = st.tabs(["💬 Chat", "🔍 Search", "📑 Summaries"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 Pro RAG Chatbot</h1>
        <p>Chat with your documents locally and privately.</p>
        <div class="model-badge">🧠 {selected_model}</div>
        <div class="session-badge">💬 {st.session_state.active_session} &nbsp;|&nbsp; {st.session_state.persona} &nbsp;|&nbsp; {st.session_state.language}</div>
    </div>
    """, unsafe_allow_html=True)

    # Keyboard shortcuts modal
    if st.session_state.show_shortcuts:
        st.markdown(f"""
        <div class="shortcut-modal">
            <h3>⌨️ Keyboard Shortcuts</h3>
            <div class="shortcut-row"><span>Focus chat input</span><span class="shortcut-key">/</span></div>
            <div class="shortcut-row"><span>Send message</span><span class="shortcut-key">Enter</span></div>
            <div class="shortcut-row"><span>New line in message</span><span class="shortcut-key">Shift + Enter</span></div>
            <div class="shortcut-row"><span>Toggle sidebar</span><span class="shortcut-key">Ctrl + [</span></div>
            <div class="shortcut-row"><span>Toggle wide mode</span><span class="shortcut-key">Settings → Wide</span></div>
            <div class="shortcut-row"><span>Rerun app</span><span class="shortcut-key">R</span></div>
            <div class="shortcut-row"><span>Clear cache</span><span class="shortcut-key">C</span></div>
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
            with qp_cols[i % 4]:
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
                    tps = resp_tokens / resp_time if resp_time > 0 else 0
                    st.markdown(
                        f'<div class="response-metrics">'
                        f'<span class="metric-item">⏱️ {resp_time:.1f}s</span>'
                        f'<span class="metric-item">📝 {resp_tokens} tokens</span>'
                        f'<span class="metric-item">⚡ {tps:.1f} tok/s</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Action buttons row
                content_safe = msg["content"].replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")[:1000]
                tts_safe = msg["content"].replace("'", "\\'").replace("\n", " ")[:500]
                st.markdown(
                    f"""<div class="action-row">
                        <button class="copy-btn" onclick="navigator.clipboard.writeText('{content_safe}').then(()=>this.textContent='✅ Copied!')">
                            📋 Copy
                        </button>
                        <button class="copy-btn" onclick="
                            if(window.speechSynthesis.speaking){{window.speechSynthesis.cancel();this.textContent='🔊 Read Aloud';}}
                            else{{const u=new SpeechSynthesisUtterance('{tts_safe}');u.rate=1.0;window.speechSynthesis.speak(u);this.textContent='⏹️ Stop';}}
                        ">🔊 Read Aloud</button>
                    </div>""",
                    unsafe_allow_html=True,
                )

            if msg.get("docs"):
                _render_sources(msg["docs"])

    # Suggested follow-up questions
    if st.session_state.followups:
        st.markdown("#### 🔗 Suggested Follow-ups")
        fu_cols = st.columns(min(len(st.session_state.followups), 3))
        for i, fu_q in enumerate(st.session_state.followups):
            with fu_cols[i % 3]:
                if st.button(fu_q, use_container_width=True, key=f"fu_{i}"):
                    st.session_state.history.append({"role": "user", "content": fu_q})
                    st.session_state.followups = []
                    st.rerun()

    # Regenerate button
    if (st.session_state.history
            and len(st.session_state.history) >= 2
            and st.session_state.history[-1]["role"] == "assistant"):
        if st.button("🔄 Regenerate Last Response", use_container_width=False):
            # Remove last assistant message, keep the user query
            st.session_state.history.pop()
            st.session_state.followups = []
            st.rerun()

    # Handle new input
    if prompt := st.chat_input("Ask about your documents…"):
        if not st.session_state.history or st.session_state.history[-1].get("content") != prompt:
            st.session_state.history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            token_count = 0
            start_time = time.time()

            effective_prompt = _build_effective_prompt()

            stream, docs = get_rag_stream_with_scores(
                query=prompt,
                db=vector_db,
                llm=llm,
                top_k=top_k,
                filter_path=focus_path,
                chat_history=st.session_state.history[:-1],
                system_prompt=effective_prompt,
            )

            for chunk in stream:
                content = getattr(chunk, "content", str(chunk))
                full_response += content
                token_count += 1
                placeholder.markdown(full_response + "▌")

            elapsed = time.time() - start_time
            placeholder.markdown(full_response)

            # Show metrics
            tps = token_count / elapsed if elapsed > 0 else 0
            st.markdown(
                f'<div class="response-metrics">'
                f'<span class="metric-item">⏱️ {elapsed:.1f}s</span>'
                f'<span class="metric-item">📝 {token_count} tokens</span>'
                f'<span class="metric-item">⚡ {tps:.1f} tok/s</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Action buttons
            content_safe = full_response.replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")[:1000]
            tts_safe = full_response.replace("'", "\\'").replace("\n", " ")[:500]
            st.markdown(
                f"""<div class="action-row">
                    <button class="copy-btn" onclick="navigator.clipboard.writeText('{content_safe}').then(()=>this.textContent='✅ Copied!')">
                        📋 Copy
                    </button>
                    <button class="copy-btn" onclick="
                        if(window.speechSynthesis.speaking){{window.speechSynthesis.cancel();this.textContent='🔊 Read Aloud';}}
                        else{{const u=new SpeechSynthesisUtterance('{tts_safe}');u.rate=1.0;window.speechSynthesis.speak(u);this.textContent='⏹️ Stop';}}
                    ">🔊 Read Aloud</button>
                </div>""",
                unsafe_allow_html=True,
            )

            # Update stats
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

            _render_sources(docs)

            # Generate follow-up suggestions (async-ish)
            with st.spinner("Generating follow-up suggestions…"):
                followups = generate_followups(prompt, full_response, llm)
                st.session_state.followups = followups


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SEMANTIC SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.markdown(f"""
    <div class="main-header">
        <h1>🔍 Semantic Search</h1>
        <p>Search your documents without asking the AI — find relevant passages directly.</p>
    </div>
    """, unsafe_allow_html=True)

    if vector_db is None:
        st.warning("No document index. Upload and ingest documents first.")
    else:
        search_query = st.text_input("Search query", placeholder="Enter a topic or phrase…", key="search_q")
        search_k = st.slider("Number of results", 1, 20, 10, key="search_k")

        if search_query:
            results = semantic_search(vector_db, search_query, top_k=search_k, filter_path=focus_path)

            if results:
                st.markdown(f"**{len(results)}** results for *\"{search_query}\"*")
                for i, r in enumerate(results):
                    score = r["score"]
                    cls = "score-high" if score >= 0.7 else ("score-mid" if score >= 0.4 else "score-low")
                    st.markdown(
                        f"""<div class="search-result">
                            <div class="search-result-header">
                                <span><strong>{r['source']}</strong> · Page {r['page']}</span>
                                <span class="score-badge {cls}">{score:.0%} match</span>
                            </div>
                            <p style="color:#ccc; font-size:0.9rem; margin:0;">{r['content'][:500]}</p>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No results found.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: DOCUMENT SUMMARIES
# ═══════════════════════════════════════════════════════════════════════════════

with tab_summary:
    st.markdown(f"""
    <div class="main-header">
        <h1>📑 Document Summaries</h1>
        <p>Generate AI-powered summaries for each indexed document.</p>
    </div>
    """, unsafe_allow_html=True)

    if vector_db is None:
        st.warning("No document index. Upload and ingest documents first.")
    elif not files:
        st.info("No documents found.")
    else:
        # Initialize summaries storage
        if "doc_summaries" not in st.session_state:
            st.session_state.doc_summaries = {}

        for f in files:
            with st.expander(f"📄 {f.name}  ({f.stat().st_size / 1024:.0f} KB)", expanded=False):
                # Show existing summary
                if f.name in st.session_state.doc_summaries:
                    st.markdown(st.session_state.doc_summaries[f.name])
                    if st.button(f"🔄 Regenerate", key=f"regen_{f.name}"):
                        del st.session_state.doc_summaries[f.name]
                        st.rerun()
                else:
                    st.caption("No summary generated yet.")
                    if st.button(f"✨ Generate Summary", key=f"gen_{f.name}", use_container_width=True):
                        with st.spinner(f"Summarizing {f.name}…"):
                            # Search for this specific document's content
                            doc_results = semantic_search(
                                vector_db, "summary overview main content",
                                top_k=8, filter_path=DATA_DIR / f.name,
                            )
                            if doc_results:
                                context = "\n\n".join([r["content"] for r in doc_results])
                                from langchain_core.messages import SystemMessage, HumanMessage
                                msgs = [
                                    SystemMessage(content="You are a document summarizer. Provide a comprehensive, well-structured summary of the document content provided below. Use markdown formatting with headers and bullet points."),
                                    HumanMessage(content=f"Document: {f.name}\n\nContent:\n{context}\n\nProvide a detailed summary:"),
                                ]
                                result = llm.invoke(msgs)
                                summary = getattr(result, "content", str(result))
                                st.session_state.doc_summaries[f.name] = summary
                                st.rerun()
                            else:
                                st.warning("Could not retrieve content for this document.")

        # Export all summaries
        if st.session_state.doc_summaries:
            st.divider()
            all_summaries = "\n\n---\n\n".join([
                f"# {name}\n\n{summary}"
                for name, summary in st.session_state.doc_summaries.items()
            ])
            st.download_button(
                "📄 Export All Summaries",
                data=all_summaries,
                file_name=f"summaries_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
