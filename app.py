"""
Pro RAG Chatbot — Streamlit interface for chatting with your documents.
"""

from __future__ import annotations

import html
import io
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    DATA_DIR,
    VECTOR_DIR,
    TOP_K,
    DEFAULT_TEMPERATURE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LLM_MODEL,
)
import importlib
import src.utils
importlib.reload(src.utils)

from src.utils import (
    get_embeddings,
    get_llm,
    load_faiss_index,
    list_ollama_models,
    pull_ollama_model,
    get_index_stats,
    semantic_search,
)
from src.ingestion import ingest_all, ingest_url
from src.core import (
    get_rag_stream_with_scores,
    DEFAULT_SYSTEM_PROMPT,
    PERSONAS,
    generate_followups,
)


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pro RAG Chatbot", page_icon="🤖", layout="wide")


# ── Constants ──────────────────────────────────────────────────────────────────

THEMES: dict[str, dict[str, str]] = {
    "Midnight Purple": {
        "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "sidebar_bg": "#0d0d1a",
        "sidebar_bg2": "#12122a",
        "card_bg": "rgba(22, 22, 48, 0.7)",
        "card_solid": "#16163a",
        "accent": "#7c8cf8",
        "accent2": "#a78bfa",
        "accent_glow": "rgba(124, 140, 248, 0.15)",
        "border": "rgba(124, 140, 248, 0.15)",
        "text_primary": "#e8e8f0",
        "text_secondary": "#9191b8",
        "text_muted": "#6060a0",
        "hover_bg": "rgba(124, 140, 248, 0.08)",
        "danger": "#ef4444",
    },
    "Ocean Blue": {
        "gradient": "linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%)",
        "sidebar_bg": "#080e1c",
        "sidebar_bg2": "#0c1428",
        "card_bg": "rgba(14, 26, 50, 0.7)",
        "card_solid": "#0e1a32",
        "accent": "#38bdf8",
        "accent2": "#60a5fa",
        "accent_glow": "rgba(56, 189, 248, 0.15)",
        "border": "rgba(56, 189, 248, 0.15)",
        "text_primary": "#e2ecf5",
        "text_secondary": "#7ea8cc",
        "text_muted": "#4a7da8",
        "hover_bg": "rgba(56, 189, 248, 0.08)",
        "danger": "#ef4444",
    },
    "Emerald": {
        "gradient": "linear-gradient(135deg, #34d399 0%, #10b981 100%)",
        "sidebar_bg": "#06120d",
        "sidebar_bg2": "#0a1f16",
        "card_bg": "rgba(10, 34, 24, 0.7)",
        "card_solid": "#0a2218",
        "accent": "#34d399",
        "accent2": "#6ee7b7",
        "accent_glow": "rgba(52, 211, 153, 0.15)",
        "border": "rgba(52, 211, 153, 0.15)",
        "text_primary": "#e2f5ec",
        "text_secondary": "#7eccaa",
        "text_muted": "#4aa880",
        "hover_bg": "rgba(52, 211, 153, 0.08)",
        "danger": "#ef4444",
    },
    "Sunset": {
        "gradient": "linear-gradient(135deg, #fb923c 0%, #f43f5e 100%)",
        "sidebar_bg": "#140a06",
        "sidebar_bg2": "#1f100a",
        "card_bg": "rgba(34, 16, 10, 0.7)",
        "card_solid": "#22100a",
        "accent": "#fb923c",
        "accent2": "#f97316",
        "accent_glow": "rgba(251, 146, 60, 0.15)",
        "border": "rgba(251, 146, 60, 0.15)",
        "text_primary": "#f5ece2",
        "text_secondary": "#ccaa7e",
        "text_muted": "#a8804a",
        "hover_bg": "rgba(251, 146, 60, 0.08)",
        "danger": "#ef4444",
    },
    "Rose Gold": {
        "gradient": "linear-gradient(135deg, #fb7185 0%, #e879f9 100%)",
        "sidebar_bg": "#140810",
        "sidebar_bg2": "#1f0c1a",
        "card_bg": "rgba(34, 12, 26, 0.7)",
        "card_solid": "#220c1a",
        "accent": "#fb7185",
        "accent2": "#f0abfc",
        "accent_glow": "rgba(251, 113, 133, 0.15)",
        "border": "rgba(251, 113, 133, 0.15)",
        "text_primary": "#f5e2ec",
        "text_secondary": "#cc7eaa",
        "text_muted": "#a84a80",
        "hover_bg": "rgba(251, 113, 133, 0.08)",
        "danger": "#ef4444",
    },
}

LANGUAGES: dict[str, str] = {
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

QUICK_PROMPTS: list[tuple[str, str]] = [
    ("📋 Summarize", "Provide a detailed summary of the document."),
    ("🔑 Key Points", "What are the key points and main takeaways?"),
    ("❓ What is this?", "What is this document about? Give an overview."),
    ("👤 Author Info", "Who is the author and what are their credentials?"),
    ("📊 Main Topics", "List all the main topics covered in this document."),
    ("💡 Key Insights", "What are the most interesting insights from this document?"),
    ("📖 Chapter List", "List all chapters or sections in this document."),
    ("🎯 Conclusions", "What are the main conclusions or recommendations?"),
]

SESSIONS_FILE = VECTOR_DIR.parent / ".chat_sessions.json"


# ── Session state defaults ────────────────────────────────────────────────────

_DEFAULTS: dict = {
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
    "doc_summaries": {},
    "openai_key": "",
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ── Session persistence ───────────────────────────────────────────────────────

def _load_sessions() -> None:
    if not SESSIONS_FILE.exists():
        return
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            st.session_state.saved_sessions = data
    except (json.JSONDecodeError, OSError):
        pass


def _save_sessions() -> None:
    try:
        clean: dict[str, list] = {}
        for name, hist in st.session_state.saved_sessions.items():
            clean[name] = [
                {"role": m["role"], "content": m["content"]}
                for m in hist
                if isinstance(m, dict) and "role" in m and "content" in m
            ]
        SESSIONS_FILE.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    except (OSError, TypeError):
        pass


_load_sessions()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _escape_js(text: str, max_len: int = 1000) -> str:
    return (
        text[:max_len]
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
        .replace("<", "\\x3c")
        .replace(">", "\\x3e")
    )


def _build_effective_prompt() -> str:
    base = PERSONAS.get(st.session_state.persona, DEFAULT_SYSTEM_PROMPT)
    lang_instruction = LANGUAGES.get(st.session_state.language, "")
    if lang_instruction:
        base += f"\n\nIMPORTANT: {lang_instruction}"
    return base


def _render_sources(docs: list, label: str = "📎 Sources") -> None:
    with st.expander(f"{label} ({len(docs)} chunks)"):
        for doc in docs:
            source = Path(doc.metadata.get("source", "Unknown")).name
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("score")
            if score is not None:
                cls = "score-high" if score >= 0.7 else ("score-mid" if score >= 0.4 else "score-low")
                st.markdown(
                    f'**{html.escape(source)}** (p.{page}) '
                    f'<span class="score-badge {cls}">{score:.0%}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.write(f"- **{source}** (p.{page})")
            content = doc.page_content
            st.caption(content[:300] + "…" if len(content) > 300 else content)



def _inject_tts_listener():
    """Injects a global event listener to handle TTS clicks, bypassing React sanitization."""
    import streamlit.components.v1 as components
    js = """
    <script>
        (function() {
            try {
                // Target the main window (parent of the iframe)
                const win = window.parent;
                const doc = win.document;
                
                // Identify our listener with a unique ID to prevent duplicates
                if (!doc.getElementById('tts-listener-v2')) {
                    const marker = doc.createElement('div');
                    marker.id = 'tts-listener-v2';
                    marker.style.display = 'none';
                    doc.body.appendChild(marker);

                    console.log("[TTS] Injecting global listener...");

                    doc.body.addEventListener('click', function(e) {
                        // Find the closest button with data-tts attribute
                        const btn = e.target.closest('[data-tts]');
                        if (!btn) return;
                        
                        // Prevent default behavior (navigation, etc)
                        e.preventDefault();
                        e.stopPropagation();
                        
                        const text = btn.getAttribute('data-tts');
                        if (!text) {
                            console.warn("[TTS] Button clicked but no text found");
                            return;
                        }

                        console.log("[TTS] Exploring text length:", text.length);

                        // Use window.speechSynthesis from the main window context
                        const synth = win.speechSynthesis;

                        if (synth.speaking) {
                            console.log("[TTS] Stopping speech");
                            synth.cancel();
                            btn.textContent = '🔊 Read Aloud';
                        } else {
                            console.log("[TTS] Starting speech");
                            synth.cancel(); // Safety cancel
                            
                            const u = new win.SpeechSynthesisUtterance(text);
                            u.lang = 'en-US'; 
                            u.rate = 1.0;
                            u.pitch = 1.0;
                            
                            u.onstart = () => { console.log("[TTS] Started"); btn.textContent = '⏹️ Stop'; };
                            u.onend = () => { console.log("[TTS] Finished"); btn.textContent = '🔊 Read Aloud'; };
                            u.onerror = (err) => { 
                                console.error('[TTS] Error:', err); 
                                btn.textContent = '❌ Error'; 
                                setTimeout(() => btn.textContent = '🔊 Read Aloud', 2000);
                            };
                            
                            synth.speak(u);
                        }
                    }, true); // Capture phase to intercept early
                    
                    console.log("[TTS] Listener v2 attached successfully");
                }
            } catch(e) {
                console.error("[TTS] Injection failed:", e);
            }
        })();
    </script>
    """
    components.html(js, height=0, width=0)


def _render_action_buttons(text: str, theme_dict: dict, key_suffix: str = "") -> None:
    # Escape for JS string literal (for Copy button)
    copy_safe = _escape_js(text)
    
    # Escape for HTML attribute (for TTS data attribute)
    tts_text = text[:1500].replace("\n", " ").strip()
    tts_safe_attr = html.escape(tts_text, quote=True)
    
    accent = theme_dict["accent"]
    
    # Note: We removed the inline onclick handler to prevent React Error #231.
    # The click is now handled by the global listener injected by _inject_tts_listener().
    st.markdown(
        f"""<div class="action-row">
            <button class="action-btn" onclick="navigator.clipboard.writeText('{copy_safe}').then(()=>this.textContent='✅ Copied!')">📋 Copy</button>
            <button class="action-btn tts-btn" data-tts="{tts_safe_attr}">🔊 Read Aloud</button>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_metrics(resp_time: float, resp_tokens: int) -> None:
    if not resp_time and not resp_tokens:
        return
    tps = resp_tokens / resp_time if resp_time > 0 else 0
    st.markdown(
        f'<div class="perf-metrics">'
        f'<span class="perf-chip">⏱️ {resp_time:.1f}s</span>'
        f'<span class="perf-chip">📝 {resp_tokens} tok</span>'
        f'<span class="perf-chip">⚡ {tps:.1f} t/s</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _reset_chat_state() -> None:
    st.session_state.history = []
    st.session_state.response_count = 0
    st.session_state.total_tokens = 0
    st.session_state.total_time = 0.0
    st.session_state.followups = []


def _generate_chat_pdf(session_name: str, history: list, model: str = "") -> bytes:
    """Generate a professional PDF export of the chat history."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Header ──
    pdf.set_fill_color(88, 28, 135)  # Purple
    pdf.rect(0, 0, 210, 35, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_y(8)
    pdf.cell(0, 10, "Pro RAG Intelligence", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, "Immersive Intelligence Engine  |  Chat Export", ln=True, align="C")

    # ── Metadata ──
    pdf.set_y(42)
    pdf.set_text_color(80, 80, 80)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Session: {session_name}    |    Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}    |    Model: {model}", ln=True, align="C")
    pdf.cell(0, 5, f"Messages: {len(history)}    |    AI Responses: {sum(1 for m in history if m['role'] == 'assistant')}", ln=True, align="C")
    pdf.ln(8)

    # ── Separator ──
    pdf.set_draw_color(88, 28, 135)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # ── Messages ──
    for i, msg in enumerate(history):
        is_user = msg["role"] == "user"

        # Role label
        if is_user:
            pdf.set_fill_color(240, 237, 255)
            pdf.set_text_color(88, 28, 135)
            label = "YOU"
        else:
            pdf.set_fill_color(237, 247, 237)
            pdf.set_text_color(22, 101, 52)
            label = "AI ASSISTANT"

        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(28, 5, f"  {label}", fill=True, ln=True)
        pdf.ln(1)

        # Content
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 10)
        content = msg["content"]
        # Clean content for PDF (remove markdown formatting)
        content = content.replace("**", "").replace("*", "").replace("`", "")
        # Encode to latin-1 safe
        content = content.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 5.5, content)

        # Performance metrics for AI messages
        if not is_user and msg.get("time"):
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(130, 130, 130)
            tokens = msg.get("tokens", 0)
            tps = tokens / msg["time"] if msg["time"] > 0 else 0
            pdf.cell(0, 4, f"    {msg['time']:.1f}s  |  {tokens} tokens  |  {tps:.1f} t/s", ln=True)

        pdf.ln(4)

        # Separator between messages
        if i < len(history) - 1:
            pdf.set_draw_color(220, 220, 220)
            pdf.set_line_width(0.2)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(4)

    # ── Footer ──
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, f"Generated by Pro RAG Intelligence  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    return pdf.output()


# ── Premium CSS ────────────────────────────────────────────────────────────────

t = THEMES[st.session_state.theme]

st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">', unsafe_allow_html=True)

st.markdown(f"""
<style>
    /* ── Fonts ──────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

    /* ── Animated Background ─────────────────────────────── */
    .stApp {{
        background: #050510 !important;
        overflow-x: hidden;
    }}
    .bg-mesh {{
        position: fixed;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background:
            radial-gradient(ellipse at 20% 50%, {t['accent']}25 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, {t['accent2']}20 0%, transparent 40%),
            radial-gradient(ellipse at 50% 80%, {t['accent']}15 0%, transparent 45%);
        animation: bgDrift 20s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }}
    @keyframes bgDrift {{
        0%   {{ transform: translate(0, 0) rotate(0deg); }}
        100% {{ transform: translate(-3%, 2%) rotate(3deg); }}
    }}
    .bg-orb {{
        position: fixed;
        top: 10%; right: 5%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, {t['accent']}35 0%, transparent 70%);
        border-radius: 50%;
        filter: blur(80px);
        animation: orb1 15s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }}
    .bg-orb2 {{
        position: fixed;
        bottom: 5%; left: 10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, {t['accent2']}25 0%, transparent 70%);
        border-radius: 50%;
        filter: blur(70px);
        animation: orb2 18s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }}
    @keyframes orb1 {{
        0%   {{ transform: translate(0, 0) scale(1); opacity: 0.7; }}
        50%  {{ transform: translate(-80px, 60px) scale(1.3); opacity: 0.4; }}
        100% {{ transform: translate(40px, -30px) scale(0.9); opacity: 0.6; }}
    }}
    @keyframes orb2 {{
        0%   {{ transform: translate(0, 0) scale(1); opacity: 0.5; }}
        50%  {{ transform: translate(60px, -40px) scale(1.1); opacity: 0.3; }}
        100% {{ transform: translate(-30px, 50px) scale(0.85); opacity: 0.6; }}
    }}
    .block-container {{
        padding-top: 0.8rem !important;
        position: relative;
        z-index: 1;
    }}

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(195deg, {t['sidebar_bg']} 0%, #050510 50%, {t['sidebar_bg2']} 100%) !important;
        border-right: 1px solid {t['border']};
        box-shadow: 4px 0 40px rgba(0,0,0,0.5);
        overflow-x: hidden !important;
    }}
    [data-testid="stSidebar"] {{
        overflow-x: hidden !important;
    }}
    [data-testid="stSidebar"] * {{
        max-width: 100% !important;
        box-sizing: border-box !important;
    }}
    [data-testid="stSidebar"] .block-container,
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        overflow-x: hidden !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        overflow-x: hidden !important;
        word-break: break-word;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {t['text_secondary']};
        overflow-wrap: break-word;
        word-break: break-word;
    }}

    .sb-brand {{
        padding: 0rem 0 1.5rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid {t['border']};
        position: relative;
        overflow: hidden;
    }}
    .sb-brand::after {{
        content: '';
        position: absolute;
        bottom: -1px; left: 0; right: 0;
        height: 1px;
        background: {t['gradient']};
        opacity: 0.5;
    }}
    .sb-brand-name {{
        font-size: 1.5rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        background: {t['gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 100%;
    }}
    .sb-brand-tag {{
        font-size: 0.6rem;
        font-weight: 500;
        color: {t['text_muted']};
        text-transform: uppercase;
        letter-spacing: 2.5px;
        margin-top: 2px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 100%;
    }}

    /* Sidebar sections */
    .sb-label {{
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.6rem;
        font-weight: 700;
        color: {t['accent']};
        text-transform: uppercase;
        letter-spacing: 2.5px;
        margin: 1.4rem 0 0.5rem 0;
    }}
    .sb-label::after {{
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, {t['accent']}40, transparent);
    }}

    /* File list */
    .doc-item {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.45rem 0.65rem;
        margin-bottom: 0.25rem;
        border-radius: 8px;
        background: {t['hover_bg']};
        border: 1px solid transparent;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        font-size: 0.78rem;
        color: {t['text_primary']};
        overflow: hidden;
        max-width: 100%;
    }}
    .doc-item:hover {{
        border-color: {t['accent']}30;
        background: {t['accent_glow']};
        transform: translateX(4px);
        box-shadow: -3px 0 0 {t['accent']};
    }}
    .doc-name {{
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        min-width: 0;
        flex: 1;
    }}
    .doc-icon {{
        width: 30px; height: 30px;
        border-radius: 8px;
        background: linear-gradient(135deg, {t['accent']}20, {t['accent2']}15);
        border: 1px solid {t['accent']}20;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
    }}
    .doc-meta {{
        margin-left: auto;
        font-size: 0.65rem;
        color: {t['text_muted']};
        white-space: nowrap;
        flex-shrink: 0;
    }}

    /* ── Top Bar (popover buttons) ────────────────────────── */
    [data-testid="stPopover"] > button {{
        background: linear-gradient(135deg, {t['hover_bg']}, rgba(255,255,255,0.02)) !important;
        border: 1px solid {t['border']} !important;
        border-radius: 12px !important;
        padding: 0.35rem 0.85rem !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: {t['text_secondary']} !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        min-height: 36px !important;
        backdrop-filter: blur(10px) !important;
        position: relative !important;
        overflow: hidden !important;
    }}
    [data-testid="stPopover"] > button:hover {{
        border-color: {t['accent']}50 !important;
        color: {t['accent']} !important;
        box-shadow: 0 4px 20px {t['accent']}20, 0 0 30px {t['accent']}08 !important;
        transform: translateY(-2px) !important;
    }}
    [data-testid="stPopover"] [data-testid="stPopoverBody"] {{
        background: {t['card_solid']} !important;
        border: 1px solid {t['accent']}25 !important;
        border-radius: 14px !important;
        backdrop-filter: blur(24px) !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 40px {t['accent']}10 !important;
    }}

    /* ── Hero Banner (3D) ────────────────────────────── */
    .hero {{
        background: {t['gradient']};
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
        box-shadow:
            0 20px 60px {t['accent']}20,
            inset 0 -1px 0 rgba(255,255,255,0.1);
        transform: perspective(800px) rotateX(1deg);
        transition: transform 0.5s ease;
    }}
    .hero:hover {{
        transform: perspective(800px) rotateX(0deg);
    }}
    .hero::before {{
        content: '';
        position: absolute;
        top: -100px; right: -80px;
        width: 300px; height: 300px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
        animation: heroFloat 8s ease-in-out infinite;
    }}
    .hero::after {{
        content: '';
        position: absolute;
        bottom: -120px; left: 20%;
        width: 400px; height: 400px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
        animation: heroFloat 12s ease-in-out infinite reverse;
    }}
    @keyframes heroFloat {{
        0%, 100% {{ transform: translate(0, 0); }}
        50%  {{ transform: translate(20px, -15px); }}
    }}
    .hero h1 {{
        color: #fff !important;
        font-size: 2rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 20px rgba(0,0,0,0.3);
    }}
    .hero p {{
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }}
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.12);
        backdrop-filter: blur(12px);
        padding: 0.3rem 0.85rem;
        border-radius: 100px;
        font-size: 0.78rem;
        color: rgba(255,255,255,0.95);
        margin-top: 0.65rem;
        border: 1px solid rgba(255,255,255,0.15);
        font-weight: 500;
        position: relative;
        z-index: 1;
    }}

    /* ── KPI Cards (3D float) ───────────────────────── */
    .kpi-row {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.2rem;
        perspective: 1000px;
    }}
    @media (max-width: 768px) {{
        .kpi-row {{ grid-template-columns: repeat(3, 1fr); }}
    }}
    .kpi {{
        background: {t['card_bg']};
        backdrop-filter: blur(16px);
        border: 1px solid {t['border']};
        border-radius: 14px;
        padding: 1rem 0.8rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        transform-style: preserve-3d;
    }}
    .kpi::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: {t['gradient']};
        opacity: 0;
        transition: opacity 0.25s ease;
    }}
    .kpi:hover {{
        transform: translateY(-8px) perspective(600px) rotateX(3deg) scale(1.02);
        box-shadow:
            0 20px 50px {t['accent']}25,
            0 0 30px {t['accent']}10;
        border-color: {t['accent']}50;
    }}
    .kpi:hover::before {{ opacity: 1; }}
    .kpi-val {{
        font-size: 1.8rem;
        font-weight: 900;
        background: {t['gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }}
    .kpi-lbl {{
        font-size: 0.65rem;
        font-weight: 700;
        color: {t['text_muted']};
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.3rem;
    }}

    /* ── Focus Indicator ─────────────────────────────────── */
    .focus-bar {{
        background: linear-gradient(135deg, rgba(251,191,36,0.08), rgba(251,191,36,0.04));
        border: 1px solid rgba(251,191,36,0.25);
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-size: 0.82rem;
        color: #fbbf24;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* ── Quick Prompts ───────────────────────────────────── */
    .qp-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.5rem;
        margin-bottom: 1rem;
    }}
    @media (max-width: 768px) {{
        .qp-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}

    /* ── Chat Bubbles (3D slide) ───────────────────── */
    .stChatMessage {{
        animation: msgSlide 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        border-radius: 18px !important;
        backdrop-filter: blur(16px) !important;
    }}
    @keyframes msgSlide {{
        from {{ opacity: 0; transform: translateY(25px) scale(0.96); }}
        to   {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}

    /* ── Perf Metrics ────────────────────────────────────── */
    .perf-metrics {{
        display: flex;
        gap: 0.5rem;
        padding: 0.35rem 0;
        flex-wrap: wrap;
    }}
    .perf-chip {{
        background: linear-gradient(135deg, {t['hover_bg']}, rgba(255,255,255,0.02));
        border: 1px solid {t['border']};
        padding: 3px 12px;
        border-radius: 100px;
        font-size: 0.68rem;
        color: {t['text_secondary']};
        font-weight: 600;
        backdrop-filter: blur(8px);
    }}

    /* ── Action Buttons ──────────────────────────────────── */
    .action-row {{
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-top: 0.25rem;
    }}
    .action-btn {{
        background: transparent;
        border: 1px solid {t['border']};
        border-radius: 10px;
        padding: 5px 14px;
        color: {t['text_secondary']};
        cursor: pointer;
        font-size: 0.72rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}
    .action-btn:hover {{
        border-color: {t['accent']};
        color: {t['accent']};
        background: {t['accent_glow']};
        transform: translateY(-2px);
        box-shadow: 0 4px 15px {t['accent']}15;
    }}

    /* ── Score Badges ────────────────────────────────────── */
    .score-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 6px;
    }}
    .score-high {{ background: rgba(74,222,128,0.14); color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }}
    .score-mid  {{ background: rgba(251,191,36,0.14); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }}
    .score-low  {{ background: rgba(248,113,113,0.14); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }}

    /* ── Search Results (3D cards) ────────────────── */
    .sr-card {{
        background: {t['card_bg']};
        backdrop-filter: blur(16px);
        border: 1px solid {t['border']};
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }}
    .sr-card::before {{
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: {t['gradient']};
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    .sr-card:hover {{
        transform: translateY(-3px) translateX(3px);
        box-shadow: 0 12px 40px {t['accent']}15;
        border-color: {t['accent']}35;
    }}
    .sr-card:hover::before {{ opacity: 1; }}
    .sr-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }}
    .sr-header strong {{ color: {t['text_primary']}; }}
    .sr-body {{
        color: {t['text_secondary']};
        font-size: 0.84rem;
        line-height: 1.6;
        margin: 0;
    }}

    /* ── Shortcut Modal ──────────────────────────── */
    .kb-modal {{
        background: {t['card_bg']};
        backdrop-filter: blur(20px);
        border: 1px solid {t['border']};
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }}
    .kb-modal h3 {{
        color: {t['accent']};
        margin: 0 0 0.8rem 0;
        font-size: 1rem;
        font-weight: 800;
    }}
    .kb-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid {t['border']};
        font-size: 0.82rem;
        color: {t['text_secondary']};
    }}
    .kb-key {{
        background: linear-gradient(135deg, {t['hover_bg']}, rgba(255,255,255,0.03));
        border: 1px solid {t['border']};
        padding: 3px 12px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: {t['accent']};
        font-weight: 500;
    }}

    /* ── Tabs (holographic bar) ───────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {t['card_bg']};
        border-radius: 14px;
        padding: 5px;
        border: 1px solid {t['border']};
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        color: {t['text_secondary']};
        font-weight: 600;
        padding: 0.45rem 1.3rem;
        transition: all 0.3s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {t['accent_glow']}, {t['accent']}12) !important;
        color: {t['accent']} !important;
        box-shadow: 0 0 20px {t['accent']}15;
    }}

    /* ── Streamlit Elements 3D ────────────────────── */
    .stButton > button {{
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        border: 1px solid {t['border']} !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px {t['accent']}15 !important;
    }}
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] > div > div {{
        border-radius: 12px !important;
        border: 1px solid {t['border']} !important;
        background: {t['hover_bg']} !important;
        transition: all 0.3s ease !important;
    }}
    [data-testid="stTextInput"] input:focus {{
        border-color: {t['accent']}60 !important;
        box-shadow: 0 0 20px {t['accent']}10 !important;
    }}
    [data-testid="stExpander"] {{
        border: 1px solid {t['border']} !important;
        border-radius: 14px !important;
        background: {t['card_bg']} !important;
        backdrop-filter: blur(12px) !important;
    }}

    /* ── Chat Input 3D ───────────────────────────────── */
    [data-testid="stChatInput"] {{
        border-radius: 16px !important;
        border: 1px solid {t['border']} !important;
        background: {t['card_bg']} !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {t['accent']}60 !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 0 30px {t['accent']}15 !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: {t['text_primary']} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* ── Chat Message Containers ──────────────────────── */
    [data-testid="stChatMessage"] {{
        background: {t['card_bg']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 16px !important;
        backdrop-filter: blur(12px) !important;
        padding: 1rem !important;
        margin-bottom: 0.6rem !important;
        transition: all 0.3s ease !important;
    }}
    [data-testid="stChatMessage"]:hover {{
        border-color: {t['accent']}25 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
    }}

    /* ── Sources Expander 3D ──────────────────────────── */
    [data-testid="stExpander"] details {{
        border: 1px solid {t['border']} !important;
        border-radius: 14px !important;
        background: {t['card_bg']} !important;
        overflow: hidden;
        transition: all 0.3s ease !important;
    }}
    [data-testid="stExpander"] summary:hover {{
        color: {t['accent']} !important;
    }}

    /* ── File Uploader 3D ────────────────────────────── */
    [data-testid="stFileUploader"] section {{
        border: 2px dashed {t['border']} !important;
        border-radius: 14px !important;
        background: {t['hover_bg']} !important;
        transition: all 0.3s ease !important;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: {t['accent']}50 !important;
        box-shadow: 0 0 20px {t['accent']}08 !important;
    }}

    /* ── Slider 3D ───────────────────────────────────── */
    [data-testid="stSlider"] [role="slider"] {{
        background: {t['accent']} !important;
        box-shadow: 0 0 10px {t['accent']}40 !important;
    }}

    /* ── Scrollbar ────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: {t['accent']}30;
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {t['accent']}50; }}

    /* ── Streamlit Alerts 3D ──────────────────────── */
    [data-testid="stAlert"] {{
        border-radius: 14px !important;
        backdrop-filter: blur(12px) !important;
        border-width: 1px !important;
    }}

    /* ── Spinner ────────────────────────────────── */
    .stSpinner > div {{
        border-top-color: {t['accent']} !important;
    }}

    /* ── Divider ────────────────────────────────── */
    [data-testid="stSidebar"] hr {{
        border-color: {t['border']} !important;
        opacity: 0.5;
    }}

    /* ── Markdown container ──────────────────────── */
    .stMarkdown {{
        color: {t['text_primary']};
    }}
    .stMarkdown h4 {{
        color: {t['text_primary']} !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}

    /* ═══════════════════════════════════════════════════════
       RESPONSIVE — TABLETS (≤ 1024px)
       ═══════════════════════════════════════════════════════ */
    @media (max-width: 1024px) {{
        .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
        .hero {{ padding: 1.5rem 1.8rem; border-radius: 16px; transform: none; }}
        .hero:hover {{ transform: none; }}
        .hero h1 {{ font-size: 1.6rem; }}
        .kpi-row {{ grid-template-columns: repeat(3, 1fr); gap: 0.5rem; }}
        .kpi {{ padding: 0.8rem 0.5rem; }}
        .kpi-val {{ font-size: 1.4rem; }}
        .sr-card {{ padding: 0.9rem; }}
        .stTabs [data-baseweb="tab"] {{ padding: 0.4rem 0.9rem; font-size: 0.85rem; }}
    }}

    /* ═══════════════════════════════════════════════════════
       RESPONSIVE — SMALL TABLETS & LARGE PHONES (≤ 768px)
       ═══════════════════════════════════════════════════════ */
    @media (max-width: 768px) {{
        .block-container {{ padding-left: 0.5rem !important; padding-right: 0.5rem !important; padding-top: 0.4rem !important; }}
        .bg-mesh, .bg-orb, .bg-orb2 {{ display: none; }}
        .hero {{ padding: 1.2rem; border-radius: 14px; margin-bottom: 0.8rem; transform: none; }}
        .hero:hover {{ transform: none; }}
        .hero h1 {{ font-size: 1.3rem; }}
        .hero p {{ font-size: 0.78rem; }}
        .hero-badge {{ font-size: 0.68rem; padding: 0.25rem 0.7rem; }}
        .hero::before {{ width: 150px; height: 150px; top: -60px; right: -40px; }}
        .hero::after {{ width: 200px; height: 200px; }}
        .kpi-row {{ grid-template-columns: repeat(3, 1fr); gap: 0.4rem; margin-bottom: 0.8rem; }}
        .kpi {{ padding: 0.6rem 0.4rem; border-radius: 12px; }}
        .kpi:hover {{ transform: none; box-shadow: none; }}
        .kpi-val {{ font-size: 1.15rem; }}
        .kpi-lbl {{ font-size: 0.52rem; letter-spacing: 1.5px; }}
        .focus-bar {{ padding: 0.45rem 0.8rem; font-size: 0.75rem; border-radius: 10px; }}
        .qp-grid {{ grid-template-columns: repeat(2, 1fr); }}
        .sr-card {{ padding: 0.8rem; border-radius: 12px; }}
        .sr-card:hover {{ transform: none; }}
        .sr-header {{ flex-direction: column; align-items: flex-start; gap: 0.3rem; }}
        .sr-body {{ font-size: 0.78rem; }}
        .stTabs [data-baseweb="tab-list"] {{ border-radius: 12px; padding: 3px; overflow-x: auto; -webkit-overflow-scrolling: touch; }}
        .stTabs [data-baseweb="tab"] {{ padding: 0.35rem 0.75rem; font-size: 0.8rem; white-space: nowrap; }}
        .perf-chip {{ font-size: 0.62rem; padding: 2px 8px; }}
        .action-btn {{ padding: 6px 14px; font-size: 0.7rem; min-height: 38px; }}
        .kb-modal {{ padding: 1rem; border-radius: 14px; }}
        .kb-row {{ font-size: 0.75rem; }}
        [data-testid="stSidebar"] {{ min-width: 260px !important; max-width: 280px !important; }}
        [data-testid="column"] {{ padding: 0 0.15rem !important; }}
    }}

    /* ═══════════════════════════════════════════════════════
       RESPONSIVE — PHONES (≤ 480px)
       ═══════════════════════════════════════════════════════ */
    @media (max-width: 480px) {{
        .block-container {{ padding-left: 0.3rem !important; padding-right: 0.3rem !important; padding-top: 0.2rem !important; }}
        .hero {{ padding: 0.9rem; border-radius: 12px; margin-bottom: 0.5rem; }}
        .hero h1 {{ font-size: 1.05rem; }}
        .hero p {{ font-size: 0.7rem; }}
        .hero-badge {{ font-size: 0.62rem; padding: 0.2rem 0.5rem; }}
        .hero::before, .hero::after {{ display: none; }}
        .kpi-row {{ grid-template-columns: repeat(2, 1fr); gap: 0.3rem; margin-bottom: 0.5rem; }}
        .kpi {{ padding: 0.5rem 0.3rem; border-radius: 10px; }}
        .kpi-val {{ font-size: 1rem; }}
        .kpi-lbl {{ font-size: 0.48rem; letter-spacing: 1px; }}
        .focus-bar {{ padding: 0.38rem 0.6rem; font-size: 0.68rem; }}
        .sr-card {{ padding: 0.65rem; margin-bottom: 0.5rem; border-radius: 10px; }}
        .sr-body {{ font-size: 0.72rem; line-height: 1.45; }}
        .score-badge {{ font-size: 0.6rem; padding: 1px 7px; }}
        .stTabs [data-baseweb="tab-list"] {{ border-radius: 10px; padding: 2px; }}
        .stTabs [data-baseweb="tab"] {{ padding: 0.28rem 0.55rem; font-size: 0.72rem; border-radius: 8px; }}
        .action-row {{ gap: 0.3rem; }}
        .action-btn {{ padding: 8px 14px; font-size: 0.66rem; min-height: 42px; border-radius: 10px; flex: 1; text-align: center; }}
        .perf-chip {{ font-size: 0.56rem; padding: 2px 7px; }}
        .kb-modal {{ padding: 0.8rem; border-radius: 12px; }}
        .kb-modal h3 {{ font-size: 0.88rem; }}
        .kb-row {{ font-size: 0.68rem; padding: 0.25rem 0; }}
        .kb-key {{ font-size: 0.64rem; padding: 1px 7px; }}
        [data-testid="stSidebar"] {{ min-width: 240px !important; max-width: 260px !important; }}
        .sb-brand-name {{ font-size: 1.2rem; }}
        .sb-brand-tag {{ font-size: 0.52rem; letter-spacing: 2px; }}
        .sb-label {{ font-size: 0.55rem; margin: 0.9rem 0 0.35rem 0; }}
        .doc-item {{ padding: 0.38rem 0.5rem; font-size: 0.72rem; gap: 0.4rem; }}
        .doc-icon {{ width: 24px; height: 24px; font-size: 0.75rem; border-radius: 6px; }}
        .doc-meta {{ font-size: 0.56rem; }}
        [data-testid="column"] {{ padding: 0 0.08rem !important; }}
        .stChatMessage {{ padding: 0.5rem !important; }}
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stTextInput"] > div > div > input {{ min-height: 42px !important; font-size: 0.85rem !important; }}
        .stButton > button {{ min-height: 42px !important; font-size: 0.82rem !important; }}
    }}

    /* ═══════════════════════════════════════════════════════
       TOUCH ENHANCEMENTS
       ═══════════════════════════════════════════════════════ */
    @media (hover: none) and (pointer: coarse) {{
        .kpi:hover, .sr-card:hover {{ transform: none; box-shadow: none; }}
        .hero {{ transform: none; }}
        .hero:hover {{ transform: none; }}
        .doc-item:hover {{ border-color: transparent; background: {t['hover_bg']}; transform: none; box-shadow: none; }}
        .action-btn {{ min-height: 44px; min-width: 44px; }}
        .stButton > button {{ min-height: 44px !important; }}
        .block-container, [data-testid="stSidebar"] > div:first-child {{ -webkit-overflow-scrolling: touch; }}
    }}
</style>
""", unsafe_allow_html=True)

# ── Animated background elements (injected as real HTML) ──────────────────────
st.markdown(
    f'<div class="bg-mesh"></div>'
    f'<div class="bg-orb"></div>'
    f'<div class="bg-orb2"></div>',
    unsafe_allow_html=True,
)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def _cached_embeddings():
    return get_embeddings()


@st.cache_resource
def _cached_llm(temperature: float, model: str, api_key: str | None = None):
    return get_llm(temperature=temperature, model=model, api_key=api_key)


@st.cache_resource
def _cached_vector_db(_embeddings):
    return load_faiss_index(_embeddings)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Brand ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-brand">'
        '<div class="sb-brand-name">🤖 Pro RAG</div>'
        '<div class="sb-brand-tag">Immersive Intelligence Engine</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Documents ────────────────────────────────────────────────────
    st.markdown('<div class="sb-label">📚 Documents</div>', unsafe_allow_html=True)
    files = sorted(DATA_DIR.glob("*.*")) if DATA_DIR.exists() else []
    if files:
        for f in files:
            sz = f.stat().st_size / 1024
            ext = f.suffix.lstrip(".").upper()
            icon = "📕" if ext == "PDF" else ("📝" if ext in ("TXT", "MD") else "📄")
            st.markdown(
                f'<div class="doc-item">'
                f'<div class="doc-icon">{icon}</div>'
                f'<span class="doc-name">{html.escape(f.name)}</span>'
                f'<span class="doc-meta">{ext} · {sz:.0f} KB</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents indexed yet.")

    # ── Focus Mode ───────────────────────────────────────────────────
    st.markdown('<div class="sb-label">🎯 Focus Mode</div>', unsafe_allow_html=True)
    focus_path = None
    if files:
        file_names = [f.name for f in files]
        selected_doc = st.selectbox(
            "Focus",
            ["All Documents"] + file_names,
            help="Lock AI answers to a single document.",
            label_visibility="collapsed",
        )
        if selected_doc != "All Documents":
            focus_path = DATA_DIR / selected_doc
    else:
        st.caption("Upload files to enable.")

    # ── Upload ───────────────────────────────────────────────────────
    st.markdown('<div class="sb-label">📤 Upload</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Files",
        label_visibility="collapsed",
        accept_multiple_files=True,
        type=["pdf", "txt", "md"],
    )
    if st.button("🚀 Ingest & Index", use_container_width=True):
        if uploaded:
            DATA_DIR.mkdir(exist_ok=True)
            for f in uploaded:
                (DATA_DIR / f.name).write_bytes(f.getbuffer())
            with st.status("Indexing…") as status:
                st.write("Chunking & embedding…")
                ingest_all(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                status.update(label="✅ Index ready!", state="complete")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.warning("Upload files first.")

    # ── Web Ingest ───────────────────────────────────────────────────
    st.markdown('<div class="sb-label">🌐 Web Ingest</div>', unsafe_allow_html=True)
    url_input = st.text_input("URL", placeholder="https://…", label_visibility="collapsed")
    if st.button("🔗 Fetch & Index", use_container_width=True):
        if url_input.strip():
            with st.status("Fetching…") as status:
                ok, msg = ingest_url(url_input.strip())
                if ok:
                    st.write(msg)
                    ingest_all(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                    status.update(label="✅ Indexed!", state="complete")
                    st.cache_resource.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    status.update(label=f"❌ {msg}", state="error")
        else:
            st.warning("Paste a URL first.")

    # ── Engine ───────────────────────────────────────────────────────
    st.markdown('<div class="sb-label">⚙️ Engine</div>', unsafe_allow_html=True)
    temperature = st.slider("Creativity", 0.0, 1.0, DEFAULT_TEMPERATURE, help="Higher = creative · Lower = precise")
    top_k = st.slider("Search Depth", 1, 10, TOP_K, help="Chunks retrieved per query")

    # ── System Prompt ────────────────────────────────────────────────
    st.markdown('<div class="sb-label">✏️ System Prompt</div>', unsafe_allow_html=True)
    with st.expander("Edit Behavior"):
        custom_prompt = st.text_area("Prompt", value=st.session_state.system_prompt, height=110, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Apply", use_container_width=True, key="sp_save"):
                st.session_state.system_prompt = custom_prompt
                st.success("Saved!")
        with c2:
            if st.button("↩ Reset", use_container_width=True, key="sp_reset"):
                st.session_state.system_prompt = PERSONAS.get(st.session_state.persona, DEFAULT_SYSTEM_PROMPT)
                st.rerun()

    # ── Cloud API ────────────────────────────────────────────────────
    st.markdown('<div class="sb-label">☁️ Cloud Models</div>', unsafe_allow_html=True)
    with st.expander("API Keys"):
        key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_key,
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed"
        )
        if key_input != st.session_state.openai_key:
            st.session_state.openai_key = key_input
            st.rerun()
        if st.session_state.openai_key:
            st.caption("✅ Key active")

    # ── Model Download ───────────────────────────────────────────────
    st.markdown('<div class="sb-label">📥 Pull Model</div>', unsafe_allow_html=True)
    new_model = st.text_input("Name", placeholder="gemma3:4b", label_visibility="collapsed")
    if st.button("⬇️ Download", use_container_width=True):
        if new_model.strip():
            with st.status(f"Pulling {new_model}…", expanded=True) as status:
                success = pull_ollama_model(new_model.strip())
                if success:
                    status.update(label=f"✅ {new_model} ready!", state="complete")
                    time.sleep(1)
                    st.rerun()
                else:
                    status.update(label=f"❌ Failed", state="error")
        else:
            st.warning("Enter model name.")

    # ── Danger Zone ──────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Factory Reset", use_container_width=True, type="secondary"):
        for d in (DATA_DIR, VECTOR_DIR):
            if d.exists():
                shutil.rmtree(d)
        st.cache_resource.clear()
        _reset_chat_state()
        st.success("Reset complete!")
        time.sleep(1)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TOP BAR — Popover buttons (tap to reveal each control)
# ═══════════════════════════════════════════════════════════════════════════════

ollama_models = list_ollama_models()
model_names = [m["name"] for m in ollama_models]
model_labels = [f"🦙 {m['name']}  ({m['size_gb']} GB)" for m in ollama_models]

if st.session_state.openai_key:
    gpt_models = [
        {"name": "gpt-4o", "label": "☁️ GPT-4o"},
        {"name": "gpt-4-turbo", "label": "☁️ GPT-4 Turbo"},
        {"name": "gpt-3.5-turbo", "label": "☁️ GPT-3.5 Turbo"},
    ]
    model_names.extend([m["name"] for m in gpt_models])
    model_labels.extend([m["label"] for m in gpt_models])

default_idx = model_names.index(LLM_MODEL) if LLM_MODEL in model_names else 0

# Short display names
_model_short = model_names[default_idx]
_persona_short = st.session_state.persona.split(" ", 1)[-1] if " " in st.session_state.persona else st.session_state.persona
_lang_short = st.session_state.language.split(" ", 1)[-1] if " " in st.session_state.language else st.session_state.language

# ── Row of popover buttons ──────────────────────────────────────────────────
pb1, pb2, pb3, pb4, pb5 = st.columns(5)

with pb1:
    with st.popover(f"🧠 {_model_short}", use_container_width=True):
        selected_model_idx = st.selectbox(
            "Select Model",
            range(len(model_labels)),
            index=default_idx,
            format_func=lambda i: model_labels[i],
        )
        selected_model = model_names[selected_model_idx]

with pb2:
    with st.popover(f"🎭 {_persona_short}", use_container_width=True):
        persona_keys = list(PERSONAS.keys())
        persona_choice = st.selectbox(
            "AI Persona",
            persona_keys,
            index=persona_keys.index(st.session_state.persona)
                  if st.session_state.persona in persona_keys else 0,
        )
        if persona_choice != st.session_state.persona:
            st.session_state.persona = persona_choice
            st.session_state.system_prompt = PERSONAS[persona_choice]

with pb3:
    with st.popover(f"🌍 {_lang_short}", use_container_width=True):
        lang_keys = list(LANGUAGES.keys())
        lang_choice = st.selectbox(
            "Language",
            lang_keys,
            index=lang_keys.index(st.session_state.language)
                  if st.session_state.language in lang_keys else 0,
        )
        if lang_choice != st.session_state.language:
            st.session_state.language = lang_choice

with pb4:
    with st.popover(f"🎨 {st.session_state.theme[:8]}", use_container_width=True):
        theme_keys = list(THEMES.keys())
        theme_choice = st.selectbox(
            "Color Theme",
            theme_keys,
            index=theme_keys.index(st.session_state.theme)
                  if st.session_state.theme in theme_keys else 0,
        )
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

with pb5:
    with st.popover(f"💬 {st.session_state.active_session[:10]}", use_container_width=True):
        session_name = st.text_input(
            "Session Name", value=st.session_state.active_session,
        )
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("💾 Save", use_container_width=True, key="tb_save"):
                if session_name.strip():
                    st.session_state.saved_sessions[session_name] = list(st.session_state.history)
                    st.session_state.active_session = session_name
                    _save_sessions()
                    st.toast("💾 Saved!")
        with sc2:
            if st.button("🆕 New", use_container_width=True, key="tb_new"):
                if st.session_state.history and st.session_state.active_session:
                    st.session_state.saved_sessions[st.session_state.active_session] = list(st.session_state.history)
                    _save_sessions()
                st.session_state.active_session = f"Chat {len(st.session_state.saved_sessions) + 1}"
                _reset_chat_state()
                st.rerun()
        with sc3:
            if st.button("🧹 Clear", use_container_width=True, key="tb_clear"):
                _reset_chat_state()
                st.rerun()

        if st.session_state.saved_sessions:
            st.markdown("---")
            load_session = st.selectbox(
                "Load Session",
                ["—"] + list(st.session_state.saved_sessions.keys()),
            )
            if load_session != "—":
                if st.button("📂 Load", use_container_width=True, key="load_btn"):
                    st.session_state.history = list(st.session_state.saved_sessions[load_session])
                    st.session_state.active_session = load_session
                    st.session_state.followups = []
                    st.rerun()

        # Initialize session state (chat history, etc.)
        if "history" not in st.session_state:
            st.session_state.history = []

        # Inject TTS Listener for global event delegation
        _inject_tts_listener()

    # ── Theme & CSS ─────────────────────────────────────────────────────────────
    if "theme" not in st.session_state:
        st.session_state.theme = "Midnight Purple"
        
    theme_dict = THEMES.get(st.session_state.theme, THEMES["Midnight Purple"])
    
    CSS = f"""
    <style>
    /* Main Background & Font */
    .stApp {{
        background: {theme_dict['gradient']};
        font-family: 'Inter', sans-serif;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {theme_dict['sidebar_bg']};
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* Glassmorphism Cards */
    .glass-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Chat Message Bubbles */
    .user-msg {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px 12px 0 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #fff;
        max-width: 85%;
        margin-left: auto;
    }}
    .ai-msg {{
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px 12px 12px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e0e0e0;
        max-width: 85%;
    }}
    
    /* Action Buttons (Copy/TTS) */
    .action-row {{
        display: flex;
        gap: 8px;
        margin-top: 8px;
        opacity: 0.7;
        transition: opacity 0.2s;
    }}
    .action-row:hover {{ opacity: 1; }}
    
    .action-btn {{
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ccc;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 0.75rem;
        cursor: pointer;
        transition: all 0.2s;
    }}
    .action-btn:hover {{
        background: rgba(255, 255, 255, 0.1);
        border-color: {theme_dict['accent']};
        color: white;
    }}

    /* Metrics Chips */
    .perf-metrics {{
        display: flex;
        gap: 8px;
        font-size: 0.7rem;
        color: #888;
        margin-top: 4px;
    }}
    .perf-chip {{
        background: rgba(0,0,0,0.2);
        padding: 2px 6px;
        border-radius: 4px;
    }}
    
    /* Badge Styles */
    .score-badge {{
        font-size: 0.75em;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 6px;
        font-weight: 600;
    }}
    .score-high {{ background-color: rgba(76, 175, 80, 0.2); color: #81c784; }}
    .score-mid  {{ background-color: rgba(255, 152, 0, 0.2); color: #ffb74d; }}
    .score-low  {{ background-color: rgba(244, 67, 54, 0.2);  color: #e57373; }}

    /* Hide Streamlit default branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """

    # Apply custom CSS
    st.markdown(CSS, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        exp_c1, exp_c2 = st.columns(2)
        
        # Markdown Export
        lines_exp = [
            "# Pro RAG Chat Export",
            f"_Session: {st.session_state.active_session}_",
            f"_Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        ]
        for msg in st.session_state.history:
            role = "**You**" if msg["role"] == "user" else "**AI**"
            lines_exp.append(f"{role}: {msg['content']}\n")
        
        with exp_c1:
            st.download_button(
                "📄 Export MD",
                data="\n".join(lines_exp),
                file_name=f"{st.session_state.active_session}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # PDF Export
        with exp_c2:
            try:
                # Ensure selected_model is available or fallback
                model_name = selected_model if 'selected_model' in locals() else "Unknown Model"
                pdf_bytes = _generate_chat_pdf(
                    st.session_state.active_session,
                    st.session_state.history,
                    model=model_name
                )
                st.download_button(
                    label="📕 Export PDF",
                    data=bytes(pdf_bytes),
                    file_name=f"{st.session_state.active_session}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF Gen Error: {e}")


# ── Initialise resources ──────────────────────────────────────────────────────

try:
    embeddings = _cached_embeddings()
    vector_db = _cached_vector_db(embeddings)
    llm = _cached_llm(temperature, selected_model, api_key=st.session_state.openai_key)
except Exception as e:
    st.error(f"Failed to initialize AI engine: {e}")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_chat, tab_search, tab_summary = st.tabs(["💬  Chat", "🔍  Search", "📑  Summaries"])


# ── TAB 1: CHAT ──────────────────────────────────────────────────────────────

with tab_chat:
    # Hero banner
    st.markdown(
        f'<div class="hero">'
        f'<h1>🤖 Pro RAG Intelligence</h1>'
        f'<p>Explore your documents with AI — private, powerful, precise.</p>'
        f'<div class="hero-badge">🧠 {html.escape(selected_model)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Keyboard shortcuts
    if st.session_state.show_shortcuts:
        st.markdown(
            '<div class="kb-modal">'
            '<h3>⌨️ Keyboard Shortcuts</h3>'
            '<div class="kb-row"><span>Focus chat input</span><span class="kb-key">/</span></div>'
            '<div class="kb-row"><span>Send message</span><span class="kb-key">Enter</span></div>'
            '<div class="kb-row"><span>New line</span><span class="kb-key">Shift+Enter</span></div>'
            '<div class="kb-row"><span>Toggle sidebar</span><span class="kb-key">Ctrl+[</span></div>'
            '<div class="kb-row"><span>Rerun app</span><span class="kb-key">R</span></div>'
            '<div class="kb-row"><span>Clear cache</span><span class="kb-key">C</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # KPI dashboard
    if vector_db is not None:
        stats = get_index_stats(vector_db)
        avg_t = (
            st.session_state.total_time / st.session_state.response_count
            if st.session_state.response_count > 0 else 0
        )
        st.markdown(
            '<div class="kpi-row">'
            f'<div class="kpi"><div class="kpi-val">{stats["unique_sources"]}</div><div class="kpi-lbl">Documents</div></div>'
            f'<div class="kpi"><div class="kpi-val">{stats["total_pages"]}</div><div class="kpi-lbl">Pages</div></div>'
            f'<div class="kpi"><div class="kpi-val">{stats["total_chunks"]}</div><div class="kpi-lbl">Chunks</div></div>'
            f'<div class="kpi"><div class="kpi-val">{st.session_state.response_count}</div><div class="kpi-lbl">Queries</div></div>'
            f'<div class="kpi"><div class="kpi-val">{avg_t:.1f}s</div><div class="kpi-lbl">Avg Time</div></div>'
            f'<div class="kpi"><div class="kpi-val">{st.session_state.total_tokens}</div><div class="kpi-lbl">Tokens</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if vector_db is None:
        st.warning("No document index found. Upload and ingest documents in the sidebar.")
        st.stop()

    # Focus indicator
    if focus_path:
        st.markdown(
            f'<div class="focus-bar">🎯 <strong>Focus Mode:</strong>&nbsp; {html.escape(focus_path.name)}</div>',
            unsafe_allow_html=True,
        )

    # Quick prompts
    if not st.session_state.history:
        st.markdown("#### ⚡ Quick Prompts")
        qp_cols = st.columns(4)
        for i, (label, prompt_text) in enumerate(QUICK_PROMPTS):
            with qp_cols[i % 4]:
                if st.button(label, use_container_width=True, key=f"qp_{i}"):
                    st.session_state.history.append({"role": "user", "content": prompt_text})
                    st.rerun()

    # Chat history
    for idx, msg in enumerate(st.session_state.history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_metrics(msg.get("time", 0), msg.get("tokens", 0))
                _render_action_buttons(msg["content"], t, key_suffix=f"hist_{idx}")
            if msg.get("docs"):
                _render_sources(msg["docs"])

    # Follow-ups
    if st.session_state.followups:
        st.markdown("#### 🔗 Suggested Follow-ups")
        fu_cols = st.columns(min(len(st.session_state.followups), 3))
        for i, fu_q in enumerate(st.session_state.followups):
            with fu_cols[i % 3]:
                if st.button(fu_q, use_container_width=True, key=f"fu_{i}"):
                    st.session_state.history.append({"role": "user", "content": fu_q})
                    st.session_state.followups = []
                    st.rerun()

    # Regenerate
    if len(st.session_state.history) >= 2 and st.session_state.history[-1]["role"] == "assistant":
        if st.button("🔄 Regenerate Last Response"):
            st.session_state.history.pop()
            st.session_state.followups = []
            st.rerun()

    # ── Determine if we need to generate a response ──────────────────────
    pending_prompt = None

    # Check if last message is an unanswered user message (from quick prompt / follow-up / regenerate)
    if st.session_state.history and st.session_state.history[-1]["role"] == "user":
        pending_prompt = st.session_state.history[-1]["content"]

    # Chat input (new message from the text box)
    if new_input := st.chat_input("Ask about your documents…"):
        st.session_state.history.append({"role": "user", "content": new_input})
        with st.chat_message("user"):
            st.markdown(new_input)
        pending_prompt = new_input

    # ── Generate AI response if there's a pending prompt ─────────────────
    if pending_prompt:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            token_count = 0
            start_time = time.time()

            effective_prompt = _build_effective_prompt()
            stream, docs = get_rag_stream_with_scores(
                pending_prompt, vector_db, llm,
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

            _render_metrics(elapsed, token_count)
            _render_action_buttons(full_response, t, key_suffix="new")

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

            with st.spinner("Generating follow-ups…"):
                st.session_state.followups = generate_followups(pending_prompt, full_response, llm)


# ── TAB 2: SEARCH ──────────────────────────────────────────────────────────

with tab_search:
    st.markdown(
        '<div class="hero">'
        '<h1>🔍 Deep Semantic Search</h1>'
        '<p>Surface relevant passages from your knowledge base instantly.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if vector_db is None:
        st.warning("No document index. Upload and ingest documents first.")
    else:
        search_query = st.text_input("Search query", placeholder="Enter a topic or phrase…", key="search_q")
        search_k = st.slider("Results", 1, 20, 10, key="search_k")

        if search_query:
            results = semantic_search(vector_db, search_query, top_k=search_k, filter_path=focus_path)
            if results:
                st.markdown(f'**{len(results)}** results for *"{html.escape(search_query)}"*')
                for r in results:
                    score = r["score"]
                    cls = "score-high" if score >= 0.7 else ("score-mid" if score >= 0.4 else "score-low")
                    st.markdown(
                        f'<div class="sr-card">'
                        f'<div class="sr-header">'
                        f'<span><strong>{html.escape(r["source"])}</strong> · Page {r["page"]}</span>'
                        f'<span class="score-badge {cls}">{score:.0%}</span>'
                        f'</div>'
                        f'<p class="sr-body">{html.escape(r["content"][:500])}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No results found.")


# ── TAB 3: SUMMARIES ──────────────────────────────────────────────────────

with tab_summary:
    st.markdown(
        '<div class="hero">'
        '<h1>📑 AI-Powered Summaries</h1>'
        '<p>Generate comprehensive overviews of your indexed documents.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if vector_db is None:
        st.warning("No document index. Upload and ingest documents first.")
    elif not files:
        st.info("No documents found.")
    else:
        for f in files:
            with st.expander(f"📄 {f.name}  ({f.stat().st_size / 1024:.0f} KB)", expanded=False):
                if f.name in st.session_state.doc_summaries:
                    st.markdown(st.session_state.doc_summaries[f.name])
                    if st.button("🔄 Regenerate", key=f"regen_{f.name}"):
                        del st.session_state.doc_summaries[f.name]
                        st.rerun()
                else:
                    st.caption("No summary yet.")
                    if st.button("✨ Generate Summary", key=f"gen_{f.name}", use_container_width=True):
                        with st.spinner(f"Summarizing {f.name}…"):
                            doc_results = semantic_search(
                                vector_db, "summary overview main content",
                                top_k=8, filter_path=DATA_DIR / f.name,
                            )
                            if doc_results:
                                context = "\n\n".join(r["content"] for r in doc_results)
                                msgs = [
                                    SystemMessage(content=(
                                        "You are a document summarizer. Provide a comprehensive, "
                                        "well-structured summary. Use markdown with headers and bullets."
                                    )),
                                    HumanMessage(content=(
                                        f"Document: {f.name}\n\nContent:\n{context}\n\nProvide a detailed summary:"
                                    )),
                                ]
                                result = llm.invoke(msgs)
                                summary = getattr(result, "content", str(result))
                                st.session_state.doc_summaries[f.name] = summary
                                st.rerun()
                            else:
                                st.warning("Could not retrieve content for this document.")

        if st.session_state.doc_summaries:
            st.divider()
            all_summaries = "\n\n---\n\n".join(
                f"# {name}\n\n{summary}"
                for name, summary in st.session_state.doc_summaries.items()
            )
            st.download_button(
                "📄 Export All Summaries",
                data=all_summaries,
                file_name=f"summaries_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
