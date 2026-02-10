"""
Pro RAG Chatbot — Streamlit interface for chatting with your documents.
"""

import shutil
import time
from pathlib import Path

import streamlit as st

from src.config import (
    DATA_DIR,
    VECTOR_DIR,
    TOP_K,
    DEFAULT_TEMPERATURE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.utils import get_embeddings, get_llm, load_faiss_index, list_ollama_models, pull_ollama_model
from src.ingestion import ingest_all
from src.core import get_retriever, get_rag_stream

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pro RAG Chatbot", page_icon="🤖", layout="wide")


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


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📁 Document Control")

    # -- Indexed files --
    st.subheader("Currently Indexed Files")
    files = list(DATA_DIR.glob("*.*")) if DATA_DIR.exists() else []
    if files:
        for f in files:
            st.caption(f"✅ {f.name}")
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
        st.success("System reset! Upload new files.")
        time.sleep(1)
        st.rerun()

    st.divider()

    # -- Select Model --
    st.subheader("🧠 Select Model")
    ollama_models = list_ollama_models()
    model_names = [m["name"] for m in ollama_models]
    model_labels = [f"{m['name']}  ({m['size_gb']} GB)" for m in ollama_models]

    # Default to the configured model from config.py
    from src.config import LLM_MODEL
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

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ── Initialise resources ──────────────────────────────────────────────────────

embeddings = _cached_embeddings()
vector_db = _cached_vector_db(embeddings)
llm = _cached_llm(temperature, selected_model)

# ── Main chat area ────────────────────────────────────────────────────────────

st.title("🤖 Pro RAG Chatbot")
st.caption("Chat with your documents locally and privately.")
st.markdown(f"**🧠 Active Model:** `{selected_model}`")

if vector_db is None:
    st.warning("No document index found. Upload and ingest PDFs in the sidebar.")
    st.stop()

retriever = get_retriever(vector_db, top_k=top_k, filter_path=focus_path)

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("docs"):
            with st.expander("Sources"):
                for doc in msg["docs"]:
                    source = Path(doc.metadata.get("source", "Unknown")).name
                    st.write(f"- **{source}**: {doc.page_content[:200]}…")

# Handle new input
if prompt := st.chat_input("Ask about your documents…"):
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        stream, docs = get_rag_stream(prompt, retriever, llm)
        for chunk in stream:
            full_response += getattr(chunk, "content", str(chunk))
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        st.session_state.history.append(
            {"role": "assistant", "content": full_response, "docs": docs}
        )

        with st.expander("View Sources"):
            for doc in docs:
                source = Path(doc.metadata.get("source", "Unknown")).name
                page = doc.metadata.get("page", "?")
                st.write(f"- **{source}** (Page {page})")
                st.info(doc.page_content)
