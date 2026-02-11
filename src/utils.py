"""
Utilities — Embedding model, LLM, FAISS index loaders, and Ollama helpers.
"""

from __future__ import annotations

from pathlib import Path

import requests
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import (
    VECTOR_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    DEFAULT_TEMPERATURE,
)

OLLAMA_BASE_URL = "http://localhost:11434"

_EMPTY_STATS: dict = {
    "total_chunks": 0,
    "unique_sources": 0,
    "total_pages": 0,
    "sources": [],
}


# ── Ollama helpers ─────────────────────────────────────────────────────────────

def list_ollama_models() -> list[dict]:
    """Query the local Ollama server for all installed models.

    Returns a list of dicts with 'name' and 'size_gb' keys, sorted by size.
    Falls back to a single default entry if the server is unreachable.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [
            {
                "name": m["name"],
                "size_gb": round(m.get("size", 0) / 1e9, 1),
                "family": m.get("details", {}).get("family", ""),
            }
            for m in sorted(models, key=lambda x: x.get("size", 0))
        ]
    except Exception:
        return [{"name": LLM_MODEL, "size_gb": 0, "family": "unknown"}]


def pull_ollama_model(model_name: str) -> bool:
    """Pull (download) a model from the Ollama registry. Returns True on success."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            timeout=600,
            stream=True,
        )
        resp.raise_for_status()
        for _ in resp.iter_lines():
            pass
        return True
    except Exception:
        return False


def delete_ollama_model(model_name: str) -> bool:
    """Delete a locally installed Ollama model. Returns True on success."""
    try:
        resp = requests.delete(
            f"{OLLAMA_BASE_URL}/api/delete",
            json={"name": model_name},
            timeout=30,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ── Core loaders ───────────────────────────────────────────────────────────────

def get_embeddings() -> FastEmbedEmbeddings:
    """Return a FastEmbed embedding model (CPU-optimised, no PyTorch needed)."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm(
    temperature: float = DEFAULT_TEMPERATURE,
    model: str | None = None,
    api_key: str | None = None,
):
    """Return the configured LLM (Ollama local or OpenAI cloud).

    Args:
        temperature: Sampling temperature.
        model: Model name (e.g. 'gpt-4', 'llama3'). Auto-detects provider.
        api_key: Optional API key for cloud providers.
    """
    model = model or LLM_MODEL
    
    # Check if it's an OpenAI model
    if model.startswith(("gpt-", "o1-")):
        if not api_key:
            # Fallback to env var or raise
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API Key is missing. Enter it in the sidebar.")
            api_key = OPENAI_API_KEY
            
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
        
    # Default to Ollama
    return ChatOllama(model=model, temperature=temperature)


def load_faiss_index(embeddings) -> FAISS | None:
    """Load a saved FAISS index from disk, or return None if it doesn't exist."""
    index_file = VECTOR_DIR / "index.faiss"
    if not index_file.exists():
        return None
    try:
        return FAISS.load_local(
            str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        print(f"[WARNING] Failed to load FAISS index: {exc}")
        return None


# ── Analytics ──────────────────────────────────────────────────────────────────

def get_index_stats(db) -> dict:
    """Return statistics about the loaded FAISS vector index."""
    if db is None:
        return dict(_EMPTY_STATS)

    try:
        all_docs = db.docstore._dict
        total = len(all_docs)
        sources: set[str] = set()
        pages: set[str] = set()
        for doc in all_docs.values():
            src = Path(doc.metadata.get("source", "Unknown")).name
            sources.add(src)
            page = doc.metadata.get("page", "?")
            pages.add(f"{src}:{page}")
        return {
            "total_chunks": total,
            "unique_sources": len(sources),
            "total_pages": len(pages),
            "sources": sorted(sources),
        }
    except Exception:
        return dict(_EMPTY_STATS)


# ── Semantic search ────────────────────────────────────────────────────────────

def semantic_search(
    db, query: str, top_k: int = 10, filter_path=None,
) -> list[dict]:
    """Run a pure semantic search (no LLM) and return scored results."""
    if db is None:
        return []
    try:
        kwargs: dict = {}
        if filter_path is not None:
            kwargs["filter"] = {"source": str(filter_path)}

        results = db.similarity_search_with_relevance_scores(query, k=top_k, **kwargs)
        return [
            {
                "content": doc.page_content,
                "source": Path(doc.metadata.get("source", "Unknown")).name,
                "page": doc.metadata.get("page", "?"),
                "score": round(score, 4),
            }
            for doc, score in results
        ]
    except Exception:
        return []
