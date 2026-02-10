"""
Utilities — Embedding model, LLM, FAISS index loaders, and Ollama helpers.
"""

import requests
from pathlib import Path

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


def list_ollama_models() -> list[dict]:
    """Query the local Ollama server for all installed models.

    Returns a list of dicts with 'name' and 'size' keys, sorted by size.
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
        # Consume the stream to complete the download
        for _ in resp.iter_lines():
            pass
        return True
    except Exception:
        return False


def get_embeddings() -> FastEmbedEmbeddings:
    """Return a FastEmbed embedding model (CPU-optimised, no PyTorch needed)."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm(temperature: float = DEFAULT_TEMPERATURE, model: str | None = None):
    """Return the configured LLM (Ollama local or OpenAI cloud).

    Args:
        temperature: Sampling temperature.
        model: Override the default Ollama model name. Ignored for OpenAI.
    """
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                "Add it to your .env file."
            )
        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)
    return ChatOllama(model=model or LLM_MODEL, temperature=temperature)


def load_faiss_index(embeddings) -> FAISS | None:
    """Load a saved FAISS index from disk, or return None if it doesn't exist."""
    index_file = VECTOR_DIR / "index.faiss"
    if not index_file.exists():
        return None
    try:
        return FAISS.load_local(
            str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True
        )
    except Exception as exc:
        print(f"[WARNING] Failed to load FAISS index: {exc}")
        return None


def get_index_stats(db) -> dict:
    """Return statistics about the loaded FAISS vector index."""
    if db is None:
        return {"total_chunks": 0, "unique_sources": 0, "sources": []}

    try:
        all_docs = db.docstore._dict
        total = len(all_docs)
        sources = set()
        pages = set()
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
        return {"total_chunks": 0, "unique_sources": 0, "sources": []}


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


def semantic_search(db, query: str, top_k: int = 10, filter_path=None) -> list[dict]:
    """Run a pure semantic search (no LLM) and return scored results."""
    if db is None:
        return []
    try:
        kwargs = {}
        if filter_path is not None:
            kwargs["filter"] = {"source": str(filter_path)}

        results = db.similarity_search_with_relevance_scores(query, k=top_k, **kwargs)
        return [
            {
                "content": doc.page_content,
                "source": Path(doc.metadata.get("source", "Unknown")).name,
                "page": doc.metadata.get("page", "?"),
                "score": score,
            }
            for doc, score in results
        ]
    except Exception:
        return []
