"""
Ingestion — Load documents, chunk them, and build a FAISS vector index.
"""

import concurrent.futures
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import DATA_DIR, VECTOR_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import get_embeddings

# File extensions we support and their corresponding loaders.
_LOADERS = {
    ".pdf": lambda p: PyPDFLoader(str(p)),
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".md":  lambda p: TextLoader(str(p), encoding="utf-8"),
}


# ── Loading ────────────────────────────────────────────────────────────────────

def _load_single(file_path: Path) -> List:
    """Load a single file using the appropriate LangChain loader."""
    loader_fn = _LOADERS.get(file_path.suffix.lower())
    if loader_fn is None:
        return []
    try:
        return loader_fn(file_path).load()
    except Exception as exc:
        print(f"[WARNING] Skipped {file_path.name}: {exc}")
        return []


def load_documents(source_dir: Path = DATA_DIR) -> List:
    """Load every supported file from *source_dir* in parallel (I/O-bound)."""
    if not source_dir.exists():
        print(f"[INFO] Source directory does not exist: {source_dir}")
        return []

    files = [f for ext in _LOADERS for f in source_dir.glob(f"*{ext}")]
    if not files:
        return []

    with concurrent.futures.ThreadPoolExecutor() as pool:
        batches = list(pool.map(_load_single, files))

    return [doc for batch in batches for doc in batch]


# ── URL Ingestion ──────────────────────────────────────────────────────────────

def ingest_url(url: str) -> tuple[bool, str]:
    """Fetch a web page and save its text content for indexing.

    Returns (success, message).
    """
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (ProRAG Bot)"
        })
        resp.raise_for_status()

        # Extract text — try to strip HTML tags simply
        content = resp.text
        if "<html" in content.lower():
            # Basic HTML stripping
            import re
            content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.S)
            content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.S)
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()

        if not content or len(content) < 50:
            return False, "Page content too short or empty."

        # Save as text file
        domain = urlparse(url).netloc.replace(".", "_")
        path_slug = urlparse(url).path.strip("/").replace("/", "_")[:50]
        filename = f"web_{domain}_{path_slug}.txt" if path_slug else f"web_{domain}.txt"

        DATA_DIR.mkdir(exist_ok=True)
        (DATA_DIR / filename).write_text(content, encoding="utf-8")

        return True, f"Saved as {filename} ({len(content)} chars)"
    except Exception as exc:
        return False, str(exc)


# ── Chunking ───────────────────────────────────────────────────────────────────

def split_documents(
    docs: List,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List:
    """Split documents into overlapping chunks for embedding."""
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


# ── Index creation ─────────────────────────────────────────────────────────────

def create_vector_index(
    docs: List,
    save_path: Path = VECTOR_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> FAISS | None:
    """Chunk *docs*, embed them, build a FAISS index, and save to disk."""
    if not docs:
        print("[INFO] Nothing to index — no documents provided.")
        return None

    chunks = split_documents(docs, chunk_size, chunk_overlap)
    print(f"[INFO] Created {len(chunks)} chunks from {len(docs)} pages.")

    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)

    save_path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(save_path))
    print(f"[INFO] Vector index saved to {save_path}")
    return db


# ── Public entry-point ─────────────────────────────────────────────────────────

def ingest_all(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    """Full ingestion pipeline: load → chunk → embed → save."""
    print(f"[INFO] Ingesting from {DATA_DIR} …")
    docs = load_documents(DATA_DIR)
    print(f"[INFO] Loaded {len(docs)} pages / sections.")

    if docs:
        create_vector_index(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        print("[WARNING] No valid documents found to ingest.")


if __name__ == "__main__":
    ingest_all()
