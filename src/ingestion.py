"""
Ingestion — Load documents, chunk them, and build a FAISS vector index.
"""

import concurrent.futures
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
