"""
Core RAG logic — retriever setup, prompt construction, and response generation.
"""

from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import TOP_K, SEARCH_TYPE

# ── System Prompt (single source of truth) ─────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise document-analysis assistant. Your ONLY job is to answer questions \
using the CONTEXT provided below. Follow these rules strictly:

1. Answer DIRECTLY — no preamble, no "Based on the context…", no meta-commentary.
2. Use ONLY information from the CONTEXT. Never use outside knowledge.
3. If the context contains the answer, give a thorough, detailed response.
4. If the user asks for a summary, provide a comprehensive summary of ALL the content in the context.
5. If the answer is truly not in the context, say exactly: \
"I'm sorry, but the provided documents do not contain information to answer this question."
6. Treat document filenames as metadata — if the author or title appears in the filename, state it as fact."""

_CONTEXT_TEMPLATE = """\
=== RETRIEVED DOCUMENT CONTEXT ===

{context}

=== END OF CONTEXT ==="""


# ── Helpers ────────────────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    """Combine retrieved documents into a single context string with metadata."""
    if not docs:
        return "(No documents were retrieved.)"

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", "?")
        header = f"[Source {i}: {source} | Page {page}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_messages(query: str, docs: List[Document]) -> list:
    """Build a proper list of chat messages with system + human roles."""
    context_block = _CONTEXT_TEMPLATE.format(context=format_docs(docs))
    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"{context_block}\n\nQuestion: {query}"),
    ]


# ── Retriever ──────────────────────────────────────────────────────────────────

def get_retriever(
    db,
    top_k: int = TOP_K,
    search_type: str = SEARCH_TYPE,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    filter_path=None,
):
    """Return a LangChain retriever, optionally filtered to a single source file."""
    search_kwargs = {"k": top_k}

    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult

    if filter_path is not None:
        search_kwargs["filter"] = {"source": str(filter_path)}

    return db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


# ── Response generation ───────────────────────────────────────────────────────

def get_rag_response(query: str, retriever, llm) -> Tuple:
    """Retrieve docs → build messages → return (full_response, docs)."""
    docs = retriever.invoke(query)
    messages = _build_messages(query, docs)
    response = llm.invoke(messages)
    return response, docs


def get_rag_stream(query: str, retriever, llm) -> Tuple:
    """Retrieve docs → build messages → return (streaming_iterator, docs)."""
    docs = retriever.invoke(query)
    messages = _build_messages(query, docs)
    return llm.stream(messages), docs
