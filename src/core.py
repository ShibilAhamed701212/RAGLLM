"""
Core RAG logic — retriever setup, prompt construction, and response generation.
"""

from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import TOP_K, SEARCH_TYPE

# ── Default System Prompt ──────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
You are a precise document-analysis assistant. Your ONLY job is to answer questions \
using the CONTEXT provided below. Follow these rules strictly:

1. Answer DIRECTLY — no preamble, no "Based on the context…", no meta-commentary.
2. Use ONLY information from the CONTEXT. Never use outside knowledge.
3. If the context contains the answer, give a thorough, detailed response.
4. If the user asks for a summary, provide a comprehensive summary of ALL the content in the context.
5. If the answer is truly not in the context, say exactly: \
"I'm sorry, but the provided documents do not contain information to answer this question."
6. Treat document filenames as metadata — if the author or title appears in the filename, state it as fact.
7. When continuing a conversation, use the chat history for context but always ground answers in the documents."""

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
        score = doc.metadata.get("score")
        score_str = f" | Relevance: {score:.0%}" if score is not None else ""
        header = f"[Source {i}: {source} | Page {page}{score_str}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_messages(
    query: str,
    docs: List[Document],
    chat_history: list | None = None,
    system_prompt: str | None = None,
) -> list:
    """Build a proper list of chat messages with system + human roles.

    Includes conversation history for multi-turn context.
    """
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    context_block = _CONTEXT_TEMPLATE.format(context=format_docs(docs))

    messages = [SystemMessage(content=prompt)]

    # Add recent conversation history for multi-turn context (last 6 turns)
    if chat_history:
        recent = chat_history[-6:]
        for msg in recent:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=msg["content"]))

    # Current query with context
    messages.append(HumanMessage(content=f"{context_block}\n\nQuestion: {query}"))
    return messages


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


# ── Similarity search with scores ─────────────────────────────────────────────

def retrieve_with_scores(
    db, query: str, top_k: int = TOP_K, filter_path=None
) -> List[Document]:
    """Retrieve documents with similarity scores attached to metadata."""
    kwargs = {}
    if filter_path is not None:
        kwargs["filter"] = {"source": str(filter_path)}

    results = db.similarity_search_with_relevance_scores(query, k=top_k, **kwargs)

    docs = []
    for doc, score in results:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


# ── Response generation ───────────────────────────────────────────────────────

def get_rag_response(
    query: str, retriever, llm,
    chat_history=None, system_prompt=None,
) -> Tuple:
    """Retrieve docs → build messages → return (full_response, docs)."""
    docs = retriever.invoke(query)
    messages = _build_messages(query, docs, chat_history, system_prompt)
    response = llm.invoke(messages)
    return response, docs


def get_rag_stream(
    query: str, retriever, llm,
    chat_history=None, system_prompt=None,
) -> Tuple:
    """Retrieve docs → build messages → return (streaming_iterator, docs)."""
    docs = retriever.invoke(query)
    messages = _build_messages(query, docs, chat_history, system_prompt)
    return llm.stream(messages), docs


def get_rag_stream_with_scores(
    query: str, db, llm,
    top_k: int = TOP_K, filter_path=None,
    chat_history=None, system_prompt=None,
) -> Tuple:
    """Retrieve docs with scores → build messages → return (streaming_iterator, docs)."""
    docs = retrieve_with_scores(db, query, top_k, filter_path)
    messages = _build_messages(query, docs, chat_history, system_prompt)
    return llm.stream(messages), docs
