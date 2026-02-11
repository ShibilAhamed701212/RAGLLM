"""
CLI Chat — Terminal-based interface for the RAG system.

Usage:
    python cli.py
"""

from __future__ import annotations

from src.config import DEFAULT_TEMPERATURE, TOP_K
from src.core import get_rag_stream
from src.utils import get_embeddings, get_llm, load_faiss_index


def main() -> None:
    print("\n" + "=" * 44)
    print("        LOCAL RAG CLI CHAT")
    print("=" * 44 + "\n")

    # Load resources
    print("Initialising components…")
    embeddings = get_embeddings()
    db = load_faiss_index(embeddings)

    if db is None:
        print("Error: No vector index found. Run ingestion first (or use the GUI).")
        return

    retriever = get_rag_stream.__wrapped__ if hasattr(get_rag_stream, "__wrapped__") else None  # noqa
    # Build retriever
    from src.core import get_retriever
    retriever = get_retriever(db, top_k=TOP_K)

    try:
        llm = get_llm(temperature=DEFAULT_TEMPERATURE)
    except Exception as exc:
        print(f"Error loading LLM: {exc}")
        return

    print("System ready! Type 'exit' to quit.\n")

    history: list[dict[str, str]] = []

    while True:
        try:
            query = input("\033[94mYou:\033[0m ")
            if query.strip().lower() in {"exit", "quit"}:
                break
            if not query.strip():
                continue

            print("\033[92mBot:\033[0m ", end="", flush=True)

            stream, _docs = get_rag_stream(query, retriever, llm, chat_history=history)
            full_response = ""
            for chunk in stream:
                content = getattr(chunk, "content", str(chunk))
                full_response += content
                print(content, end="", flush=True)
            print("\n")

            # Maintain conversation history
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": full_response})
            # Keep last 10 turns
            if len(history) > 20:
                history = history[-20:]

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
