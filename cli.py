"""
CLI Chat — Terminal-based interface for the RAG system.

Usage:
    python cli.py
"""

from src.config import DEFAULT_TEMPERATURE, TOP_K
from src.utils import get_embeddings, get_llm, load_faiss_index
from src.core import get_retriever, get_rag_stream


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

    retriever = get_retriever(db, top_k=TOP_K)

    try:
        llm = get_llm(temperature=DEFAULT_TEMPERATURE)
    except Exception as exc:
        print(f"Error loading LLM: {exc}")
        return

    print("System ready! Type 'exit' to quit.\n")

    while True:
        try:
            query = input("\033[94mYou:\033[0m ")
            if query.strip().lower() in {"exit", "quit"}:
                break
            if not query.strip():
                continue

            print("\033[92mBot:\033[0m ", end="", flush=True)

            stream, _docs = get_rag_stream(query, retriever, llm)
            for chunk in stream:
                print(getattr(chunk, "content", str(chunk)), end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
