# Technical Architecture & Engineering Report: Pro RAG Chatbot

## 1. Executive Summary
The **Pro RAG Chatbot** is a local-first, privacy-focused Retrieval-Augmented Generation (RAG) application designed to enable secure interaction with private documents. Unlike cloud-native solutions that require uploading sensitive data to external servers, this platform runs primarily on the user's local machine, leveraging efficient open-source models (via **Ollama**) and local vector stores (**FAISS**). It functionality extends to cloud models (**OpenAI**) via a hybrid architecture, offering users the choice between total privacy and maximum performance.

The system transforms static documents (PDFs) and web content into an interactive knowledge base, allowing users to query their data using natural language.

---

## 2. System Architecture

### 2.1 Architectural Pattern: Monolithic Streamlit App
The application follows a **Monolithic Architecture** powered by **Streamlit**. This choice prioritizes rapid development, ease of deployment, and a unified codebase where the User Interface (UI) and Application Logic are tightly coupled but logically separated via module imports.

*   **State Management:** Relies on Streamlit's `st.session_state` and `st.cache_resource` to persist user context (chat history, API keys) and expensive objects (loaded models, vector indices) across re-runs.
*   **Execution Model:** Synchronous, event-driven execution triggered by user interactions (button clicks, text input).

### 2.2 Directory Structure & Component Responsibility
*   **`app.py` (The Controller):** The main entry point. It handles the UI rendering, user input capture, session state management, and orchestrates the RAG pipeline calls.
*   **`src/utils.py` (The Service Layer):** Contains core business logic and helper functions. It abstracts away the complexity of model initialization (`get_llm`), Ollama API communication (`list_ollama_models`), and embedding management.
*   **`src/html_templates.py` (The View Layer):** Stores CSS styles and HTML templates to decouple presentation logic from the main application code (e.g., chat message styling).
*   **`vector_db/` (The Persistence Layer):** Local directory where FAISS vector indices are serialized and stored.

---

## 3. Technology Stack (Tools Used)

The platform utilizes a modern Python-based stack optimized for local inference and RAG.

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Frontend Framework** | **Streamlit** | Rapid UI development for data apps; handles state and reactivity automatically. |
| **RAG Framework** | **LangChain** | Provides the abstraction layer for chains, retrievers, document loaders, and prompt templates. |
| **Local Inference** | **Ollama** | Lightweight, easy-to-deploy runner for open-source LLMs (Llama 3, Mistral, Gemma). |
| **Cloud Inference** | **OpenAI API** | Optional integration for high-reasoning tasks using GPT-4o. |
| **Vector Database** | **FAISS (CPU)** | Facebook AI Similarity Search. Highly efficient for local, dense vector similarity search. No external server required. |
| **Embeddings** | **FastEmbed** | High-performance, lightweight embedding generation (using `Qdrant/FastEmbed`). Faster and smaller than HuggingFace implementations. |
| **Document Parsing** | **PyPDF, WebBaseLoader** | reliable extraction of text from PDF files and Web URLs. |
| **Visualization** | **Streamlit Native** | Used for chat interfaces, sidebars, and interactive widgets. |
| **Document Export** | **FPDF2** | Programmatic generation of PDF reports from chat history. |

---

## 4. Operational Data Flow

The application's logic is divided into two primary pipelines: **Ingestion** (Indexing) and **Inference** (RAG).

### 4.1 Pipeline A: Document Ingestion & Indexing
This process converts raw data into a searchable vector format.

1.  **Input:** User uploads a PDF or provides a URL via the Sidebar.
2.  **Loading:**
    *   **PDFs:** Processed by `PyPDFLoader` to extract raw text pages.
    *   **URLs:** Processed by `WebBaseLoader` to scrape page content.
3.  **Splitting (Chunking):**
    *   The `RecursiveCharacterTextSplitter` breaks long text into smaller, manageable chunks (e.g., 1500 characters) with overlap (e.g., 200 characters). This ensures context is preserved across chunk boundaries.
4.  **Embedding:**
    *   Chunks are passed to `FastEmbedEmbeddings`.
    *   The model converts text into high-dimensional vectors (numerical representations of semantic meaning).
5.  **Indexing:**
    *   Vectors are added to a **FAISS Index**.
    *   This index is saved to disk (`faiss_index`) to strictly avoid re-embedding unchanged documents on every run.

### 4.2 Pipeline B: Retrieval-Augmented Generation (Chat Loop)
This process generates answers based on the indexed data.

1.  **User Query:** The user types a question into the chat input.
2.  **Retrieval:**
    *   The query is embedded into a vector using the same model as the ingestion pipeline.
    *   **FAISS** performs a similarity search (Cosine Similarity) to find the `k` most relevant text chunks from the index.
3.  **Prompt Construction:**
    *   A system prompt is constructed using a template: *"Answer the question based ONLY on the following context..."*
    *   The retrieved text chunks are injected into this template.
4.  **LLM Inference:**
    *   The prompt is sent to the selected LLM (e.g., **Llama 3** running locally on Ollama, or **GPT-4o** via API).
    *   **Hybrid Logic:** `get_llm()` in `src/utils.py` dynamically selects the provider based on the user's dropdown choice.
5.  **Response Streaming:**
    *   The LLM generates the response, which is streamed token-by-token to the UI for a responsive experience.
6.  **Persistence:**
    *   The interaction (User Query + AI Response) is appended to `st.session_state.messages` and optionally saved to `.chat_sessions.json`.

---

## 5. Security & Privacy Architecture

*   **Local-First Design:** By default, no data leaves the user's machine. The vector database and embedding model run entirely on the CPU.
*   **API Key Management:** Cloud API keys (OpenAI) are stored in `st.session_state` (memory only) or `env` variables. They are masked in the UI (`type="password"`).
*   **Git Hygiene:** Sensitive files like `.env`, `vector_db/`, and `.chat_sessions.json` are strictly excluded from version control via `.gitignore`.

## 6. Current Implementation State vs. Future Goals

| Feature | Current Implementation | Status |
| :--- | :--- | :--- |
| **UI Engine** | Streamlit (Python) | ✅ Stable |
| **Vector Store** | FAISS (Local Files) | ✅ Stable |
| **Model Support** | Ollama (Local) + OpenAI (Cloud) | ✅ Hybrid Implemented |
| **Export** | PDF & Markdown | ✅ Implemented |
| **Document Types** | PDF & Web URL | ✅ Implemented |
| **Multi-Modal** | Text Only | 🚧 Planned (Image Support) |
| **Agentic Flow** | Linear Chain | 🚧 Planned (LangGraph) |

---

*This document accurately reflects the codebase as of February 2026.*
