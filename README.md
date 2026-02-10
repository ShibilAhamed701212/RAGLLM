# 🤖 Pro RAG Chatbot

A powerful **Retrieval-Augmented Generation (RAG)** chatbot that lets you chat with your documents locally and privately. Upload PDFs, text files, or markdown — and get accurate, context-aware answers powered by local LLMs via [Ollama](https://ollama.com/).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?logo=ollama&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-yellow)
![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)

---

## ✨ Features

### 🔒 Core
- **Fully Local & Private** — Your documents never leave your machine. No cloud APIs required.
- **Multi-Format Support** — Upload and chat with PDFs, `.txt`, and `.md` files.
- **🌐 URL Ingestion** — Paste any web URL to fetch and index the page content.
- **⚡ Streaming Responses** — Real-time token-by-token output for a smooth chat experience.
- **🖥️ CLI Mode** — Prefer the terminal? Use `cli.py` for a command-line chat interface.

### 🧠 Intelligence
- **💬 Conversation Memory** — Follow-up questions work naturally with 6-turn context window.
- **🔍 Relevance Scores** — Color-coded confidence badges (🟢 high / 🟡 mid / 🔴 low) on every source.
- **🎯 Focus Mode** — Lock the AI's attention to a single document for precise answers.
- **✏️ Custom System Prompt** — Tune the AI's behavior and personality from the sidebar.
- **⚡ Quick Prompts** — One-click starter suggestions (Summarize, Key Points, Author Info, etc.).

### 🎨 Experience
- **🎨 5 Color Themes** — Midnight Purple, Ocean Blue, Emerald, Sunset, Rose Gold.
- **📊 Analytics Dashboard** — Live stats: documents, pages, chunks, queries, avg response time, tokens.
- **⏱️ Response Metrics** — Per-answer timing, token count, and tokens/second.
- **🔊 Text-to-Speech** — Click "Read Aloud" to hear any answer spoken by the browser.
- **💬 Chat Sessions** — Save, name, and switch between multiple conversations.
- **💾 Export Chat** — Download your conversation as a Markdown file.

### 🛠️ Model Management
- **🧠 Dynamic Model Switching** — Switch between any locally installed Ollama model instantly.
- **📥 Download Models from UI** — Pull new Ollama models directly from the sidebar.
- **🔧 Configurable** — Tune temperature, search depth, chunk size, and more via `.env` or the UI.

---

## 📁 Project Structure

```
RAGLLM/
├── app.py              # Streamlit web interface
├── cli.py              # Terminal-based chat interface
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (create this)
├── src/
│   ├── config.py       # All settings & defaults
│   ├── utils.py        # Embeddings, LLM, FAISS, Ollama helpers
│   ├── ingestion.py    # Document loading, chunking, indexing
│   └── core.py         # RAG logic — retriever, prompts, generation
├── data/               # Uploaded documents (auto-created)
└── vector_index/       # FAISS index storage (auto-created)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** installed and running
- At least one Ollama model pulled (e.g., `ollama pull llama3.2:3b`)

### Installation

```bash
# Clone the repo
git clone https://github.com/ShibilAhamed701212/RAGLLM.git
cd RAGLLM

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration (Optional)

Create a `.env` file in the project root to customize settings:

```env
# LLM Provider: "ollama" (default) or "openai"
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b

# OpenAI (only if LLM_PROVIDER=openai)
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini

# Embedding model
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Retrieval settings
TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_TEMPERATURE=0.1
SEARCH_TYPE=similarity
```

### Run the App

```bash
# Web UI (recommended)
streamlit run app.py

# CLI mode
python cli.py
```

---

## 🧠 Supported Ollama Models

You can use **any model** from the [Ollama Library](https://ollama.com/library). Here are some recommended ones:

| Model | Size | Best For |
|---|---|---|
| `llama3.2:1b` | 1.3 GB | Fast responses, low RAM |
| `llama3.2:3b` | 2.0 GB | Good balance (default) |
| `llama3:8b` | 4.7 GB | High quality answers |
| `gemma3:4b` | 3.3 GB | Strong reasoning |
| `phi4-mini` | 2.5 GB | Efficient & capable |
| `mistral` | 4.1 GB | Great all-rounder |
| `qwen3:4b` | 2.5 GB | Multilingual support |
| `deepseek-r1:8b` | 4.9 GB | Advanced reasoning |

**Download any model** directly from the app sidebar or via terminal:
```bash
ollama pull gemma3:4b
```

---

## 🔧 How It Works

1. **Upload** — Drop your PDFs, text files, or markdown into the app.
2. **Ingest** — Documents are chunked, embedded using `bge-small-en-v1.5`, and stored in a FAISS vector index.
3. **Query** — Your question is embedded and matched against the most relevant document chunks.
4. **Generate** — The retrieved context + your question are sent to the local LLM, which generates a grounded answer.

```
User Question → Embed → FAISS Search → Top-K Chunks → LLM → Answer
```

---

## 🛡️ Privacy

This project is designed for **complete privacy**:
- All processing happens **locally** on your machine.
- No data is sent to external servers (when using Ollama).
- Documents are stored in a local `data/` directory.
- Vector embeddings are stored in a local `vector_index/` directory.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/ShibilAhamed701212">Shibil Ahamed</a>
</p>
