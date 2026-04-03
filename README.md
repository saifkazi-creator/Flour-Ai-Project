# 🏭 Flour Mill AI Maintenance Assistant

A Streamlit RAG chatbot for industrial flour-mill maintenance teams, powered by **Gemini 2.0 Flash**, ChromaDB, and LangChain.

---

## Project Structure

```
flour_ai_project/
├── app.py                          ← Streamlit UI
├── requirements.txt
├── .env.example                    ← copy to .env and add your API key
├── agent/
│   └── agent_controller.py         ← keyword router + streaming generator
├── rag/
│   ├── llm_engine.py               ← Gemini 2.0 Flash streaming wrapper
│   ├── retriever.py                ← ChromaDB singleton loader
│   └── qa_engine.py                ← standalone debug/test script
├── tools/
│   ├── manual_tool.py
│   ├── log_tool.py
│   ├── schedule_tool.py
│   └── troubleshooting_tool.py
├── ingestion/
│   ├── build_vector_store.py       ← run this FIRST
│   ├── load_manuals.py
│   ├── load_logs.py
│   ├── load_schedule.py
│   └── load_troubleshooting.py
└── data/
    ├── manuals/                    ← drop PDF manuals here
    ├── logs/maintenance_logs.csv
    ├── schedules/maintenance_schedule.csv
    ├── troubleshooting/troubleshooting_guide.txt
    └── uploads/                    ← runtime PDF uploads land here
```

---

## Setup

### 1. Clone / copy the project and install dependencies

```bash
cd flour_ai_project
pip install -r requirements.txt
```

### 2. Set your Google API key

```bash
cp .env.example .env
# Edit .env and paste your key from https://aistudio.google.com/app/apikey
```

Then export it before running:

```bash
export GOOGLE_API_KEY=your_actual_key_here
```

Or on Windows:

```cmd
set GOOGLE_API_KEY=your_actual_key_here
```

### 3. Build the vector store (required before first launch)

```bash
python ingestion/build_vector_store.py
```

This embeds all logs, schedules, troubleshooting entries, and any PDFs in `data/manuals/` into a local ChromaDB database.

### 4. Launch the app

```bash
streamlit run app.py
```

---

## Usage

| Query type | Example question |
|---|---|
| 📋 Maintenance logs | "What happened with the Roller Mill last time?" |
| 📅 Schedule | "What is the upcoming service due for the Purifier?" |
| 🔧 Troubleshooting | "Why is the conveyor motor overheating?" |
| 📘 Manual | "How do I adjust the roll gap?" |

### Adding PDF manuals
Drop `.pdf` files into `data/manuals/` and click **🔄 Rebuild AI Knowledge Base** in the sidebar (or re-run `python ingestion/build_vector_store.py`).

---

## Architecture

```
User query
    │
    ▼
detect_query_type()   ← keyword router
    │
    ├── "log"           → log_tool       → ChromaDB (type=maintenance_log)
    ├── "schedule"      → schedule_tool  → ChromaDB (type=maintenance_schedule)
    ├── "troubleshooting" → troubleshooting_tool → ChromaDB + Gemini stream
    └── "manual"        → manual_tool    → ChromaDB (type=manual) + Gemini stream
```

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, no API needed)
- **Vector store:** ChromaDB (persisted to `./vectorstore/`)
- **LLM:** Gemini 2.0 Flash via `langchain-google-genai`
