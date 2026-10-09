# AI Agent — RAG Over Your Documents

A simple AI agent that answers questions **only** from documents you give it, using
Retrieval-Augmented Generation (RAG). Ask a question in plain English and get an answer
pulled directly from your files — along with the exact passages it used.

Think of it as an assistant who has read your entire employee handbook and can instantly
tell you *"here's the vacation policy, and here's the paragraph I got it from."* The value
is the time saved: no scrolling through 50 pages to find one fact.

The key difference from asking ChatGPT or Gemini directly is that this agent answers
**only from your documents**, not its general knowledge. Answers are specific to your
content and cite their source. If the answer isn't in your documents, it says so instead
of guessing (or hallucinating).

---

## Tech Stack

| Layer | Technology | Role |
| --- | --- | --- |
| HTTP API | **FastAPI** + Uvicorn | Receives requests, returns JSON |
| Orchestration | **LangChain** | Chains together the RAG steps |
| Vector DB | **ChromaDB** | Stores embeddings; semantic similarity search (HNSW index). Lightweight and well-suited to small projects (vs. Elastic) |
| Embeddings | **Google `gemini-embedding-001`** | Converts text ↔ vectors (3072-dim) |
| LLM | **Google `gemini-2.5-flash`** | Generates the answer from retrieved context |
| Frontend | **React** (single file, CDN, no build) | Minimal UI for ingest + ask |

---

## How It Works

RAG runs in two phases:

```
INDEXING  (run once)    documents → chunks → embeddings → ChromaDB
QUERYING  (every ask)   question  → embedding → similarity search → top-k chunks → LLM → answer
```

- **Indexing** — `.txt` files are loaded, split into ~500-character chunks (50-char
  overlap), embedded with Gemini, and stored in ChromaDB. Chunks persist to disk, so the
  server can answer immediately after a restart without re-embedding.
- **Querying** — the question is embedded, ChromaDB returns the top-`k` most similar
  chunks, and those chunks + the question are sent to Gemini 2.5 Flash to generate a
  grounded answer with citations.

---

## Setup

**Prerequisites:** Python 3.10+ and a Google Generative AI API key.

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

Provide your Google API key via a `.env` file in this directory (it is loaded
automatically at startup):

```
GOOGLE_API_KEY=your-key-here
```

---

## Running

```bash
# Start the API (http://localhost:8000)
python main.py
```

- Interactive Swagger UI: **http://localhost:8000/docs**
- Frontend: open **`frontend/index.html`** in a browser while the server is running.

**First-time use:** put your `.txt` files in a `documents/` folder, then index them —
either click **Ingest** in the UI or:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents_path": "./documents"}'
```

Re-run ingest whenever your source documents change.

---

## API Reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Is the server alive and are documents loaded? |
| `POST` | `/ingest` | Index `.txt` documents from a folder into ChromaDB |
| `POST` | `/query` | Ask a question, get an AI answer with sources |

**`POST /ingest`**
```json
// request
{ "documents_path": "./documents" }
// response
{ "documents_loaded": 1, "chunks_created": 8, "status": "success" }
```

**`POST /query`**
```json
// request  (num_results: 1–10, default 3)
{ "question": "What is FastAPI?", "num_results": 3 }
// response
{
  "question": "What is FastAPI?",
  "answer": "FastAPI is a modern Python web framework...",
  "sources": [
    { "content": "FastAPI is a modern, fast...", "metadata": { "source": "documents/sample.txt" } }
  ]
}
```

---

## Project Structure

```
code/
├── main.py            # FastAPI app — wires the RAG engine + schemas into 3 endpoints
├── rag_engine.py      # Core RAG pipeline: load → split → embed → store → retrieve → generate
├── schemas.py         # Pydantic request/response models (validation)
├── frontend/
│   └── index.html     # Single-file React UI (no build step)
└── chroma_db/         # Persisted vector storage (auto-created on first ingest)
```

---

## Sample Tests

**Document:** `documents/investing.txt`

1. *"What is the difference between a Roth IRA and a traditional IRA?"* — answer spans two separate sections
2. *"How does compound interest work?"* — single focused section
3. *"What are some common psychological mistakes investors make?"* — behavioral-finance section
4. *"Should I invest in ETFs or mutual funds?"* — needs to synthesize across multiple chunks

## Example Usage
### Step 1: Open index.html in browser
<img width="819" height="552" alt="S1" src="https://github.com/user-attachments/assets/9930113f-a5da-4065-bcc7-2ac5c8e84c7b" />

### Step 2: Startup main.py & Click on Refresh to See the Status Change to Healthy
<img width="822" height="104" alt="s2" src="https://github.com/user-attachments/assets/69b6da29-50fa-446f-96f8-fa1dd95c27e2" />

### Step 3: Set the document Folder Path and Click Ingest 
<img width="1035" height="604" alt="s3" src="https://github.com/user-attachments/assets/bc75d7a6-4f95-48ba-9790-9086321ea098" />

### Step 4: Ask a Question Related to a .txt File in the documents Folder & Set the # of Sources
###         (Note: More sources => more token usage but better context)
<img width="831" height="555" alt="s4" src="https://github.com/user-attachments/assets/1b638908-3364-460a-91d8-2093d91e0715" />


