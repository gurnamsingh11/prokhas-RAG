# RAG Backend — Production-Ready FastAPI

A fully production-ready Retrieval-Augmented Generation (RAG) backend built with FastAPI, LangChain, LangGraph, and FAISS.

---

## Architecture

```
User
 │
 ├── POST /api/v1/sessions/upload  (zip file)
 │        │
 │        ├── Extract zip → collect supported files (PDF / DOCX / images)
 │        ├── UniversalDocumentLoader  (PDF → PyPDF, DOCX → docx2txt, img → OCR)
 │        ├── SmartChunker  (SemanticChunker → falls back to RecursiveCharacterTextSplitter)
 │        ├── HuggingFaceEmbeddings  (google/gemma-embedding-exp-03-07)
 │        ├── FAISS in-memory store  (keyed by session_id)
 │        └── SessionRegistry  (metadata + InMemorySaver checkpointer)
 │
 └── POST /api/v1/sessions/{session_id}/chat  (message)
          │
          ├── LangGraph agent  (create_agent)
          │    ├── Short-term memory  →  InMemorySaver (per session, thread_id = session_id)
          │    ├── Tool: retrieve_context  →  FAISS similarity search
          │    └── Structured output  →  ToolStrategy[RAGResponse]  (TypedDict, no Pydantic)
          │
          └── ChatResponseOut  { answer, sources[] { document_name, page_label, excerpt } }
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Optional but recommended for semantic chunking:
pip install langchain-experimental
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your HUGGINGFACEHUB_API_TOKEN
```

### 3. Run the server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

---

## API Reference

### `POST /api/v1/sessions/upload`

Upload a `.zip` archive containing PDFs, DOCX files, and/or images.

**Request:** `multipart/form-data` with field `file` (the zip).

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Session created. You can now chat using the session_id.",
  "files_processed": 3,
  "chunks_indexed": 142
}
```

---

### `POST /api/v1/sessions/{session_id}/chat`

Ask a question grounded in the uploaded documents.

**Request body:**
```json
{ "message": "What is the main topic of document X?" }
```

**Response:**
```json
{
  "session_id": "550e8400-...",
  "answer": "Document X covers ...",
  "sources": [
    {
      "document_name": "report.pdf",
      "page_label": "3",
      "relevant_excerpt": "...the main topic is renewable energy..."
    }
  ]
}
```

---

### `GET /api/v1/sessions`

List all active sessions.

### `GET /api/v1/sessions/{session_id}`

Get metadata for a session (files, chunk count, timestamps).

### `DELETE /api/v1/sessions/{session_id}`

Delete a session and **permanently remove its vector store from memory**.

---

## Session Lifecycle

| Event | Effect |
|-------|--------|
| Upload zip | New session created with UUID |
| Each chat message | `last_active` timestamp refreshed |
| Idle > `SESSION_TTL_SECONDS` | Session auto-expired on next request |
| `DELETE /sessions/{id}` | Immediate deletion |

Default TTL: **1 hour** (configurable via `SESSION_TTL_SECONDS` env var).

---

## Making Embeddings Persistent (Future)

By default, embeddings are in-memory and deleted with the session. To make them
**persistent and reusable by session name**:

1. Open `src/vectorstore/session_store.py`
2. In `build_session_store()`, add after creating the FAISS index:
   ```python
   store.save_local(f"./faiss_indexes/{session_id}")
   ```
3. In `delete_session_store()`, remove or comment out the `_STORE_REGISTRY.pop()` call (keep the file on disk).
4. Add a `load_session_store(session_id, embeddings)` function:
   ```python
   FAISS.load_local(f"./faiss_indexes/{session_id}", embeddings, allow_dangerous_deserialization=True)
   ```
5. In `session_registry.py`, call `load_session_store()` when a session is resumed by name.

No other changes are needed — the rest of the codebase is unaffected.

---

## Structured Output Design

The LLM returns a **TypedDict**-based structured response (`RAGResponse`) — no Pydantic used for the LLM schema, per spec:

```python
class SourceReference(TypedDict):
    document_name: str
    page_label: str
    relevant_excerpt: str

class RAGResponse(TypedDict):
    answer: str
    sources: List[SourceReference]
```

`ToolStrategy` is used explicitly for compatibility with HuggingFace endpoints that may not support native structured output. For OpenAI / Anthropic / xAI models, you can switch to `ProviderStrategy` or pass the schema directly (LangChain auto-selects the best strategy).

---

## Short-Term Memory

Each session has its own `InMemorySaver` checkpointer (from LangGraph). The `thread_id` is set to `session_id` on every `agent.invoke()` call, so:

- Conversation history is preserved **within a session** across multiple `/chat` calls.
- Different sessions are fully isolated — no cross-contamination.
- Memory is cleared automatically when the session is deleted or expires.

---

## Project Structure

```
rag_backend/
├── src/
│   ├── main.py                     # FastAPI app + startup
│   ├── config.py                   # Settings (pydantic-settings + .env)
│   ├── schemas/
│   │   └── responses.py            # TypedDict LLM schemas + Pydantic HTTP models
│   ├── loaders/
│   │   └── universal_loader.py     # PDF / DOCX / image loader
│   ├── chunking/
│   │   └── smart_chunker.py        # Semantic + recursive chunking
│   ├── embeddings/
│   │   ├── embedding_model.py      # HuggingFaceEmbeddings singleton
│   │   └── llm_model.py            # ChatHuggingFace singleton
│   ├── vectorstore/
│   │   └── session_store.py        # Per-session FAISS registry
│   ├── memory/
│   │   └── session_registry.py     # Session metadata + TTL expiry
│   ├── agents/
│   │   └── rag_agent.py            # LangGraph agent (memory + structured output)
│   └── api/
│       ├── router.py               # All FastAPI endpoints
│       └── upload_service.py       # Zip → ingest pipeline
├── requirements.txt
├── .env.example
└── README.md
```