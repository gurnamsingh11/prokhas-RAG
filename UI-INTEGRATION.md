# RAG Backend — UI Integration Guide

> **Audience:** Frontend / UI developers integrating this API into a web application.  
> **Base URL:** `http://localhost:8000` (configurable via `.env`)  
> **API prefix:** All endpoints live under `/api/v1/`  
> **Interactive docs:** `http://localhost:8000/docs` (Swagger UI)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Core Concepts](#2-core-concepts)
3. [Endpoint Reference](#3-endpoint-reference)
   - [Health Check](#31-health-check)
   - [Create Session (Upload ZIP)](#32-create-session--upload-zip)
   - [Add ZIP to Existing Session](#33-add-zip-to-existing-session)
   - [Chat](#34-chat)
   - [Get Session Info](#35-get-session-info)
   - [List All Sessions](#36-list-all-sessions)
   - [Delete Session](#37-delete-session)
4. [Error Handling](#4-error-handling)
5. [Typical UI Flows](#5-typical-ui-flows)
6. [TypeScript Type Definitions](#6-typescript-type-definitions)
7. [CORS & Headers](#7-cors--headers)
8. [Session Lifecycle & TTL](#8-session-lifecycle--ttl)
9. [File & ZIP Requirements](#9-file--zip-requirements)

---

## 1. Quick Start

```bash
# Start the backend
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Three-step integration from any frontend:

```js
// Step 1 — Upload a zip, get a session_id back
const { session_id } = await uploadZip(zipFile);

// Step 2 — Ask questions against the uploaded documents
const { answer, sources } = await chat(session_id, "What is this document about?");

// Step 3 — Clean up when done
await deleteSession(session_id);
```

---

## 2. Core Concepts

### Session
A **session** is the unit of work. Each session has:
- Its own isolated vector store (FAISS index in memory)
- Its own conversation history (short-term memory)
- A UUID `session_id` that the frontend must persist to make further calls

Sessions expire automatically after **1 hour of inactivity** (configurable server-side). Once expired, the `session_id` is no longer valid — any call with it returns `404`.

### ZIP Upload
Documents must be sent inside a `.zip` archive. The backend extracts, parses, chunks, and embeds them automatically. Supported file types inside the zip:

| Type | Extensions |
|------|-----------|
| PDF | `.pdf` |
| Word | `.docx` |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` |

### Multi-ZIP Sessions
You can upload **multiple ZIP files into the same session** using the append endpoint. Each new zip is merged into the existing vector store. Conversation history is preserved between uploads.

---

## 3. Endpoint Reference

---

### 3.1 Health Check

Useful for showing a server-online indicator in the UI.

```
GET /health
```

**Response `200`**
```json
{ "status": "ok" }
```

**Example**
```js
async function isServerOnline() {
  try {
    const res = await fetch('http://localhost:8000/health');
    return res.ok;
  } catch {
    return false;
  }
}
```

---

### 3.2 Create Session — Upload ZIP

Uploads a `.zip` archive, indexes all documents inside it, and returns a new `session_id`.

```
POST /api/v1/sessions/upload
Content-Type: multipart/form-data
```

**Request**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File (`.zip`) | ✅ | The zip archive to index |

**Response `201 Created`**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Session created. You can now chat using the session_id.",
  "files_processed": 3,
  "chunks_indexed": 142
}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `string` (UUID) | **Store this.** Required for all subsequent calls |
| `message` | `string` | Human-readable status |
| `files_processed` | `number` | How many files were successfully loaded |
| `chunks_indexed` | `number` | How many text chunks were embedded |

**Example**
```js
async function createSession(zipFile) {
  const form = new FormData();
  form.append('file', zipFile, zipFile.name);   // field name must be "file"

  const res = await fetch('http://localhost:8000/api/v1/sessions/upload', {
    method: 'POST',
    body: form,
    // ⚠️ Do NOT set Content-Type manually — the browser sets it with the boundary
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail);
  }

  return await res.json();  // { session_id, message, files_processed, chunks_indexed }
}
```

**Error responses**

| Status | When |
|--------|------|
| `400 Bad Request` | File is not a `.zip` |
| `422 Unprocessable Entity` | Zip contains no supported files, or all files were empty |
| `500 Internal Server Error` | Unexpected server error during indexing |

---

### 3.3 Add ZIP to Existing Session

Uploads an **additional** `.zip` into an already-existing session. New documents are merged into the same vector store. The chat history is NOT reset — users can immediately ask questions that span all uploaded files.

```
POST /api/v1/sessions/{session_id}/upload
Content-Type: multipart/form-data
```

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `string` | The existing session to append to |

**Request**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File (`.zip`) | ✅ | The zip archive to merge |

**Response `200 OK`**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Successfully added 2 file(s) and 47 chunk(s) to the session.",
  "files_added": 2,
  "chunks_added": 47,
  "total_files": 5,
  "total_chunks": 189
}
```

| Field | Type | Description |
|-------|------|-------------|
| `files_added` | `number` | Files added by this upload (deduplicated) |
| `chunks_added` | `number` | New chunks embedded by this upload |
| `total_files` | `number` | Total files in the session after merge |
| `total_chunks` | `number` | Total chunks in the session after merge |

**Example**
```js
async function addZipToSession(sessionId, zipFile) {
  const form = new FormData();
  form.append('file', zipFile, zipFile.name);

  const res = await fetch(
    `http://localhost:8000/api/v1/sessions/${sessionId}/upload`,
    { method: 'POST', body: form }
  );

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail);
  }

  return await res.json();  // { session_id, files_added, chunks_added, total_files, total_chunks }
}
```

**Error responses**

| Status | When |
|--------|------|
| `400 Bad Request` | File is not a `.zip` |
| `404 Not Found` | `session_id` does not exist or has expired |
| `422 Unprocessable Entity` | Zip contains no supported files |
| `500 Internal Server Error` | Unexpected server error |

---

### 3.4 Chat

Sends a user message and returns a grounded answer with source attribution. Each call is stateful — the backend remembers the full conversation history for the session.

```
POST /api/v1/sessions/{session_id}/chat
Content-Type: application/json
```

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `string` | Active session to query |

**Request Body**
```json
{
  "message": "What does the document say about Q3 revenue?"
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|------------|
| `message` | `string` | ✅ | Must not be empty or whitespace-only |

**Response `200 OK`**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer": "According to the Q3 report, total revenue was $4.2 billion, representing a 12% year-over-year increase.",
  "sources": [
    {
      "document_name": "q3_report.pdf",
      "page_label": "4",
      "relevant_excerpt": "Total revenue for Q3 was $4.2 billion, up 12% year-over-year."
    },
    {
      "document_name": "finance_summary.docx",
      "page_label": "",
      "relevant_excerpt": "Q3 operating expenses decreased by 5% due to cost optimisation."
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `string` | Echoes back the session ID |
| `answer` | `string` | The full answer from the model |
| `sources` | `Source[]` | Array of source references (may be empty if no relevant docs found) |
| `sources[].document_name` | `string` | Filename of the source document (e.g. `"report.pdf"`) |
| `sources[].page_label` | `string` | Page number/label if available, empty string `""` otherwise |
| `sources[].relevant_excerpt` | `string` | Short snippet from the source chunk (≤ 200 characters) |

**Example**
```js
async function sendMessage(sessionId, message) {
  const res = await fetch(
    `http://localhost:8000/api/v1/sessions/${sessionId}/chat`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    }
  );

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail);
  }

  return await res.json();  // { session_id, answer, sources[] }
}
```

**UI rendering tips**
- `sources` may be an empty array `[]` — always guard before mapping
- `page_label` can be `""` — omit the page badge in that case
- `relevant_excerpt` is ≤ 200 chars — safe to render as a single line quote
- The model preserves conversation context — follow-up questions like *"Can you elaborate?"* work without re-sending context

**Error responses**

| Status | When |
|--------|------|
| `400 Bad Request` | `message` is empty or whitespace |
| `404 Not Found` | `session_id` does not exist or has expired |
| `422 Unprocessable Entity` | Missing `message` field in body |
| `500 Internal Server Error` | Model or retrieval error |

---

### 3.5 Get Session Info

Returns metadata for a single session.

```
GET /api/v1/sessions/{session_id}
```

**Response `200 OK`**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "files_processed": ["report.pdf", "summary.docx", "diagram.png"],
  "chunks_indexed": 142,
  "created_at": "2025-03-17T10:30:00.123456+00:00",
  "last_active": "2025-03-17T11:02:45.654321+00:00"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `files_processed` | `string[]` | List of filenames successfully indexed |
| `chunks_indexed` | `number` | Total number of indexed chunks |
| `created_at` | `string` (ISO 8601) | When the session was first created |
| `last_active` | `string` (ISO 8601) | Last time the session was touched (upload or chat) |

**Example**
```js
async function getSession(sessionId) {
  const res = await fetch(
    `http://localhost:8000/api/v1/sessions/${sessionId}`
  );
  if (res.status === 404) return null;
  return await res.json();
}
```

**Error responses**

| Status | When |
|--------|------|
| `404 Not Found` | Session does not exist or has expired |

---

### 3.6 List All Sessions

Returns an array of all currently active sessions. Useful for building a session picker or sidebar.

```
GET /api/v1/sessions
```

**Response `200 OK`**
```json
[
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "files_processed": ["report.pdf", "notes.docx"],
    "chunks_indexed": 98,
    "created_at": "2025-03-17T10:30:00.123456+00:00",
    "last_active": "2025-03-17T11:02:45.654321+00:00"
  },
  {
    "session_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
    "files_processed": ["manual.pdf"],
    "chunks_indexed": 210,
    "created_at": "2025-03-17T09:15:22.000000+00:00",
    "last_active": "2025-03-17T09:50:00.000000+00:00"
  }
]
```

Returns an empty array `[]` when no sessions are active — never `null`.

**Example**
```js
async function listSessions() {
  const res = await fetch('http://localhost:8000/api/v1/sessions');
  return await res.json();  // always an array
}
```

---

### 3.7 Delete Session

Permanently deletes a session and frees its vector store from memory. Any subsequent call using this `session_id` will return `404`.

```
DELETE /api/v1/sessions/{session_id}
```

**Response `200 OK`**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "deleted": true,
  "message": "Session and embeddings deleted successfully."
}
```

**Example**
```js
async function deleteSession(sessionId) {
  const res = await fetch(
    `http://localhost:8000/api/v1/sessions/${sessionId}`,
    { method: 'DELETE' }
  );

  if (res.status === 404) {
    console.warn('Session already gone');
    return;
  }

  return await res.json();  // { session_id, deleted: true, message }
}
```

**Error responses**

| Status | When |
|--------|------|
| `404 Not Found` | Session does not exist (already deleted or expired) |

---

## 4. Error Handling

All error responses follow the same shape:

```json
{
  "detail": "Human-readable description of what went wrong."
}
```

**Recommended pattern**
```js
async function apiFetch(url, options = {}) {
  const res = await fetch(url, options);

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }

  return res.json();
}
```

**Status code summary**

| Code | Meaning | Common cause |
|------|---------|--------------|
| `200` | OK | Successful GET / DELETE / append upload |
| `201` | Created | New session created |
| `400` | Bad Request | Wrong file type, empty message |
| `404` | Not Found | Session expired or never existed |
| `405` | Method Not Allowed | Wrong HTTP verb on a route |
| `422` | Unprocessable Entity | Missing required field, empty zip, validation error |
| `500` | Internal Server Error | Model failure, unexpected exception |

---

## 5. Typical UI Flows

### Flow A — New conversation

```
User picks a ZIP file
        │
        ▼
POST /sessions/upload  ──► store session_id in component state / localStorage
        │
        ▼
Show chat interface
        │
User types a message
        │
        ▼
POST /sessions/{session_id}/chat  ──► render answer + sources
        │
User types another message (conversation continues)
        │
        ▼
POST /sessions/{session_id}/chat  ──► backend remembers prior messages automatically
```

### Flow B — Adding more documents mid-conversation

```
Active session exists (user already chatting)
        │
User drops a second ZIP
        │
        ▼
POST /sessions/{session_id}/upload  ──► returns { files_added, chunks_added, total_files, total_chunks }
        │
Update file list and chunk counters in UI
        │
User continues chatting — new docs are immediately searchable
        │
        ▼
POST /sessions/{session_id}/chat  (same session, now has more context)
```

### Flow C — Resuming a previous session

```
Page loads
        │
        ▼
GET /sessions  ──► render list of past sessions
        │
User clicks a session
        │
        ▼
GET /sessions/{session_id}  ──► load file list and metadata
        │
Show chat interface (message history is on the server, not sent to UI on load)
        │
User types a message
        │
        ▼
POST /sessions/{session_id}/chat  ──► server remembers the conversation automatically
```

### Flow D — Cleanup

```
User closes the conversation
        │
        ▼
DELETE /sessions/{session_id}  ──► frees server memory immediately

(alternatively, let TTL expire automatically after 1 hour of inactivity)
```

---

## 6. TypeScript Type Definitions

Copy these directly into your project:

```typescript
// ── Requests ──────────────────────────────────────────────────────────────

interface ChatRequest {
  message: string;
}

// ── Responses ─────────────────────────────────────────────────────────────

interface HealthResponse {
  status: "ok";
}

interface SessionCreatedResponse {
  session_id: string;
  message: string;
  files_processed: number;   // count of files indexed
  chunks_indexed: number;
}

interface ZipAddedResponse {
  session_id: string;
  message: string;
  files_added: number;       // new files from this upload
  chunks_added: number;      // new chunks from this upload
  total_files: number;       // session total after merge
  total_chunks: number;      // session total after merge
}

interface SessionInfo {
  session_id: string;
  files_processed: string[]; // list of filenames
  chunks_indexed: number;
  created_at: string;        // ISO 8601
  last_active: string;       // ISO 8601
}

interface Source {
  document_name: string;     // e.g. "report.pdf"
  page_label: string;        // page number, or "" if unavailable
  relevant_excerpt: string;  // ≤ 200 chars
}

interface ChatResponse {
  session_id: string;
  answer: string;
  sources: Source[];         // may be empty []
}

interface DeleteResponse {
  session_id: string;
  deleted: boolean;
  message: string;
}

// ── Error ─────────────────────────────────────────────────────────────────

interface ApiError {
  detail: string;
}
```

---

## 7. CORS & Headers

CORS is **fully open** on the backend (`allow_origins: ["*"]`). No `Origin` or credential headers are required from the frontend.

For all requests the only required header is:

| Endpoint | Required header |
|----------|----------------|
| `POST .../chat` | `Content-Type: application/json` |
| `POST .../upload` | **None** — let the browser set `multipart/form-data` with boundary automatically |

> ⚠️ **Never manually set `Content-Type: multipart/form-data`** on upload requests. The boundary parameter is added automatically by the browser when you pass a `FormData` object. Setting it manually will break the request.

---

## 8. Session Lifecycle & TTL

| Event | Effect on session |
|-------|------------------|
| `POST /sessions/upload` | Session created, TTL clock starts |
| `POST /sessions/{id}/upload` | `last_active` refreshed |
| `GET /sessions/{id}` | `last_active` refreshed |
| `POST /sessions/{id}/chat` | `last_active` refreshed |
| Idle for **1 hour** | Session auto-expired on next request to the API |
| `DELETE /sessions/{id}` | Immediately deleted, memory freed |

**Implication for UI:** If a user leaves a tab open for more than an hour without interaction, the next chat or upload call will return `404`. Handle this gracefully:

```js
async function chatWithFallback(sessionId, message, onSessionExpired) {
  try {
    return await sendMessage(sessionId, message);
  } catch (err) {
    if (err.message.includes('404') || err.message.toLowerCase().includes('not found')) {
      onSessionExpired();  // prompt user to re-upload
    }
    throw err;
  }
}
```

---

## 9. File & ZIP Requirements

| Requirement | Detail |
|-------------|--------|
| Archive format | Must be a valid `.zip` file |
| File field name | Must be `file` in the `FormData` |
| Supported file types inside zip | `.pdf`, `.docx`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` |
| Unsupported types | Silently skipped (`.txt`, `.csv`, `.xlsx`, etc.) |
| Empty zip | Returns `422` |
| Zip with only unsupported files | Returns `422` |
| Nested folders inside zip | ✅ Supported — the backend walks all subdirectories |
| Multiple zips per session | ✅ Use the append endpoint |
| Duplicate filenames across zips | Filenames are deduplicated in metadata; vectors are always added |
| Max file size | No hard limit set — constrained by server RAM and processing time |

---

*For questions about the backend internals, see `README.md`. For running tests, see `tests/test_api.py`.*