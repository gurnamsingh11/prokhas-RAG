# Session Persistence — Integration Guide

> **Who is this for?**  
> Integration and backend developers who need to understand how to upload documents, chat against them, come back to the same session later (with the same pre-built embeddings), and clean up when done.

---

## Table of Contents

1. [How Persistence Works — The 30-Second Version](#1-how-persistence-works--the-30-second-version)
2. [What Gets Stored and Where](#2-what-gets-stored-and-where)
3. [Scenario A — Without a Name (UUID-only)](#3-scenario-a--without-a-name-uuid-only)
4. [Scenario B — With a Name (human-readable label)](#4-scenario-b--with-a-name-human-readable-label)
5. [Adding More Documents to the Same Session](#5-adding-more-documents-to-the-same-session)
6. [Server Restart — What Happens Automatically](#6-server-restart--what-happens-automatically)
7. [Session Lifecycle & TTL](#7-session-lifecycle--ttl)
8. [Deleting a Session](#8-deleting-a-session)
9. [Checking What's on Disk](#9-checking-whats-on-disk)
10. [Decision Guide — Name vs No Name](#10-decision-guide--name-vs-no-name)
11. [Common Mistakes](#11-common-mistakes)

---

## 1. How Persistence Works — The 30-Second Version

When a user uploads a ZIP, the server:
1. Extracts the documents, chunks them, and builds a FAISS vector index.
2. Saves the index to disk under `./faiss_store/{session_id}/`.
3. Saves session metadata (file list, chunk count, timestamps, and optional name) to `./faiss_store/{session_id}/meta.json`.
4. Returns a `session_id` (UUID) to the caller.

From that point on the embeddings live **permanently on disk** until an explicit `DELETE` call. Server restarts, RAM evictions (TTL), and process crashes do not erase them. The same user can come back days later, restore the session in one API call, and resume chatting — no re-uploading, no re-embedding.

---

## 2. What Gets Stored and Where

```
./faiss_store/                          ← configured via FAISS_INDEX_DIR in .env
  550e8400-e29b-41d4-a716-446655440000/ ← one folder per session (named by UUID)
    index.faiss                         ← raw FAISS binary index
    index.pkl                           ← docstore + vector-to-document id map
    meta.json                           ← human-readable session metadata
```

**meta.json example:**

```json
{
  "session_id":      "550e8400-e29b-41d4-a716-446655440000",
  "session_name":    "q3-financial-reports",
  "files_processed": ["q3_report.pdf", "budget_summary.docx"],
  "chunks_indexed":  142,
  "created_at":      "2025-03-17T10:30:00+00:00",
  "last_active":     "2025-03-17T11:02:45+00:00"
}
```

> **`session_name` is optional.** If the user didn't supply one, this field is `null`.

---

## 3. Scenario A — Without a Name (UUID-only)

Use this when your app manages state itself (e.g. stores the `session_id` in a database or `localStorage`).

### Step 1 — Upload documents

```http
POST /api/v1/sessions/upload
Content-Type: multipart/form-data

file = <your_zip_file.zip>
```

**Response `201 Created`:**
```json
{
  "session_id":      "550e8400-e29b-41d4-a716-446655440000",
  "session_name":    null,
  "message":         "Session created. You can now chat using the session_id.",
  "files_processed": 2,
  "chunks_indexed":  142
}
```

> ⚠️ **Store the `session_id` now.** This is the only identifier for this session when no name is given.

```js
// Example: store in localStorage after upload
const { session_id } = await uploadZip(zipFile);
localStorage.setItem('rag_session_id', session_id);
```

---

### Step 2 — Chat

```http
POST /api/v1/sessions/550e8400-e29b-41d4-a716-446655440000/chat
Content-Type: application/json

{ "message": "What does the Q3 report say about revenue?" }
```

**Response `200 OK`:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer":     "According to the Q3 report, revenue was $4.2 billion...",
  "sources": [
    {
      "document_name":    "q3_report.pdf",
      "page_label":       "4",
      "relevant_excerpt": "Total revenue for Q3 was $4.2 billion, up 12% YoY."
    }
  ]
}
```

---

### Step 3 — Come back later (same day, next day, after server restart)

When the user returns, retrieve the stored `session_id` and check if it's still active:

```js
const session_id = localStorage.getItem('rag_session_id');
```

**If the session is still active in RAM** (server hasn't restarted, TTL hasn't expired):

```http
GET /api/v1/sessions/550e8400-e29b-41d4-a716-446655440000
```
→ Returns `200` with session info. Go straight to Step 2 (chat).

---

**If the server restarted or the session was evicted from RAM** (returns `404` on the GET above):

Restore it from disk with one call:

```http
POST /api/v1/sessions/550e8400-e29b-41d4-a716-446655440000/restore
```

**Response `200 OK`:**
```json
{
  "session_id":      "550e8400-e29b-41d4-a716-446655440000",
  "session_name":    null,
  "message":         "Session restored from disk. You can now chat immediately.",
  "files_processed": ["q3_report.pdf", "budget_summary.docx"],
  "chunks_indexed":  142,
  "created_at":      "2025-03-17T10:30:00+00:00",
  "last_active":     "2025-03-18T09:15:00+00:00"
}
```

> The FAISS index is loaded lazily on the first chat call — restore itself is instant.

Now chat again as normal (Step 2). **No re-uploading. No re-embedding.**

---

### Recommended pattern for Scenario A

```js
async function getOrRestoreSession(sessionId) {
  // Try GET first
  let res = await fetch(`/api/v1/sessions/${sessionId}`);

  if (res.status === 404) {
    // Session evicted — restore from disk
    res = await fetch(`/api/v1/sessions/${sessionId}/restore`, { method: 'POST' });
    if (!res.ok) throw new Error('Session not found on disk. User must re-upload.');
  }

  return await res.json();
}
```

---

## 4. Scenario B — With a Name (human-readable label)

Use this when your users need to identify their sessions by a meaningful label rather than a UUID — for example in a multi-user app, or when you don't want to manage UUID storage yourself.

### Step 1 — Upload documents with a name

```http
POST /api/v1/sessions/upload
Content-Type: multipart/form-data

file         = <your_zip_file.zip>
session_name = q3-financial-reports
```

**Response `201 Created`:**
```json
{
  "session_id":      "550e8400-e29b-41d4-a716-446655440000",
  "session_name":    "q3-financial-reports",
  "message":         "Session created. Accessible by name 'q3-financial-reports'. You can now chat using the session_id.",
  "files_processed": 2,
  "chunks_indexed":  142
}
```

You can still store the `session_id`, but you don't have to — the name is enough to get back to the session.

---

### Step 2 — Chat (same as Scenario A)

```http
POST /api/v1/sessions/550e8400-e29b-41d4-a716-446655440000/chat
Content-Type: application/json

{ "message": "Summarise the key financial highlights." }
```

---

### Step 3 — Come back later using the name

The user types "q3-financial-reports" (or your app hardcodes it). One endpoint handles everything — it finds the session in RAM or on disk, auto-restores if needed, and returns it ready to chat:

```http
GET /api/v1/sessions/lookup?name=q3-financial-reports
```

**Response `200 OK` (session was active in RAM):**
```json
{
  "session_id":      "550e8400-e29b-41d4-a716-446655440000",
  "session_name":    "q3-financial-reports",
  "files_processed": ["q3_report.pdf", "budget_summary.docx"],
  "chunks_indexed":  142,
  "created_at":      "2025-03-17T10:30:00+00:00",
  "last_active":     "2025-03-18T09:20:00+00:00",
  "status":          "active"
}
```

**Response `200 OK` (session was on disk, auto-restored):**
```json
{
  "session_id":      "550e8400-e29b-41d4-a716-446655440000",
  "session_name":    "q3-financial-reports",
  "files_processed": ["q3_report.pdf", "budget_summary.docx"],
  "chunks_indexed":  142,
  "created_at":      "2025-03-17T10:30:00+00:00",
  "last_active":     "2025-03-18T09:20:00+00:00",
  "status":          "restored_from_disk"
}
```

Both responses give you the `session_id`. Use it to chat immediately.

**Response `404 Not Found`:**
```json
{
  "detail": "No session named 'q3-financial-reports' found."
}
```
→ The session was explicitly deleted (or never created). User must re-upload.

---

### Recommended pattern for Scenario B

```js
async function getSessionByName(name) {
  const res = await fetch(`/api/v1/sessions/lookup?name=${encodeURIComponent(name)}`);

  if (res.status === 404) {
    return null;  // session doesn't exist — prompt user to upload
  }

  if (!res.ok) throw new Error(`Lookup failed: ${res.status}`);

  const session = await res.json();
  // session.status is "active" or "restored_from_disk"
  return session;  // { session_id, session_name, files_processed, ... }
}

// Usage
const session = await getSessionByName('q3-financial-reports');
if (!session) {
  // show upload UI
} else {
  // go straight to chat using session.session_id
}
```

---

### Name rules

| Rule | Detail |
|------|--------|
| Names are unique | Uploading with a name that already exists on disk returns `409 Conflict` |
| Case-insensitive | `Q3-Reports` and `q3-reports` are treated as the same name on lookup |
| Names are optional | Omitting `session_name` is always valid — just use the UUID workflow |
| Names can't be changed | Once set, a name is fixed. To rename: delete and re-upload |

---

## 5. Adding More Documents to the Same Session

A user can upload additional ZIP files into an existing session at any time. The new documents are merged into the existing FAISS index — no re-embedding of existing content.

```http
POST /api/v1/sessions/550e8400-e29b-41d4-a716-446655440000/upload
Content-Type: multipart/form-data

file = <additional_documents.zip>
```

**Response `200 OK`:**
```json
{
  "session_id":   "550e8400-e29b-41d4-a716-446655440000",
  "message":      "Successfully added 1 file(s) to the session.",
  "files_added":  1,
  "chunks_added": 47,
  "total_files":  3,
  "total_chunks": 189
}
```

- The session name and `session_id` do not change.
- Conversation history is preserved — the next chat message can reference content from all uploaded ZIPs.
- Duplicate filenames are ignored in the metadata count (but vectors are always added).

---

## 6. Server Restart — What Happens Automatically

When the server starts up, it scans the entire `FAISS_INDEX_DIR` and restores every session it finds — **zero manual action required**.

```
Server starts
    │
    ▼
Scans ./faiss_store/ for folders containing index.faiss + index.pkl + meta.json
    │
    ▼
Registers all valid sessions in RAM (metadata only — FAISS index is lazy-loaded)
    │
    ▼
Server is ready — all sessions immediately available for chat
```

The FAISS index itself is not loaded into RAM during startup — it is loaded lazily on the first chat call to that session. This keeps startup fast even with many sessions.

**From the user's perspective:** they come back after a server restart, call chat or lookup as normal, and everything works. They may notice a slightly slower first response (the FAISS index loads from disk), but after that it's full speed.

---

## 7. Session Lifecycle & TTL

Understanding when data is in RAM vs on disk is important for building reliable integrations.

```
Upload ZIP
    │
    ▼
Session created ──────────────────────────────┐
  RAM:  metadata + FAISS index in cache        │
  Disk: index.faiss + index.pkl + meta.json    │  (persisted immediately)
    │                                          │
    ▼                                          │
Chat / touch                                   │
  RAM:  last_active refreshed                  │
  Disk: meta.json updated                      │
    │                                          │
    ▼                                          │
Idle > SESSION_TTL_SECONDS (default: 1 hour)   │
  RAM:  session evicted from cache ────────────┘
  Disk: UNCHANGED — index and meta.json kept
    │
    ▼
User comes back
  Option A: POST /sessions/{id}/restore
  Option B: GET  /sessions/lookup?name=...
    │
    ▼
Session re-loaded into RAM
    │
    ▼
DELETE /sessions/{id}
  RAM:  removed
  Disk: index.faiss + index.pkl + meta.json ALL DELETED permanently
```

**Summary table:**

| Event | RAM | Disk |
|-------|-----|------|
| Upload | ✅ created | ✅ saved |
| Chat / touch | ✅ `last_active` refreshed | ✅ `meta.json` updated |
| TTL expiry | ❌ evicted | ✅ **kept** |
| Server restart | ❌ cleared | ✅ **kept** — auto-restored on startup |
| `POST .../restore` | ✅ re-loaded | unchanged |
| `GET .../lookup?name=` | ✅ auto-restored if needed | unchanged |
| `DELETE /sessions/{id}` | ❌ removed | ❌ **permanently deleted** |

---

## 8. Deleting a Session

Deletion is permanent and removes everything — RAM, FAISS index on disk, and `meta.json`.

```http
DELETE /api/v1/sessions/550e8400-e29b-41d4-a716-446655440000
```

**Response `200 OK`:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "deleted":    true,
  "message":    "Session and all associated data permanently deleted."
}
```

**Response `404 Not Found`:**
```json
{
  "detail": "Session not found."
}
```

After deletion:
- `GET /sessions/{id}` returns `404`
- `POST /sessions/{id}/restore` returns `404`
- `GET /sessions/lookup?name=<the_name>` returns `404`
- The disk folder `./faiss_store/{session_id}/` is gone

> **There is no recycle bin or undo.** The user must re-upload their documents to create a new session.

---

### Deleting by name (two-step)

There is no `DELETE /sessions?name=...` shortcut. If you only have the name:

```js
// Step 1: look up the session_id
const session = await getSessionByName('q3-financial-reports');
if (!session) return; // already gone

// Step 2: delete by session_id
await fetch(`/api/v1/sessions/${session.session_id}`, { method: 'DELETE' });
```

---

## 9. Checking What's on Disk

### List all active sessions (currently in RAM)

```http
GET /api/v1/sessions
```

```json
[
  {
    "session_id":      "550e8400-...",
    "session_name":    "q3-financial-reports",
    "files_processed": ["q3_report.pdf", "budget_summary.docx"],
    "chunks_indexed":  142,
    "created_at":      "2025-03-17T10:30:00+00:00",
    "last_active":     "2025-03-18T09:20:00+00:00"
  }
]
```

### List all persisted sessions (on disk, including RAM-evicted ones)

```http
GET /api/v1/sessions/persisted
```

```json
[
  { "session_id": "550e8400-...", "status": "active" },
  { "session_id": "7c9e6679-...", "status": "on_disk_only" }
]
```

`on_disk_only` means the session exists on disk but has been evicted from RAM (TTL, server restart). It can be restored via `POST /sessions/{id}/restore` or `GET /sessions/lookup?name=...`.

---

## 10. Decision Guide — Name vs No Name

```
Do your users need to find sessions by a human-readable label?
        │
       YES                              NO
        │                               │
Do you control your own DB              Your app already stores the
or localStorage to save the             session_id per user/workspace
session_id per user?                          │
        │                                     ▼
       YES               NO            Use Scenario A (UUID only)
        │                │             Store session_id → use restore
        ▼                ▼             endpoint when needed
  Either works     Use Scenario B
  (name is         (named sessions)
  convenience)
```

**Use named sessions when:**
- You have multiple users each uploading different document sets and need `user_id → session_name` mapping without a separate database
- Your UI shows a list of named workspaces the user can click to resume
- You want the simplest possible re-access (`lookup?name=` is one call vs UUID lookup + restore)

**Use UUID-only sessions when:**
- Your app already has a database where you store per-user `session_id`
- You generate sessions programmatically and don't need human labels
- You're building an API-to-API integration where names would be meaningless

---

## 11. Common Mistakes

### ❌ Not storing the session_id and not using a name

If you upload without a name and don't store the UUID, the session is on disk but unrecoverable by the user (they'd have to guess the UUID). Always do one of:
- Store `session_id` in your app's state
- Supply a meaningful `session_name` at upload time

---

### ❌ Using the same name twice

```http
POST /sessions/upload
session_name = my-project   ← already exists on disk
```

Returns `409 Conflict`. Each name must be globally unique across all sessions on disk. If you want to replace an old session with the same name, delete the old one first.

---

### ❌ Trying to chat with an evicted session without restoring first

```http
POST /sessions/550e8400-.../chat   ← session was TTL-evicted
{ "message": "Hello" }
```

Returns `404 Not Found` with detail:
> "Session not found or evicted from RAM. Use POST /sessions/{session_id}/restore or GET /sessions/lookup?name=... to bring it back."

Fix: call restore first, then chat.

---

### ❌ Expecting a deleted session to be restorable

`DELETE` is permanent. After deletion, both the disk folder and `meta.json` are gone. `POST .../restore` and `GET .../lookup` will return `404`. The user must re-upload.

---

### ❌ Changing the embedding model between sessions

The FAISS index stores vectors computed by a specific model. If you change `EMBEDDING_MODEL_NAME` in `.env` after sessions have been indexed, those sessions become incompatible — queries will return garbage results because the query vector dimensionality/space won't match the stored vectors.

**If you need to change the model:** delete all existing sessions and re-upload all documents.

---

*For the full API reference including request/response shapes, see `UI_INTEGRATION.md`. For backend internals, see `README.md`.*