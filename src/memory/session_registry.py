"""
Session registry — persistent metadata + RAM index.

What is persisted
-----------------
Each session gets a folder on disk:

    {FAISS_INDEX_DIR}/{session_id}/
        index.faiss      ← FAISS binary index  (managed by session_store.py)
        index.pkl        ← docstore + id map    (managed by session_store.py)
        meta.json        ← session metadata     (managed here)

meta.json schema
----------------
{
    "session_id":      "uuid-string",
    "session_name":    "my-project",       ← optional, human label for lookup
    "files_processed": ["doc_a.pdf", "doc_b.docx"],
    "chunks_indexed":  142,
    "created_at":      "2025-03-17T10:30:00+00:00",
    "last_active":     "2025-03-17T11:02:45+00:00"
}

Re-accessing embeddings later
------------------------------
A user has two options to find their session again:

  Option 1 — by session_id (UUID):
    Store the session_id returned at upload time (e.g. in localStorage or a DB).
    Use POST /sessions/{session_id}/restore to bring it back into RAM.

  Option 2 — by session_name (human label):
    Pass session_name="my-project" at upload time.
    Later call GET /sessions/lookup?name=my-project to get the session_id back.
    Then POST /sessions/{session_id}/restore if needed.

TTL behaviour
-------------
TTL only evicts from RAM. Disk (index + meta.json) is NEVER touched by TTL.
Explicit DELETE /sessions/{id} removes RAM + disk permanently.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from langgraph.checkpoint.memory import InMemorySaver

from src.config.config import settings
from src.vectorstore.session_store import (
    delete_session_store,
    evict_session_store,  # Fix 1: clean public API instead of internal import
    list_persisted_session_ids,
    load_session_store,  # Fix 2: eager FAISS reload on restore
    session_store_exists,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _meta_path(session_id: str) -> Optional[Path]:
    if not settings.FAISS_INDEX_DIR:
        return None
    return Path(settings.FAISS_INDEX_DIR) / session_id / "meta.json"


def _save_meta(meta: "SessionMeta") -> None:
    path = _meta_path(meta.session_id)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "session_id": meta.session_id,
                "session_name": meta.session_name,
                "files_processed": meta.files_processed,
                "chunks_indexed": meta.chunks_indexed,
                "created_at": meta.created_at.isoformat(),
                "last_active": meta.last_active.isoformat(),
            },
            f,
            indent=2,
        )


def _delete_meta(session_id: str) -> None:
    path = _meta_path(session_id)
    if path and path.exists():
        path.unlink()


def _load_meta_from_disk(session_id: str) -> Optional[dict]:
    path = _meta_path(session_id)
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read meta.json for session %s: %s", session_id, exc)
        return None


# ── SessionMeta ───────────────────────────────────────────────────────────────


class SessionMeta:
    __slots__ = (
        "session_id",
        "session_name",
        "files_processed",
        "chunks_indexed",
        "created_at",
        "last_active",
        "checkpointer",
    )

    def __init__(
        self,
        session_id: str,
        files_processed: List[str],
        chunks_indexed: int,
        session_name: Optional[str] = None,
        created_at: Optional[datetime] = None,
        last_active: Optional[datetime] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        self.session_id = session_id
        self.session_name = session_name or None  # None if not supplied
        self.files_processed = list(files_processed)
        self.chunks_indexed = chunks_indexed
        self.created_at = created_at or now
        self.last_active = last_active or now
        # Fresh InMemorySaver every time a session enters RAM.
        # Chat history is NOT persisted — it resets when the server restarts.
        self.checkpointer = InMemorySaver()

    def touch(self) -> None:
        self.last_active = datetime.now(timezone.utc)
        _save_meta(self)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "files_processed": self.files_processed,
            "chunks_indexed": self.chunks_indexed,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionMeta":
        return cls(
            session_id=d["session_id"],
            session_name=d.get("session_name"),
            files_processed=d.get("files_processed", []),
            chunks_indexed=d.get("chunks_indexed", 0),
            created_at=datetime.fromisoformat(d["created_at"]),
            last_active=datetime.fromisoformat(d["last_active"]),
        )


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, SessionMeta] = {}


# ── Public API ────────────────────────────────────────────────────────────────


def create_session(
    files_processed: List[str],
    chunks_indexed: int,
    session_name: Optional[str] = None,
) -> SessionMeta:
    session_id = str(uuid.uuid4())
    meta = SessionMeta(
        session_id=session_id,
        session_name=session_name,
        files_processed=files_processed,
        chunks_indexed=chunks_indexed,
    )
    _REGISTRY[session_id] = meta
    _save_meta(meta)
    logger.info(
        "Session created: %s (name=%r, %d files, %d chunks)",
        session_id,
        session_name,
        len(files_processed),
        chunks_indexed,
    )
    return meta


def append_to_session(
    session_id: str,
    new_files: List[str],
    new_chunks: int,
) -> SessionMeta:
    """Append new files/chunks to an existing session's metadata and persist."""
    meta = _REGISTRY.get(session_id)
    if meta is None:
        raise ValueError(f"Session '{session_id}' not found or has expired.")

    existing = set(meta.files_processed)
    truly_new = [f for f in new_files if f not in existing]
    meta.files_processed.extend(truly_new)
    meta.chunks_indexed += new_chunks
    meta.touch()

    logger.info(
        "Session %s appended: +%d files (%d new), +%d chunks → total %d chunks.",
        session_id,
        len(new_files),
        len(truly_new),
        new_chunks,
        meta.chunks_indexed,
    )
    return meta


def get_session(session_id: str) -> Optional[SessionMeta]:
    meta = _REGISTRY.get(session_id)
    if meta is not None:
        meta.touch()
    return meta


def lookup_session_by_name(name: str) -> Optional[SessionMeta]:
    """
    Find an active (in-RAM) session by its human-readable session_name.
    Case-insensitive match.
    """
    name_lower = name.strip().lower()
    for meta in _REGISTRY.values():
        if meta.session_name and meta.session_name.lower() == name_lower:
            return meta
    return None


def lookup_session_on_disk_by_name(name: str) -> Optional[str]:
    """
    Scan all on-disk meta.json files and return the session_id whose
    session_name matches *name* (case-insensitive).

    Used when the session has been evicted from RAM — caller should then
    call restore_session_from_disk(session_id) to bring it back.
    """
    if not settings.FAISS_INDEX_DIR:
        return None
    root = Path(settings.FAISS_INDEX_DIR)
    if not root.exists():
        return None

    name_lower = name.strip().lower()
    for sid in list_persisted_session_ids():
        raw = _load_meta_from_disk(sid)
        if raw and raw.get("session_name", "").lower() == name_lower:
            return sid
    return None


def delete_session(session_id: str) -> bool:
    """Delete a session fully — RAM registry, FAISS index, and meta.json."""
    meta = _REGISTRY.pop(session_id, None)
    vs_deleted = delete_session_store(session_id)
    _delete_meta(session_id)

    if meta is not None or vs_deleted:
        logger.info("Session %s deleted (all data removed).", session_id)
        return True
    return False


def restore_session_from_disk(session_id: str) -> Optional[SessionMeta]:
    """
    Load a single session from disk into the in-process registry.

    Fix 2: The FAISS index is now eagerly loaded into RAM here — previously
    it was only lazy-loaded on the first retrieval call, which made the session
    appear "active" in the registry but caused silent failures if the embeddings
    weren't threaded through correctly. Eager loading makes the restore
    atomic and easy to reason about: either the whole session is ready, or
    it isn't.
    """
    if session_id in _REGISTRY:
        return _REGISTRY[session_id]

    raw = _load_meta_from_disk(session_id)
    if raw is None:
        logger.warning("No meta.json found for session %s on disk.", session_id)
        return None

    if not session_store_exists(session_id):
        logger.warning(
            "meta.json found for session %s but FAISS index is missing.", session_id
        )
        return None

    meta = SessionMeta.from_dict(raw)
    _REGISTRY[session_id] = meta

    # Eagerly load the FAISS index into RAM so the session is fully ready.
    # Importing here avoids a circular import at module level:
    #   session_registry → session_store → (no back-reference needed)
    #   session_registry → embedding_model → config  (safe)
    from src.embeddings.embedding_model import get_embedding_model  # noqa: PLC0415

    load_session_store(session_id, get_embedding_model())

    logger.info(
        "Session %s (name=%r) restored from disk into registry (FAISS index loaded).",
        session_id,
        meta.session_name,
    )
    return meta


def restore_sessions_from_disk() -> int:
    """Scan FAISS_INDEX_DIR and restore all valid sessions at server startup."""
    ids = list_persisted_session_ids()
    count = 0
    for sid in ids:
        if restore_session_from_disk(sid) is not None:
            count += 1
    logger.info("Startup: restored %d session(s) from disk.", count)
    return count


def list_sessions() -> List[dict]:
    return [m.to_dict() for m in _REGISTRY.values()]


def maybe_expire_sessions() -> None:
    """
    Evict sessions idle longer than SESSION_TTL_SECONDS from RAM.
    Disk data (FAISS index + meta.json) is NEVER deleted by TTL.

    Fix 1: uses the public evict_session_store() instead of importing
    _STORE_REGISTRY directly inside the function body, which was fragile
    (silent no-op if the import failed or was cached differently).
    """
    now = datetime.now(timezone.utc)
    expired = [
        sid
        for sid, meta in _REGISTRY.items()
        if (now - meta.last_active).total_seconds() > settings.SESSION_TTL_SECONDS
    ]
    for sid in expired:
        logger.info("Session %s evicted from RAM (TTL). Disk index preserved.", sid)
        _REGISTRY.pop(sid, None)
        evict_session_store(sid)  # Fix 1: clean public call, no internal import hack
