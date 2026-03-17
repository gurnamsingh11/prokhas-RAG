"""
Session registry.

Tracks metadata for every active session and handles TTL-based expiry.
The session_id is a UUID string generated at upload time.

Data stored per session
-----------------------
* session_id
* files_processed   – list of original filenames
* chunks_indexed    – total chunk count embedded
* created_at        – ISO timestamp
* last_active       – ISO timestamp (updated on every chat call)
* short_term_memory – LangGraph InMemorySaver (one per session)

TTL expiry
----------
`maybe_expire_sessions()` is called on every API request (via a FastAPI
dependency).  Any session idle longer than SESSION_TTL_SECONDS is
auto-deleted (vector store + registry entry).
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langgraph.checkpoint.memory import InMemorySaver

from src.config.config import settings
from src.vectorstore.session_store import delete_session_store, session_store_exists

logger = logging.getLogger(__name__)


class SessionMeta:
    __slots__ = (
        "session_id",
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
    ) -> None:
        now = datetime.now(timezone.utc)
        self.session_id = session_id
        self.files_processed = list(files_processed)
        self.chunks_indexed = chunks_indexed
        self.created_at = now
        self.last_active = now
        # Each session owns its own InMemorySaver so conversation threads
        # are isolated.  The saver is passed to create_agent as `checkpointer`.
        self.checkpointer = InMemorySaver()

    def touch(self) -> None:
        self.last_active = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "files_processed": self.files_processed,
            "chunks_indexed": self.chunks_indexed,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }


_REGISTRY: Dict[str, SessionMeta] = {}


# ── Public API ────────────────────────────────────────────────────────────────


def create_session(files_processed: List[str], chunks_indexed: int) -> SessionMeta:
    session_id = str(uuid.uuid4())
    meta = SessionMeta(
        session_id=session_id,
        files_processed=files_processed,
        chunks_indexed=chunks_indexed,
    )
    _REGISTRY[session_id] = meta
    logger.info(
        "Session created: %s (%d files, %d chunks)",
        session_id,
        len(files_processed),
        chunks_indexed,
    )
    return meta


def append_to_session(
    session_id: str,
    new_files: List[str],
    new_chunks: int,
) -> SessionMeta:
    """
    Add more files and chunks to an existing session's metadata.

    Filenames are deduplicated so uploading the same zip twice does not
    double-count files (though the vectors will still be merged — that is
    caught earlier in the upload service).

    Raises ValueError if the session does not exist or has expired.
    """
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


def delete_session(session_id: str) -> bool:
    """Delete session metadata AND its vector store."""
    meta = _REGISTRY.pop(session_id, None)
    vs_deleted = delete_session_store(session_id)
    if meta is not None or vs_deleted:
        logger.info("Session %s deleted.", session_id)
        return True
    return False


def list_sessions() -> List[dict]:
    return [m.to_dict() for m in _REGISTRY.values()]


def maybe_expire_sessions() -> None:
    """Evict sessions that have been idle longer than SESSION_TTL_SECONDS."""
    now = datetime.now(timezone.utc)
    expired = [
        sid
        for sid, meta in _REGISTRY.items()
        if (now - meta.last_active).total_seconds() > settings.SESSION_TTL_SECONDS
    ]
    for sid in expired:
        logger.info("Session %s expired (TTL).", sid)
        delete_session(sid)
