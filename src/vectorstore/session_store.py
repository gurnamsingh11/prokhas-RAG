"""
Persistent FAISS vector store — one index per session, saved to disk.

Storage layout
--------------
{FAISS_INDEX_DIR}/
  {session_id}/
    index.faiss      ← raw FAISS binary index  (written by save_local)
    index.pkl        ← docstore + id map        (written by save_local)

Lifecycle
---------
* build_session_store()  — embeds chunks, saves to disk, caches in RAM
* add_to_session_store() — merges new chunks via merge_from(), re-saves
* get_session_retriever() — loads from RAM cache; lazy-loads from disk if needed
* delete_session_store() — removes RAM cache AND wipes the disk folder
* load_session_store()   — explicit disk → RAM load (used at startup / restore)
* evict_session_store()  — removes from RAM only, disk untouched (used by TTL)

Why merge_from instead of add_documents for appends?
------------------------------------------------------
Per the FAISS LangChain docs, FAISS.merge_from(other) merges two independent
FAISS stores including their docstores, which is safer than add_documents when
the new chunks come from a freshly built temp store (guarantees correct
docstore id mapping). We build a temp store from the new chunks then merge it.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from src.config.config import settings

logger = logging.getLogger(__name__)

# ── RAM cache: session_id → FAISS ────────────────────────────────────────────
_STORE_REGISTRY: Dict[str, FAISS] = {}


# ── Path helpers ──────────────────────────────────────────────────────────────


def _index_dir(session_id: str) -> Path:
    """Return (and create) the on-disk directory for a session's FAISS index."""
    p = Path(settings.FAISS_INDEX_DIR) / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _index_exists_on_disk(session_id: str) -> bool:
    p = Path(settings.FAISS_INDEX_DIR) / session_id
    return (p / "index.faiss").exists() and (p / "index.pkl").exists()


# ── Persistence helpers ───────────────────────────────────────────────────────


def _save(session_id: str, store: FAISS) -> None:
    """Persist *store* to disk under {FAISS_INDEX_DIR}/{session_id}/."""
    if not settings.FAISS_INDEX_DIR:
        return
    path = str(_index_dir(session_id))
    store.save_local(path)
    logger.debug("FAISS index for session %s saved to %s", session_id, path)


def _load_from_disk(session_id: str, embeddings: Embeddings) -> Optional[FAISS]:
    """
    Load a FAISS store from disk into the RAM cache.
    Returns None if no on-disk index exists.
    allow_dangerous_deserialization=True is required by LangChain when loading
    pickle-backed docstores; this is safe because we wrote the files ourselves.
    """
    if not _index_exists_on_disk(session_id):
        return None
    path = str(Path(settings.FAISS_INDEX_DIR) / session_id)
    store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _STORE_REGISTRY[session_id] = store
    logger.info("FAISS index for session %s loaded from disk.", session_id)
    return store


# ── Public API ────────────────────────────────────────────────────────────────


def build_session_store(
    session_id: str,
    chunks: List[Document],
    embeddings: Embeddings,
) -> FAISS:
    """
    Embed *chunks*, cache in RAM, and persist to disk.
    Replaces any existing store for this session.
    """
    logger.info(
        "Building FAISS store for session %s with %d chunks.", session_id, len(chunks)
    )
    store = FAISS.from_documents(chunks, embeddings)
    _STORE_REGISTRY[session_id] = store
    _save(session_id, store)
    return store


def add_to_session_store(
    session_id: str,
    chunks: List[Document],
    embeddings: Embeddings,
) -> int:
    """
    Merge new *chunks* into an existing session store using FAISS.merge_from().

    merge_from() is used (instead of add_documents) because it correctly
    reconciles the docstore id mappings when merging two independent indexes —
    exactly the pattern recommended in the LangChain FAISS docs.

    If no store exists in RAM, it is loaded from disk first.
    If no disk index exists either, a fresh store is created.

    Returns total vector count after merge.
    """
    # Build a temporary store for the new chunks
    new_store = FAISS.from_documents(chunks, embeddings)

    # Resolve existing store (RAM → disk → create new)
    existing = _STORE_REGISTRY.get(session_id)
    if existing is None:
        existing = _load_from_disk(session_id, embeddings)

    if existing is None:
        logger.info(
            "No existing store for session %s — using new store directly.", session_id
        )
        store = new_store
    else:
        logger.info(
            "Merging %d new chunks into session %s via merge_from().",
            len(chunks),
            session_id,
        )
        existing.merge_from(new_store)
        store = existing

    _STORE_REGISTRY[session_id] = store
    _save(session_id, store)

    total = store.index.ntotal
    logger.info("Session %s store now has %d vectors.", session_id, total)
    return total


def load_session_store(session_id: str, embeddings: Embeddings) -> Optional[FAISS]:
    """
    Ensure a session's FAISS index is in RAM.

    Called at startup (restore_sessions_from_disk) and by the restore endpoint.
    Returns the store if found on disk, None otherwise.
    """
    # Already in RAM
    if session_id in _STORE_REGISTRY:
        return _STORE_REGISTRY[session_id]
    return _load_from_disk(session_id, embeddings)


def get_session_retriever(
    session_id: str,
    embeddings: Optional[Embeddings] = None,
    top_k: int = 5,
) -> Optional[VectorStoreRetriever]:
    """
    Return a retriever for *session_id*.

    Tries RAM cache first; if missing, attempts a lazy disk load (requires
    *embeddings* to be provided).  Returns None if neither source has the index.
    """
    store = _STORE_REGISTRY.get(session_id)
    if store is None and embeddings is not None:
        store = _load_from_disk(session_id, embeddings)
    if store is None:
        return None
    return store.as_retriever(search_kwargs={"k": top_k})


def evict_session_store(session_id: str) -> None:
    """
    Remove a session's FAISS index from RAM only. Disk is untouched.

    Called by TTL eviction in session_registry.maybe_expire_sessions().
    This replaces the previous pattern of importing _STORE_REGISTRY directly
    inside the function body, which was fragile and hard to reason about.
    """
    _STORE_REGISTRY.pop(session_id, None)
    logger.info("Session %s FAISS index evicted from RAM (TTL).", session_id)


def delete_session_store(session_id: str) -> bool:
    """
    Remove the session store from RAM and delete its folder from disk.
    Returns True if anything was found and removed.
    """
    in_ram = _STORE_REGISTRY.pop(session_id, None) is not None
    on_disk = False

    disk_path = (
        Path(settings.FAISS_INDEX_DIR) / session_id
        if settings.FAISS_INDEX_DIR
        else None
    )
    if disk_path and disk_path.exists():
        shutil.rmtree(disk_path, ignore_errors=True)
        on_disk = True
        logger.info(
            "FAISS index directory for session %s deleted from disk.", session_id
        )

    if in_ram or on_disk:
        logger.info(
            "FAISS store for session %s deleted (RAM=%s, disk=%s).",
            session_id,
            in_ram,
            on_disk,
        )
        return True
    return False


def session_store_exists(session_id: str) -> bool:
    """True if the store is available in RAM or on disk."""
    return session_id in _STORE_REGISTRY or _index_exists_on_disk(session_id)


def list_persisted_session_ids() -> List[str]:
    """
    Scan FAISS_INDEX_DIR and return session IDs that have complete indexes on disk.
    Used at startup to discover sessions that survived a server restart.
    """
    if not settings.FAISS_INDEX_DIR:
        return []
    root = Path(settings.FAISS_INDEX_DIR)
    if not root.exists():
        return []
    return [
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "index.faiss").exists() and (p / "index.pkl").exists()
    ]
