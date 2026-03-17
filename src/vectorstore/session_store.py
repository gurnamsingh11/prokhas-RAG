"""
In-memory FAISS vector store — one store per session.

Lifecycle
---------
* Created when a session is first populated (upload flow).
* Deleted from the registry when the session is deleted or expires.
* Because the store lives only in RAM, it is automatically reclaimed when the
  process exits.

Future: persistent sessions
----------------------------
If you want embeddings to survive server restarts (and be reusable by
session name), replace the in-memory dict with a persistent store:

    1. Use `FAISS.save_local(path)` / `FAISS.load_local(path, embeddings)`
       keyed by session_id.
    2. Keep the registry dict but populate it lazily from disk.
    3. Remove the `FAISS.save_local` call from `delete_session_store` so the
       index file is NOT erased on deletion.

That is the only change required — the rest of the codebase is unaffected.
"""

import logging
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)

# session_id -> FAISS store
_STORE_REGISTRY: Dict[str, FAISS] = {}


def build_session_store(
    session_id: str,
    chunks: List[Document],
    embeddings: Embeddings,
) -> FAISS:
    """
    Embed *chunks* and store them in an in-memory FAISS index keyed by
    *session_id*.  Any previous store for the same session is replaced.
    """
    logger.info(
        "Building FAISS store for session %s with %d chunks.",
        session_id,
        len(chunks),
    )
    store = FAISS.from_documents(chunks, embeddings)
    _STORE_REGISTRY[session_id] = store
    return store


def add_to_session_store(
    session_id: str,
    chunks: List[Document],
    embeddings: Embeddings,
) -> int:
    """
    Embed *chunks* and merge them into an existing FAISS index for *session_id*.

    If no store exists yet for the session, a new one is created (same as
    build_session_store).  This lets callers always use this function for
    the "add more documents" flow without needing to check first.

    Returns the total number of vectors now in the index.
    """
    existing = _STORE_REGISTRY.get(session_id)

    if existing is None:
        logger.info(
            "No existing store for session %s — creating new one with %d chunks.",
            session_id,
            len(chunks),
        )
        store = FAISS.from_documents(chunks, embeddings)
        _STORE_REGISTRY[session_id] = store
    else:
        logger.info(
            "Merging %d new chunks into existing store for session %s.",
            len(chunks),
            session_id,
        )
        # FAISS.add_documents embeds and appends vectors in-place
        existing.add_documents(chunks)
        store = existing

    total = store.index.ntotal
    logger.info("Session %s store now has %d vectors.", session_id, total)
    return total


def get_session_retriever(
    session_id: str,
    top_k: int = 5,
) -> Optional[VectorStoreRetriever]:
    """Return a retriever for *session_id*, or None if the session is unknown."""
    store = _STORE_REGISTRY.get(session_id)
    if store is None:
        return None
    return store.as_retriever(search_kwargs={"k": top_k})


def delete_session_store(session_id: str) -> bool:
    """
    Remove the FAISS store for *session_id* from the registry.

    Returns True if a store was found and deleted, False otherwise.

    NOTE ── To make embeddings persistent (reusable after deletion), remove
    this function's body and replace with a no-op, or save to disk before
    popping from the registry.
    """
    store = _STORE_REGISTRY.pop(session_id, None)
    if store is not None:
        logger.info("FAISS store for session %s deleted.", session_id)
        # store.__del__() is called automatically by GC; no explicit cleanup needed
        return True
    return False


def session_store_exists(session_id: str) -> bool:
    return session_id in _STORE_REGISTRY
