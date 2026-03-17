"""
Upload service.

Two public entry points
-----------------------
ingest_zip(zip_bytes)
    Create a brand-new session from a zip file.

ingest_zip_into_session(zip_bytes, session_id)
    Append documents from a zip file to an existing session.
    The FAISS index is extended in-place; conversation history is preserved.
"""

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

from src.chunking.smart_chunker import smart_chunk_documents
from src.config.config import settings
from src.embeddings.embedding_model import get_embedding_model
from src.loaders.universal_loader import UniversalDocumentLoader
from src.memory.session_registry import (
    SessionMeta,
    append_to_session,
    create_session,
    get_session,
)
from src.vectorstore.session_store import add_to_session_store, build_session_store

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = (
    UniversalDocumentLoader.PDF_EXTENSIONS
    | UniversalDocumentLoader.WORD_EXTENSIONS
    | UniversalDocumentLoader.IMAGE_EXTENSIONS
)


# ── Public: create new session ────────────────────────────────────────────────


def ingest_zip(zip_bytes: bytes) -> SessionMeta:
    """
    Full ingest pipeline for a zip file — creates a new session.

    Returns SessionMeta for the newly created session.
    """
    chunks, filenames = _extract_and_chunk(zip_bytes)

    embedding_model = get_embedding_model()
    session_meta = create_session(
        files_processed=filenames,
        chunks_indexed=len(chunks),
    )
    build_session_store(
        session_id=session_meta.session_id,
        chunks=chunks,
        embeddings=embedding_model,
    )

    logger.info(
        "Ingest complete for NEW session %s: %d files, %d chunks.",
        session_meta.session_id,
        len(filenames),
        len(chunks),
    )
    return session_meta


# ── Public: append to existing session ───────────────────────────────────────


def ingest_zip_into_session(zip_bytes: bytes, session_id: str) -> SessionMeta:
    """
    Append documents from a zip file into an existing session.

    * Embeds new chunks and merges them into the existing FAISS index.
    * Updates session metadata (file list, chunk count).
    * Conversation history (InMemorySaver) is untouched — the chat context
      simply gains more searchable documents immediately.

    Raises ValueError if the session does not exist or has expired.
    """
    # Guard: session must exist before we do any heavy work
    meta = get_session(session_id)
    if meta is None:
        raise ValueError(f"Session '{session_id}' not found or has expired.")

    chunks, filenames = _extract_and_chunk(zip_bytes)

    embedding_model = get_embedding_model()

    # Merge new vectors into the existing FAISS index (creates one if absent)
    total_vectors = add_to_session_store(
        session_id=session_id,
        chunks=chunks,
        embeddings=embedding_model,
    )

    # Update session metadata
    updated_meta = append_to_session(
        session_id=session_id,
        new_files=filenames,
        new_chunks=len(chunks),
    )

    logger.info(
        "Append complete for session %s: +%d files, +%d chunks → %d total vectors.",
        session_id,
        len(filenames),
        len(chunks),
        total_vectors,
    )
    return updated_meta


# ── Shared pipeline: extract → load → chunk ──────────────────────────────────


def _extract_and_chunk(zip_bytes: bytes) -> Tuple[List[Document], List[str]]:
    """
    Extract a zip, load all supported documents, smart-chunk them.
    Returns (chunks, filenames).
    Raises ValueError for empty or unsupported archives.
    """
    extract_dir = tempfile.mkdtemp(prefix="rag_", dir=settings.UPLOAD_TMP_DIR)
    try:
        supported_files = _extract_zip(zip_bytes, extract_dir)
        if not supported_files:
            raise ValueError(
                "The zip archive contains no supported files "
                f"({', '.join(sorted(_SUPPORTED_EXTENSIONS))})."
            )

        raw_docs, filenames = _load_documents(supported_files)
        if not raw_docs:
            raise ValueError("All supported files produced empty documents.")

        embedding_model = get_embedding_model()
        chunks = smart_chunk_documents(
            docs=raw_docs,
            embeddings=embedding_model,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            breakpoint_threshold_type=settings.SEMANTIC_BREAKPOINT_THRESHOLD_TYPE,
            breakpoint_threshold_amount=settings.SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT,
        )
        return chunks, filenames

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


# ── Low-level helpers ─────────────────────────────────────────────────────────


def _extract_zip(zip_bytes: bytes, extract_dir: str) -> List[str]:
    """Extract zip and return absolute paths of supported files."""
    zip_path = os.path.join(extract_dir, "upload.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    os.remove(zip_path)

    supported: List[str] = []
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if Path(fname).suffix.lower() in _SUPPORTED_EXTENSIONS:
                supported.append(os.path.join(root, fname))

    logger.info("Extracted %d supported files from zip.", len(supported))
    return supported


def _load_documents(file_paths: List[str]) -> Tuple[List[Document], List[str]]:
    """Load all files; skip and log any that raise an error."""
    loader = UniversalDocumentLoader()
    all_docs: List[Document] = []
    loaded_names: List[str] = []

    for path in file_paths:
        try:
            docs = loader.load(path)
            all_docs.extend(docs)
            loaded_names.append(os.path.basename(path))
            logger.debug("Loaded %d docs from %s", len(docs), path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s — skipping.", path, exc)

    return all_docs, loaded_names
