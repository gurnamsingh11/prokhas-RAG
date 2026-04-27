"""
Upload service.

Two public entry points
-----------------------
ingest_zip(zip_bytes, session_name=None)
    Create a brand-new session from a zip file.
    session_name is an optional human-readable label for later lookup.

ingest_zip_into_session(zip_bytes, session_id)
    Append documents from a zip file to an existing session.

Fix 3: ingest_zip_into_session previously called get_session() only, which
returned None for any TTL-evicted session and raised a 422 error even though
the session's data was intact on disk. It now mirrors the auto-restore pattern
used by the router's _get_or_restore() helper.
"""

import logging
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

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
    restore_session_from_disk,  # Fix 3: needed for auto-restore on append
)
from src.vectorstore.session_store import add_to_session_store, build_session_store

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = (
    UniversalDocumentLoader.PDF_EXTENSIONS
    | UniversalDocumentLoader.WORD_EXTENSIONS
    | UniversalDocumentLoader.IMAGE_EXTENSIONS
)


# ── Public: create new session ────────────────────────────────────────────────


def ingest_zip(
    zip_bytes: bytes,
    session_name: Optional[str] = None,
) -> SessionMeta:
    """
    Full ingest pipeline for a zip file — creates a new session.

    Parameters
    ----------
    zip_bytes:    Raw bytes of the uploaded zip archive.
    session_name: Optional human-readable label (e.g. "q3-reports").
                  If provided, users can look up this session by name later via
                  GET /sessions/lookup?name=q3-reports instead of remembering
                  the UUID session_id.

    Returns SessionMeta for the newly created session.
    """
    chunks, filenames = _extract_and_chunk(zip_bytes)

    embedding_model = get_embedding_model()
    session_meta = create_session(
        files_processed=filenames,
        chunks_indexed=len(chunks),
        session_name=session_name,
    )
    build_session_store(
        session_id=session_meta.session_id,
        chunks=chunks,
        embeddings=embedding_model,
    )

    logger.info(
        "Ingest complete for NEW session %s (name=%r): %d files, %d chunks.",
        session_meta.session_id,
        session_name,
        len(filenames),
        len(chunks),
    )
    return session_meta


# ── Public: append to existing session ───────────────────────────────────────


def ingest_zip_into_session(zip_bytes: bytes, session_id: str) -> SessionMeta:
    """
    Append documents from a zip file into an existing session.

    Fix 3: Previously only checked RAM (get_session), which raised a ValueError
    for any session that had been TTL-evicted — even though the session's FAISS
    index and metadata were intact on disk. Now mirrors the auto-restore pattern
    used everywhere else: RAM first, then disk, then 404.
    """
    # Try RAM first, then transparently restore from disk
    meta = get_session(session_id)
    if meta is None:
        logger.info(
            "Session %s not in RAM during append — attempting restore from disk.",
            session_id,
        )
        meta = restore_session_from_disk(session_id)

    if meta is None:
        raise ValueError(
            f"Session '{session_id}' not found. "
            "It may have been permanently deleted or never created."
        )

    chunks, filenames = _extract_and_chunk(zip_bytes)
    embedding_model = get_embedding_model()

    total_vectors = add_to_session_store(
        session_id=session_id,
        chunks=chunks,
        embeddings=embedding_model,
    )
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


# ── Shared pipeline ───────────────────────────────────────────────────────────


def _extract_and_chunk(zip_bytes: bytes) -> Tuple[List[Document], List[str]]:
    extract_dir = tempfile.mkdtemp(prefix="rag_", dir=settings.UPLOAD_TMP_DIR)
    try:
        supported_files = _extract_zip(zip_bytes, extract_dir)
        if not supported_files:
            raise ValueError(
                "The zip archive contains no supported files "
                f"({', '.join(sorted(_SUPPORTED_EXTENSIONS))})."
            )

        embedding_model = get_embedding_model()

        # --- Parallel per-file load + chunk pipeline ---
        max_workers = min(len(supported_files), os.cpu_count() or 4, 8)
        all_chunks: List[Document] = []
        filenames: List[str] = []

        def _process_single_file(path: str) -> Tuple[List[Document], str]:
            """Load and chunk a single file — runs in a worker thread."""
            loader = UniversalDocumentLoader()
            docs = loader.load(path)
            if not docs:
                return [], os.path.basename(path)
            chunks = smart_chunk_documents(
                docs=docs,
                embeddings=embedding_model,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                breakpoint_threshold_type=settings.SEMANTIC_BREAKPOINT_THRESHOLD_TYPE,
                breakpoint_threshold_amount=settings.SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT,
            )
            return chunks, os.path.basename(path)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_single_file, fp): fp for fp in supported_files
            }
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    chunks, fname = future.result()
                    if chunks:
                        all_chunks.extend(chunks)
                        filenames.append(fname)
                except Exception as exc:
                    logger.warning("Failed to process %s: %s — skipping.", fp, exc)

        if not all_chunks:
            raise ValueError("All supported files produced empty documents.")

        return all_chunks, filenames

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def _extract_zip(zip_bytes: bytes, extract_dir: str) -> List[str]:
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
