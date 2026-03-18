"""
Upload service.

Two public entry points
-----------------------
ingest_zip(zip_bytes, session_name=None)
    Create a brand-new session from a zip file.
    session_name is an optional human-readable label for later lookup.

ingest_zip_into_session(zip_bytes, session_id)
    Append documents from a zip file to an existing session.
"""

import logging
import os
import shutil
import tempfile
import zipfile
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
    """Append documents from a zip file into an existing session."""
    meta = get_session(session_id)
    if meta is None:
        raise ValueError(f"Session '{session_id}' not found or has expired.")

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


def _load_documents(file_paths: List[str]) -> Tuple[List[Document], List[str]]:
    loader = UniversalDocumentLoader()
    all_docs: List[Document] = []
    loaded_names: List[str] = []

    for path in file_paths:
        try:
            docs = loader.load(path)
            all_docs.extend(docs)
            loaded_names.append(os.path.basename(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s — skipping.", path, exc)

    return all_docs, loaded_names
