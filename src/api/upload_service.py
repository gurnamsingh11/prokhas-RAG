"""
Upload service.

Handles the full ingest pipeline:
  zip upload → extract → load docs → smart chunk → embed → FAISS store → session
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
from src.memory.session_registry import SessionMeta, create_session
from src.vectorstore.session_store import build_session_store

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = (
    UniversalDocumentLoader.PDF_EXTENSIONS
    | UniversalDocumentLoader.WORD_EXTENSIONS
    | UniversalDocumentLoader.IMAGE_EXTENSIONS
)


def ingest_zip(zip_bytes: bytes) -> SessionMeta:
    """
    Full ingest pipeline for a zip file.

    Parameters
    ----------
    zip_bytes : raw bytes of the uploaded zip archive.

    Returns
    -------
    SessionMeta for the newly created session.
    """
    extract_dir = tempfile.mkdtemp(prefix="rag_", dir=settings.UPLOAD_TMP_DIR)
    try:
        # ── 1. Extract zip ────────────────────────────────────────────────────
        supported_files = _extract_zip(zip_bytes, extract_dir)
        if not supported_files:
            raise ValueError(
                "The zip archive contains no supported files "
                f"({', '.join(sorted(_SUPPORTED_EXTENSIONS))})."
            )

        # ── 2. Load documents ─────────────────────────────────────────────────
        raw_docs, filenames = _load_documents(supported_files)
        if not raw_docs:
            raise ValueError("All supported files produced empty documents.")

        # ── 3. Smart chunk ────────────────────────────────────────────────────
        embedding_model = get_embedding_model()
        chunks = smart_chunk_documents(
            docs=raw_docs,
            embeddings=embedding_model,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            breakpoint_threshold_type=settings.SEMANTIC_BREAKPOINT_THRESHOLD_TYPE,
            breakpoint_threshold_amount=settings.SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT,
        )

        # ── 4. Embed + build FAISS store ──────────────────────────────────────
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
            "Ingest complete for session %s: %d files, %d chunks.",
            session_meta.session_id,
            len(filenames),
            len(chunks),
        )
        return session_meta

    finally:
        # Always clean up extracted files — embeddings are now in RAM
        shutil.rmtree(extract_dir, ignore_errors=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


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
