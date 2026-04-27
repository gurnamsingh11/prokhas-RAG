"""
Smart chunking pipeline.

Strategy
--------
1. Try **SemanticChunker** (embedding-based) for meaningful boundary detection.
2. Fall back to **RecursiveCharacterTextSplitter** if semantic chunker fails
   (e.g. very short documents, single-sentence pages).
3. Attach / preserve source metadata on every chunk so the retriever can
   report which file and page each chunk came from.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# SemanticChunker lives in langchain_experimental
try:
    from langchain_experimental.text_splitter import SemanticChunker  # type: ignore

    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False
    logger.warning(
        "langchain_experimental not installed — semantic chunking disabled. "
        "Install with: pip install langchain-experimental"
    )


def smart_chunk_documents(
    docs: List[Document],
    embeddings: Embeddings,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 85.0,
) -> List[Document]:
    """
    Chunk a list of Documents using semantic boundaries where possible.

    Parameters
    ----------
    docs:
        Raw documents from the universal loader.
    embeddings:
        Embedding model used by SemanticChunker to detect topic shifts.
    chunk_size / chunk_overlap:
        Fallback parameters for RecursiveCharacterTextSplitter.
    breakpoint_threshold_type:
        One of "percentile", "std_dev", "interquartile".
    breakpoint_threshold_amount:
        Numeric threshold for the chosen type.

    Returns
    -------
    List[Document] with source + page_label metadata intact on every chunk.
    """
    if not docs:
        return []

    all_chunks: List[Document] = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page_label = doc.metadata.get("page_label", "")

        # Images are kept as a single chunk — splitting a multi-line description
        # across chunk boundaries would break retrieval for image content.
        if doc.metadata.get("do_not_split"):
            chunks = [doc]
            logger.debug("Skipping chunking for image doc: %s", source)
        else:
            chunks = _chunk_single_doc(
                doc=doc,
                embeddings=embeddings,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
            )

        # Guarantee metadata on every chunk AND prepend a contextual header
        # so the embedding captures document identity. This is critical when
        # multiple documents share the same structure but different values
        # (e.g. claims, invoices, forms) — without the header, their chunk
        # embeddings would be nearly identical and retrieval would mix them up.
        for chunk in chunks:
            chunk.metadata["source"] = source
            chunk.metadata["page_label"] = page_label

            header_parts = [f"[Document: {source}"]
            if page_label:
                header_parts.append(f" | Page: {page_label}")
            header_parts.append("]\n")
            header = "".join(header_parts)

            if not chunk.page_content.startswith(header):
                chunk.page_content = header + chunk.page_content

        all_chunks.extend(chunks)

    logger.info(
        "Chunking complete: %d raw docs → %d chunks", len(docs), len(all_chunks)
    )
    return all_chunks


def _chunk_single_doc(
    doc: Document,
    embeddings: Embeddings,
    chunk_size: int,
    chunk_overlap: int,
    breakpoint_threshold_type: str,
    breakpoint_threshold_amount: float,
) -> List[Document]:
    text = doc.page_content.strip()

    # Skip near-empty pages
    if len(text) < 50:
        return [doc]

    # ── Attempt semantic chunking ─────────────────────────────────────────────
    if _SEMANTIC_AVAILABLE:
        try:
            splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
            )
            chunks = splitter.create_documents([text], metadatas=[doc.metadata])
            if chunks:
                return chunks
        except Exception as exc:  # noqa: BLE001
            logger.debug("SemanticChunker failed (%s) — using recursive splitter.", exc)

    # ── Fallback: recursive character splitter ────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.create_documents([text], metadatas=[doc.metadata])
