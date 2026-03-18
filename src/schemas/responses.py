"""
API-level response schemas.

Rule: structured output uses TypedDict only — no Pydantic models per spec.
Pydantic is allowed for FastAPI *request* validation (multipart bodies, etc.)
but NOT for LLM structured output schemas.
"""

from typing import List, Optional
from typing_extensions import TypedDict


# ── LLM structured-output schema ─────────────────────────────────────────────


class SourceReference(TypedDict):
    """A single source document that contributed to the answer."""

    document_name: str  # filename (e.g. "report.pdf")
    page_label: str  # page number / label if available, else ""
    relevant_excerpt: str  # short excerpt from that chunk (≤ 200 chars)


class RAGResponse(TypedDict):
    """
    Structured output returned by the RAG agent for every user query.
    The LLM is instructed to populate both fields.
    """

    answer: str  # the full natural-language answer
    sources: List[SourceReference]  # which documents / pages backed the answer


# ── HTTP response models (Pydantic — only for FastAPI serialisation) ──────────

from pydantic import BaseModel


class SourceReferenceOut(BaseModel):
    document_name: str
    page_label: str
    relevant_excerpt: str


class ChatResponseOut(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceReferenceOut]


class SessionCreatedOut(BaseModel):
    session_id: str
    session_name: Optional[str] = None  # human label if supplied at upload time
    message: str
    files_processed: int
    chunks_indexed: int


class SessionInfoOut(BaseModel):
    session_id: str
    session_name: Optional[str] = None
    files_processed: List[str]
    chunks_indexed: int
    created_at: str
    last_active: str


class DeleteResponseOut(BaseModel):
    session_id: str
    deleted: bool
    message: str


class ZipAddedOut(BaseModel):
    session_id: str
    message: str
    files_added: int
    chunks_added: int
    total_files: int
    total_chunks: int


class SessionRestoredOut(BaseModel):
    session_id: str
    session_name: Optional[str] = None
    message: str
    files_processed: List[str]
    chunks_indexed: int
    created_at: str
    last_active: str


class SessionLookupOut(BaseModel):
    session_id: str
    session_name: str
    files_processed: List[str]
    chunks_indexed: int
    created_at: str
    last_active: str
    status: str  # "active" | "on_disk_only"
