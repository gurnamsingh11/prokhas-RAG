"""
FastAPI router — all endpoints.

Endpoints
---------
POST /sessions/upload          – Upload zip, create session, return session_id
GET  /sessions                 – List all active sessions
GET  /sessions/{session_id}    – Session info
DELETE /sessions/{session_id}  – Delete session + embeddings
POST /sessions/{session_id}/chat – Send a message, get RAGResponse
"""

import logging

from fastapi import APIRouter, File, HTTPException, Path, UploadFile, status
from pydantic import BaseModel

from src.agents.rag_agent import run_rag_query
from src.api.upload_service import ingest_zip
from src.memory.session_registry import (
    delete_session,
    get_session,
    list_sessions,
    maybe_expire_sessions,
)
from src.schemas.responses import (
    ChatResponseOut,
    DeleteResponseOut,
    SessionCreatedOut,
    SessionInfoOut,
    SourceReferenceOut,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── dependency: TTL sweep ─────────────────────────────────────────────────────


def _expire_sweep():
    """Called on every request to clean up idle sessions."""
    maybe_expire_sessions()


# ── Upload ────────────────────────────────────────────────────────────────────


@router.post(
    "/sessions/upload",
    response_model=SessionCreatedOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a zip archive to create a new RAG session",
)
async def upload_zip(
    file: UploadFile = File(
        ..., description="A .zip archive containing PDFs, DOCX, and/or images"
    )
):
    _expire_sweep()

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip files are accepted.",
        )

    try:
        zip_bytes = await file.read()
        session_meta = ingest_zip(zip_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during ingest.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    return SessionCreatedOut(
        session_id=session_meta.session_id,
        message="Session created. You can now chat using the session_id.",
        files_processed=len(session_meta.files_processed),
        chunks_indexed=session_meta.chunks_indexed,
    )


# ── List sessions ─────────────────────────────────────────────────────────────


@router.get(
    "/sessions",
    summary="List all active sessions",
)
async def list_all_sessions():
    _expire_sweep()
    return list_sessions()


# ── Session info ──────────────────────────────────────────────────────────────


@router.get(
    "/sessions/{session_id}",
    response_model=SessionInfoOut,
    summary="Get metadata for a session",
)
async def get_session_info(session_id: str = Path(...)):
    _expire_sweep()
    meta = get_session(session_id)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
        )
    return SessionInfoOut(**meta.to_dict())


# ── Delete session ────────────────────────────────────────────────────────────


@router.delete(
    "/sessions/{session_id}",
    response_model=DeleteResponseOut,
    summary="Delete a session and its vector store",
)
async def delete_session_endpoint(session_id: str = Path(...)):
    _expire_sweep()
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
        )
    return DeleteResponseOut(
        session_id=session_id,
        deleted=True,
        message="Session and embeddings deleted successfully.",
    )


# ── Chat ──────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str


@router.post(
    "/sessions/{session_id}/chat",
    response_model=ChatResponseOut,
    summary="Send a message and receive a RAG-grounded answer with sources",
)
async def chat(
    session_id: str = Path(...),
    body: ChatRequest = ...,
):
    _expire_sweep()

    if not body.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    meta = get_session(session_id)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
        )

    try:
        rag_response = run_rag_query(session_id, body.message)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("RAG query failed for session %s.", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    sources_out = [
        SourceReferenceOut(
            document_name=s["document_name"],
            page_label=s.get("page_label", ""),
            relevant_excerpt=s.get("relevant_excerpt", ""),
        )
        for s in rag_response.get("sources", [])
    ]

    return ChatResponseOut(
        session_id=session_id,
        answer=rag_response["answer"],
        sources=sources_out,
    )
