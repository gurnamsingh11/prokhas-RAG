"""
FastAPI router — all endpoints.

Endpoints
---------
POST /sessions/upload                   – Upload zip → create NEW session
POST /sessions/{session_id}/upload      – Upload zip → append to EXISTING session
GET  /sessions                          – List all active sessions
GET  /sessions/{session_id}             – Session info
DELETE /sessions/{session_id}           – Delete session + embeddings
POST /sessions/{session_id}/chat        – Send a message, get RAGResponse
"""

import logging

from fastapi import APIRouter, File, HTTPException, Path, UploadFile, status
from pydantic import BaseModel

from src.agents.rag_agent import run_rag_query
from src.api.upload_service import ingest_zip, ingest_zip_into_session
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
    ZipAddedOut,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Dependency: TTL sweep ─────────────────────────────────────────────────────


def _expire_sweep():
    maybe_expire_sessions()


# ── Upload: create new session ────────────────────────────────────────────────


@router.post(
    "/sessions/upload",
    response_model=SessionCreatedOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a zip archive to create a new RAG session",
)
async def upload_zip_new(
    file: UploadFile = File(
        ..., description="A .zip archive containing PDFs, DOCX, and/or images"
    ),
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


# ── Upload: append to existing session ───────────────────────────────────────


@router.post(
    "/sessions/{session_id}/upload",
    response_model=ZipAddedOut,
    status_code=status.HTTP_200_OK,
    summary="Upload an additional zip archive into an existing session",
    description=(
        "Embeds the new documents and merges them into the session's existing "
        "FAISS vector store. Conversation history is preserved and the new "
        "content is immediately searchable."
    ),
)
async def upload_zip_append(
    session_id: str = Path(..., description="ID of the session to append to"),
    file: UploadFile = File(
        ..., description="A .zip archive containing PDFs, DOCX, and/or images"
    ),
):
    _expire_sweep()

    # Validate session exists before reading file bytes
    if get_session(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip files are accepted.",
        )

    # Snapshot counts before append for the diff in the response
    meta_before = get_session(session_id)
    files_before = len(meta_before.files_processed)
    chunks_before = meta_before.chunks_indexed

    try:
        zip_bytes = await file.read()
        updated_meta = ingest_zip_into_session(zip_bytes, session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected error during append ingest for session %s.", session_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    files_added = len(updated_meta.files_processed) - files_before
    chunks_added = updated_meta.chunks_indexed - chunks_before

    return ZipAddedOut(
        session_id=session_id,
        message=f"Successfully added {files_added} file(s) and {chunks_added} chunk(s) to the session.",
        files_added=files_added,
        chunks_added=chunks_added,
        total_files=len(updated_meta.files_processed),
        total_chunks=updated_meta.chunks_indexed,
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
