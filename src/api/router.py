"""
FastAPI router — all endpoints.

Endpoints
---------
POST   /sessions/upload                 – Upload zip → create NEW session (optional name)
POST   /sessions/{session_id}/upload    – Upload zip → append to EXISTING session
POST   /sessions/{session_id}/restore   – Re-load a disk-persisted session into RAM
GET    /sessions/lookup?name=...        – Find a session by human-readable name
GET    /sessions                        – List all active (in-RAM) sessions
GET    /sessions/persisted              – List all sessions saved on disk
GET    /sessions/{session_id}           – Session info
DELETE /sessions/{session_id}           – Delete session + embeddings (disk + RAM)
POST   /sessions/{session_id}/chat      – Send a message, get RAGResponse
"""

import logging
from typing import Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel

from src.agents.rag_agent import run_rag_query
from src.api.upload_service import ingest_zip, ingest_zip_into_session
from src.memory.session_registry import (
    delete_session,
    get_session,
    list_sessions,
    lookup_session_by_name,
    lookup_session_on_disk_by_name,
    maybe_expire_sessions,
    restore_session_from_disk,
)
from src.schemas.responses import (
    ChatResponseOut,
    DeleteResponseOut,
    SessionCreatedOut,
    SessionInfoOut,
    SessionLookupOut,
    SessionRestoredOut,
    SourceReferenceOut,
    ZipAddedOut,
)
from src.vectorstore.session_store import list_persisted_session_ids

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
    description=(
        "Creates a new session and indexes the documents inside the zip. "
        "Supply an optional `session_name` (e.g. 'q3-reports') so you can "
        "look up this session by name later via GET /sessions/lookup?name=..."
    ),
)
async def upload_zip_new(
    file: UploadFile = File(
        ..., description="A .zip archive containing PDFs, DOCX, and/or images"
    ),
    session_name: Optional[str] = Form(
        default=None,
        description=(
            "Optional human-readable label for this session (e.g. 'my-project'). "
            "Used to look up the session later without needing the UUID. "
            "Must be unique across all sessions on disk — duplicate names are rejected."
        ),
    ),
):
    _expire_sweep()

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip files are accepted.",
        )

    # Reject duplicate session names to avoid ambiguous lookups
    if session_name:
        session_name = session_name.strip()
        if lookup_session_by_name(session_name) is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A session named '{session_name}' is already active. Choose a different name or delete the existing one.",
            )
        disk_sid = lookup_session_on_disk_by_name(session_name)
        if disk_sid is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"A session named '{session_name}' already exists on disk "
                    f"(session_id: {disk_sid}). "
                    "Restore it with POST /sessions/{session_id}/restore, "
                    "or delete it first."
                ),
            )

    try:
        zip_bytes = await file.read()
        session_meta = ingest_zip(zip_bytes, session_name=session_name or None)
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
        session_name=session_meta.session_name,
        message=(
            f"Session created. "
            + (
                f"Accessible by name '{session_meta.session_name}'. "
                if session_meta.session_name
                else ""
            )
            + "You can now chat using the session_id."
        ),
        files_processed=len(session_meta.files_processed),
        chunks_indexed=session_meta.chunks_indexed,
    )


# ── Upload: append to existing session ───────────────────────────────────────


@router.post(
    "/sessions/{session_id}/upload",
    response_model=ZipAddedOut,
    status_code=status.HTTP_200_OK,
    summary="Upload an additional zip archive into an existing session",
)
async def upload_zip_append(
    session_id: str = Path(...),
    file: UploadFile = File(...),
):
    _expire_sweep()

    if get_session(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
        )
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip files are accepted.",
        )

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
        logger.exception("Append ingest failed for session %s.", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    return ZipAddedOut(
        session_id=session_id,
        message=f"Successfully added {len(updated_meta.files_processed) - files_before} file(s) to the session.",
        files_added=len(updated_meta.files_processed) - files_before,
        chunks_added=updated_meta.chunks_indexed - chunks_before,
        total_files=len(updated_meta.files_processed),
        total_chunks=updated_meta.chunks_indexed,
    )


# ── Restore session from disk ─────────────────────────────────────────────────


@router.post(
    "/sessions/{session_id}/restore",
    response_model=SessionRestoredOut,
    status_code=status.HTTP_200_OK,
    summary="Restore a previously created session from disk into RAM",
)
async def restore_session(session_id: str = Path(...)):
    _expire_sweep()

    existing = get_session(session_id)
    if existing is not None:
        return SessionRestoredOut(
            **existing.to_dict(), message="Session is already active in RAM."
        )

    meta = restore_session_from_disk(session_id)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No persisted data found for session '{session_id}'. "
                "It may have been explicitly deleted or never created."
            ),
        )

    return SessionRestoredOut(
        **meta.to_dict(),
        message="Session restored from disk. You can now chat immediately.",
    )


# ── Lookup session by name ────────────────────────────────────────────────────


@router.get(
    "/sessions/lookup",
    response_model=SessionLookupOut,
    summary="Find a session by its human-readable name",
    description=(
        "Searches both active (in-RAM) sessions and on-disk sessions by the "
        "session_name supplied at upload time. "
        "If the session is found on disk but not in RAM, it is automatically "
        "restored so it is immediately ready to chat. "
        "Returns 404 if no session with that name exists."
    ),
)
async def lookup_by_name(
    name: str = Query(
        ..., description="The session_name given at upload time (case-insensitive)"
    ),
):
    _expire_sweep()

    # 1. Check RAM first
    meta = lookup_session_by_name(name)
    if meta is not None:
        return SessionLookupOut(**meta.to_dict(), status="active")

    # 2. Check disk
    sid = lookup_session_on_disk_by_name(name)
    if sid is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No session named '{name}' found. "
                "Sessions are only named if you supplied session_name at upload time."
            ),
        )

    # 3. Auto-restore from disk
    meta = restore_session_from_disk(sid)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Found session '{name}' on disk but failed to restore it.",
        )

    return SessionLookupOut(**meta.to_dict(), status="restored_from_disk")


# ── List active sessions (RAM) ────────────────────────────────────────────────


@router.get("/sessions", summary="List all sessions currently active in RAM")
async def list_all_sessions():
    _expire_sweep()
    return list_sessions()


# ── List persisted sessions (disk) ────────────────────────────────────────────


@router.get("/sessions/persisted", summary="List all session IDs saved on disk")
async def list_persisted_sessions():
    _expire_sweep()
    ids = list_persisted_session_ids()
    active = {s["session_id"] for s in list_sessions()}
    return [
        {"session_id": sid, "status": "active" if sid in active else "on_disk_only"}
        for sid in ids
    ]


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
    summary="Permanently delete a session — removes RAM, FAISS index, and metadata from disk",
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
        message="Session and all associated data permanently deleted.",
    )


# ── Chat ──────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str


@router.post(
    "/sessions/{session_id}/chat",
    response_model=ChatResponseOut,
    summary="Chat with a session",
)
async def chat(session_id: str = Path(...), body: ChatRequest = ...):
    _expire_sweep()

    if not body.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty."
        )

    meta = get_session(session_id)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Session not found or evicted from RAM. "
                "Use POST /sessions/{session_id}/restore or "
                "GET /sessions/lookup?name=... to bring it back."
            ),
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
