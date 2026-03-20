"""
FastAPI router — all endpoints.

Endpoints
---------
POST   /sessions/upload                 – Upload zip → create NEW session (optional name)
POST   /sessions/{session_id}/upload    – Upload zip → append to EXISTING session
POST   /sessions/{session_id}/restore   – Explicitly re-load a disk-persisted session
GET    /sessions/lookup?name=...        – Find a session by human-readable name
GET    /sessions                        – List sessions (RAM + auto-restores disk ones)
GET    /sessions/persisted              – List all session IDs saved on disk with status
GET    /sessions/{session_id}           – Session info (auto-restores from disk if needed)
DELETE /sessions/{session_id}           – Delete session + embeddings (disk + RAM)
POST   /sessions/{session_id}/chat      – Chat (auto-restores from disk if needed)

Fix 4: A threading.Lock now guards the name-uniqueness check + session creation
in upload_zip_new. Previously two concurrent requests with the same session_name
could both pass the duplicate check before either wrote meta.json, resulting in
two sessions with identical names. The lock makes the check-and-create atomic.
"""

import logging
import threading
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
    SessionMeta,
    delete_session,
    get_session,
    list_sessions,
    lookup_session_by_name,
    lookup_session_on_disk_by_name,
    maybe_expire_sessions,
    restore_session_from_disk,
    restore_sessions_from_disk,
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

# Fix 4: guards the name uniqueness check + session creation so concurrent
# requests with the same session_name can't both slip through the duplicate check.
_session_create_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _expire_sweep() -> None:
    """Evict RAM-only sessions that have been idle past SESSION_TTL_SECONDS."""
    maybe_expire_sessions()


def _get_or_restore(session_id: str) -> SessionMeta:
    """
    Return the SessionMeta for *session_id*, transparently restoring from disk
    if the session has been TTL-evicted from RAM or the server has restarted.

    Raises HTTP 404 only if no data exists anywhere (RAM or disk) — meaning
    the session was explicitly deleted or never created.

    This is the single place that handles the "it works right after upload but
    returns 404 hours later" problem.  Every endpoint that needs a live session
    should call this instead of get_session() directly.
    """
    # Fast path: still in RAM
    meta = get_session(session_id)
    if meta is not None:
        return meta

    # Slow path: try to restore from disk (handles TTL eviction + server restarts)
    logger.info(
        "Session %s not in RAM — attempting auto-restore from disk.", session_id
    )
    meta = restore_session_from_disk(session_id)

    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Session '{session_id}' not found. "
                "It may have been explicitly deleted or never created."
            ),
        )

    logger.info("Session %s auto-restored from disk successfully.", session_id)
    return meta


# ── Upload: create new session ────────────────────────────────────────────────


@router.post(
    "/sessions/upload",
    response_model=SessionCreatedOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a zip archive to create a new RAG session",
    description=(
        "Creates a new session and indexes the documents inside the zip. "
        "Supply an optional `session_name` so you can look up this session "
        "by name later via GET /sessions/lookup?name=..."
    ),
)
async def upload_zip_new(
    file: UploadFile = File(
        ..., description="A .zip archive containing PDFs, DOCX, and/or images"
    ),
    session_name: Optional[str] = Form(
        default=None,
        description=(
            "Optional human-readable label (e.g. 'my-project'). "
            "Use it to look up the session later without needing the UUID. "
            "Must be unique — duplicate names are rejected with 409."
        ),
    ),
):
    _expire_sweep()

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip files are accepted.",
        )

    # Fix 4: hold the lock for the duration of the name check + ingest so two
    # concurrent requests with the same name can't both pass the duplicate check.
    # The lock is only held when session_name is provided — unnamed uploads are
    # unaffected and remain fully concurrent.
    if session_name:
        session_name = session_name.strip()
        with _session_create_lock:
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
                session_meta = ingest_zip(zip_bytes, session_name=session_name)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
                ) from exc
            except Exception as exc:
                logger.exception("Unexpected error during ingest.")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
                ) from exc
    else:
        # No session_name — no lock needed, run fully concurrently
        try:
            zip_bytes = await file.read()
            session_meta = ingest_zip(zip_bytes, session_name=None)
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
            "Session created. "
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
    # Auto-restore if evicted — user shouldn't have to restore before appending
    meta_before = _get_or_restore(session_id)

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip files are accepted.",
        )

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


# ── Restore session from disk (explicit) ─────────────────────────────────────


@router.post(
    "/sessions/{session_id}/restore",
    response_model=SessionRestoredOut,
    status_code=status.HTTP_200_OK,
    summary="Explicitly restore a disk-persisted session into RAM",
    description=(
        "This endpoint exists for UIs that want to show a manual 'Restore' button. "
        "For normal use you do not need to call it — chat, GET session info, and "
        "append upload all auto-restore transparently."
    ),
)
async def restore_session(session_id: str = Path(...)):
    _expire_sweep()

    # If already in RAM, just return it
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
        "Searches RAM first, then disk. Auto-restores from disk if found there. "
        "Returns 404 if no session with that name exists anywhere."
    ),
)
async def lookup_by_name(
    name: str = Query(
        ..., description="The session_name given at upload time (case-insensitive)"
    ),
):
    _expire_sweep()

    # Check RAM
    meta = lookup_session_by_name(name)
    if meta is not None:
        return SessionLookupOut(**meta.to_dict(), status="active")

    # Check disk
    sid = lookup_session_on_disk_by_name(name)
    if sid is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No session named '{name}' found.",
        )

    # Auto-restore from disk
    meta = restore_session_from_disk(sid)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Found session '{name}' on disk but failed to restore it.",
        )

    return SessionLookupOut(**meta.to_dict(), status="restored_from_disk")


# ── List sessions ─────────────────────────────────────────────────────────────


@router.get(
    "/sessions",
    summary="List all sessions — active in RAM plus any found on disk",
    description=(
        "Returns all sessions the user has ever created (that haven't been deleted). "
        "Sessions evicted from RAM by TTL are auto-restored from disk so they "
        "always appear in this list, never disappear after idle time."
    ),
)
async def list_all_sessions():
    """
    Auto-restores all on-disk sessions before listing so the UI always sees
    every session regardless of TTL eviction or server restarts.
    """
    _expire_sweep()

    # Restore any disk sessions not already in RAM
    persisted_ids = list_persisted_session_ids()
    in_ram = {s["session_id"] for s in list_sessions()}
    for sid in persisted_ids:
        if sid not in in_ram:
            restore_session_from_disk(sid)

    return list_sessions()


# ── List persisted sessions (disk status view) ─────────────────────────────────


@router.get(
    "/sessions/persisted",
    summary="List all session IDs saved on disk with their RAM status",
)
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
    summary="Get metadata for a session (auto-restores from disk if needed)",
)
async def get_session_info(session_id: str = Path(...)):
    _expire_sweep()
    meta = _get_or_restore(session_id)
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
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
    summary="Chat with a session (auto-restores from disk if needed)",
    description=(
        "If the session has been idle past the TTL or the server has restarted, "
        "the session is silently restored from disk before answering. "
        "The caller never needs to call /restore manually."
    ),
)
async def chat(session_id: str = Path(...), body: ChatRequest = ...):
    _expire_sweep()

    if not body.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    # Auto-restore transparently — this is the fix for the "works on upload,
    # breaks after a few hours" problem reported by the UI developer.
    meta = _get_or_restore(session_id)

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
