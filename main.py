"""
FastAPI application — entry point.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.config import settings
from src.config.logging_config import setup_logging

# ── Logging (must be configured BEFORE any other module-level getLogger) ──────
setup_logging(
    log_level=settings.LOG_LEVEL,
    log_dir=settings.LOG_DIR,
    max_bytes=settings.LOG_MAX_BYTES,
    backup_count=settings.LOG_BACKUP_COUNT,
    enable_json_console=settings.LOG_JSON_CONSOLE,
)
logger = logging.getLogger(__name__)

from src.api.router import router  # noqa: E402  (after logging is configured)
from src.middleware.request_logging import RequestLoggingMiddleware  # noqa: E402

# ── Ensure upload temp dir exists ─────────────────────────────────────────────
os.makedirs(settings.UPLOAD_TMP_DIR, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Backend",
    description=(
        "Production-ready Retrieval-Augmented Generation API. "
        "Upload a zip of documents to create a session, then chat against them."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware order matters — outermost first
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

logger.info("RAG Backend v1.0.0 started — log_level=%s", settings.LOG_LEVEL)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
