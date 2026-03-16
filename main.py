"""
FastAPI application — entry point.

Run with:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import router
from src.config.config import settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
