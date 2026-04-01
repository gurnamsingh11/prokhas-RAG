"""
Central configuration — loaded once at startup.
All tuneable knobs live here so nothing is hard-coded elsewhere.

Quick reference
---------------
Copy .env.example to .env and set at minimum:
    HUGGINGFACEHUB_API_TOKEN=hf_...

Everything else has sensible defaults for local development.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    POPPLER_PATH: str = r"C:\Users\gurunaml\poppler-25.12.0\Library\bin"

    # ── HuggingFace ──────────────────────────────────────────────────────────
    # Your HuggingFace Hub API token.
    # Required for the hosted Qwen LLM endpoint.
    # Get one at https://huggingface.co/settings/tokens
    HUGGINGFACEHUB_API_TOKEN: str = ""

    # ── Embedding model ──────────────────────────────────────────────────────
    # HuggingFace model used to embed document chunks and queries.
    # Must be the SAME model across all server restarts — changing it after
    # documents have been indexed will make existing FAISS indexes incompatible.
    EMBEDDING_MODEL_NAME: str = "google/embeddinggemma-300m"

    # "cpu" for local dev; "cuda" if a GPU is available.
    EMBEDDING_DEVICE: str = "cpu"

    # Whether to L2-normalise embedding vectors before storing them.
    # Set True if your model recommends it (e.g. for cosine similarity search).
    EMBEDDING_NORMALIZE: bool = False

    # ── LLM ──────────────────────────────────────────────────────────────────
    # HuggingFace Hub repo ID for the chat model.
    LLM_REPO_ID: str = "Qwen/Qwen2.5-7B-Instruct"

    # Maximum number of new tokens the LLM may generate per response.
    LLM_MAX_NEW_TOKENS: int = 512

    # Penalises token repetition. 1.0 = no penalty; >1.0 discourages loops.
    LLM_REPETITION_PENALTY: float = 1.03

    # ── Chunking ─────────────────────────────────────────────────────────────
    # Target character length of each chunk (RecursiveCharacterTextSplitter fallback).
    CHUNK_SIZE: int = 800

    # Character overlap between consecutive chunks so context isn't lost at boundaries.
    CHUNK_OVERLAP: int = 150

    # SemanticChunker breakpoint strategy.
    # Options: "percentile" | "std_dev" | "interquartile"
    # "percentile" splits at points where cosine distance exceeds the Nth percentile.
    SEMANTIC_BREAKPOINT_THRESHOLD_TYPE: str = "percentile"

    # Numeric threshold for the chosen strategy.
    # For "percentile": 0–100 (e.g. 85 = top-15% biggest topic shifts become splits).
    # For "std_dev" / "interquartile": multiplier on the spread measure.
    SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT: float = 85.0

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # Number of chunks retrieved per query from the FAISS index.
    # Higher values give the LLM more context but increase prompt length and cost.
    RETRIEVER_TOP_K: int = 5

    # ── Session ──────────────────────────────────────────────────────────────
    # How long (in seconds) a session can be idle before it is evicted from RAM.
    # The FAISS index and metadata remain on disk — the session is NOT deleted.
    # Users can restore an evicted session via POST /sessions/{session_id}/restore
    # or by searching for it by name via GET /sessions/lookup?name=...
    # Default: 3600 (1 hour)
    SESSION_TTL_SECONDS: int = 3600

    # Temporary directory for ZIP extraction during upload.
    # Cleaned up automatically after each upload.
    UPLOAD_TMP_DIR: str = "/tmp/rag_uploads"

    # ── Persistence ──────────────────────────────────────────────────────────
    # Root directory where FAISS indexes and session metadata are stored on disk.
    #
    # Directory layout:
    #
    #   {FAISS_INDEX_DIR}/
    #     {session_id}/          ← one folder per session, named by UUID
    #       index.faiss          ← raw FAISS binary index
    #       index.pkl            ← docstore + id-to-docstore map
    #       meta.json            ← human-readable session metadata
    #
    # meta.json example:
    #   {
    #     "session_id":      "550e8400-e29b-41d4-a716-446655440000",
    #     "session_name":    "my-project",          ← optional, user-supplied
    #     "files_processed": ["report.pdf", "notes.docx"],
    #     "chunks_indexed":  142,
    #     "created_at":      "2025-03-17T10:30:00+00:00",
    #     "last_active":     "2025-03-17T11:02:45+00:00"
    #   }
    #
    # On startup the server automatically restores all sessions found here —
    # no manual intervention needed after a server restart.
    #
    # How a user re-accesses their embeddings later
    # ─────────────────────────────────────────────
    # Option 1 — by session_id (UUID):
    #   The session_id is returned on upload. Store it in your app / localStorage.
    #   On the next visit call POST /sessions/{session_id}/restore to bring it
    #   back into RAM (or it auto-restores at server startup).
    #
    # Option 2 — by session_name (human label):
    #   Pass {"session_name": "my-project"} when uploading.
    #   Later call GET /sessions/lookup?name=my-project to get the session_id back,
    #   then POST /sessions/{session_id}/restore if needed.
    #
    # Set to "" to run in pure in-memory mode (no disk persistence):
    FAISS_INDEX_DIR: str = "./faiss_store"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
