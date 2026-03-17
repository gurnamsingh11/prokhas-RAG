"""
Central configuration — loaded once at startup.
All tuneable knobs live here so nothing is hard-coded elsewhere.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── HuggingFace ──────────────────────────────────────────────────────────
    HUGGINGFACEHUB_API_TOKEN: str = ""

    # ── Embedding model ──────────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "google/embeddinggemma-300m"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NORMALIZE: bool = False

    # ── LLM ──────────────────────────────────────────────────────────────────
    LLM_REPO_ID: str = "Qwen/Qwen2.5-7B-Instruct"
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_REPETITION_PENALTY: float = 1.03

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 120
    SEMANTIC_BREAKPOINT_THRESHOLD_TYPE: str = (
        "percentile"  # percentile | std_dev | interquartile
    )
    SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT: float = 85.0

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVER_TOP_K: int = 5

    # ── Session ──────────────────────────────────────────────────────────────
    SESSION_TTL_SECONDS: int = 3600  # 1 hour idle expiry
    UPLOAD_TMP_DIR: str = "/tmp/rag_uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
