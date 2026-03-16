"""
Singleton wrapper around HuggingFaceEmbeddings.

The model is loaded once at process startup (lazy) and reused across all
sessions — embeddings are stateless so sharing is safe.
"""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return (and cache) the global HuggingFaceEmbeddings instance."""
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": settings.EMBEDDING_NORMALIZE},
    )
