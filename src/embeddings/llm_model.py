"""
Singleton wrapper around the HuggingFace chat model.

Loaded once at first use; shared across sessions (stateless).
"""

import logging
from functools import lru_cache

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from src.config.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatHuggingFace:
    """Return (and cache) the global ChatHuggingFace instance."""
    logger.info("Loading LLM: %s", settings.LLM_REPO_ID)
    llm = HuggingFaceEndpoint(
        repo_id=settings.LLM_REPO_ID,
        task="text-generation",
        max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=settings.LLM_REPETITION_PENALTY,
    )
    return ChatHuggingFace(llm=llm, verbose=False)
