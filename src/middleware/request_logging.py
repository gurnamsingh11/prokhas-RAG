"""
FastAPI middleware for production-grade request / response logging.

Responsibilities
----------------
* Generate a unique **correlation ID** for every inbound request and store it
  in the ``correlation_id_var`` context variable so that all downstream log
  lines are automatically tagged.
* Return the correlation ID in the ``X-Correlation-ID`` response header so
  clients and support teams can reference it.
* Log request start (method, path, client IP) and request completion
  (status code, duration in ms).
* Log unhandled exceptions at ERROR level with full traceback.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.config.logging_config import correlation_id_var

logger = logging.getLogger("src.middleware.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Inject correlation ID and log every HTTP request / response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Prefer client-supplied correlation ID; generate one otherwise
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex[:12]
        token = correlation_id_var.set(cid)

        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            "→ %s %s from %s",
            request.method,
            request.url.path,
            client_ip,
        )

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "✗ %s %s — unhandled exception after %.1f ms",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        finally:
            correlation_id_var.reset(token)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "← %s %s — %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Correlation-ID"] = cid
        return response
