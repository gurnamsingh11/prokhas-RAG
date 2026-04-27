"""
Production-grade logging configuration.

Features
--------
* **Structured JSON logs** to files for machine parsing (ELK, Datadog, etc.)
* **Human-readable console** output for local development
* **Rotating file handlers** to prevent unbounded disk growth
* **Separate error log** for quick incident triage
* **Correlation ID filter** populated by the request middleware so every log
  line within a single HTTP request shares a traceable ID
* **Log level** controlled via ``LOG_LEVEL`` environment variable / config

Usage
-----
Call ``setup_logging()`` once at application startup (before any other import
that calls ``logging.getLogger``).  Every module that already does
``logger = logging.getLogger(__name__)`` will automatically inherit the
configured handlers and formatters.
"""

import logging
import logging.handlers
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

# ── Correlation-ID context var (set per-request by middleware) ────────────────
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


# ── Custom filter that injects the correlation ID into every LogRecord ───────
class CorrelationIdFilter(logging.Filter):
    """Attach ``correlation_id`` to every log record so formatters can use it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get("-")  # type: ignore[attr-defined]
        return True


# ── JSON formatter for file handlers ─────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line (no external dependency)."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


# ── Setup function ────────────────────────────────────────────────────────────


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per file
    backup_count: int = 5,
    enable_json_console: bool = False,
) -> None:
    """
    Configure the root logger with production-grade handlers.

    Parameters
    ----------
    log_level : str
        Minimum severity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_dir : str
        Directory for log files.  Created automatically.
    max_bytes : int
        Max size of each rotating log file before rollover.
    backup_count : int
        Number of rotated backups to keep.
    enable_json_console : bool
        If True, console output is also JSON (useful in containerised deploys).
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove any previously-attached handlers (e.g. basicConfig defaults)
    root.handlers.clear()

    # Shared filter
    correlation_filter = CorrelationIdFilter()

    # ── Console handler (human-readable by default) ──────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    if enable_json_console:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                fmt=(
                    "%(asctime)s | %(levelname)-8s | %(name)-30s | "
                    "[%(correlation_id)s] %(message)s"
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    console_handler.addFilter(correlation_filter)
    root.addHandler(console_handler)

    # ── Rotating JSON file handler (all levels) ──────────────────────────────
    app_file = logging.handlers.RotatingFileHandler(
        filename=str(log_path / "app.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    app_file.setLevel(logging.DEBUG)
    app_file.setFormatter(JSONFormatter())
    app_file.addFilter(correlation_filter)
    root.addHandler(app_file)

    # ── Rotating error-only file handler ─────────────────────────────────────
    error_file = logging.handlers.RotatingFileHandler(
        filename=str(log_path / "error.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(JSONFormatter())
    error_file.addFilter(correlation_filter)
    root.addHandler(error_file)

    # ── Quieten noisy third-party loggers ────────────────────────────────────
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "uvicorn.access",
        "faiss",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    root_logger = logging.getLogger(__name__)
    root_logger.info(
        "Logging initialised — level=%s, log_dir=%s, json_console=%s",
        log_level,
        log_dir,
        enable_json_console,
    )
