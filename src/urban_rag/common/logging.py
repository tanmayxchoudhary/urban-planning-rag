"""Structured logging configuration for Urban RAG.

All logs are structured JSON, one event per line, with correlation IDs
propagated through the request context (query_id, corpus_version, service).
"""

from __future__ import annotations

import logging
import sys

import structlog

from urban_rag.common.settings import get_settings


def configure_logging() -> None:
    """Configure structlog for the application."""
    settings = get_settings()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.PrintLogger:
    """Return a configured structlog logger."""
    return structlog.get_logger(name)
