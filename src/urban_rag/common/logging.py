"""Structured logging configuration for Urban RAG.

All logs are structured JSON, one event per line, with correlation IDs
propagated through the request context (query_id, corpus_version, service).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.stdlib import ProcessorFormatter

from urban_rag.common.settings import get_settings


def configure_logging(
    *,
    corpus_version: str = "unversioned",
    service: str = "unknown",
) -> None:
    """Configure structlog for the application.

    Args:
        corpus_version: Version string of the indexed corpus. Will be bound to all log events.
        service: Name of the service emitting logs (e.g., "api", "cli", "embed").
    """
    settings = get_settings()

    # Set default context variables that will be merged into every log event
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(corpus_version=corpus_version, service=service)

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

    # Also configure standard library logging to use structlog processors
    # so that logging.getLogger() also outputs structured JSON
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ProcessorFormatter(
            foreign_pre_chain=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
            ],
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


def get_logger(
    name: str | None = None,
    *,
    query_id: str | None = None,
    corpus_version: str | None = None,
    service: str | None = None,
    **extra: Any,
) -> structlog.PrintLogger:
    """Return a configured structlog logger.

    Args:
        name: Logger name (e.g., __name__ of calling module).
        query_id: Unique query identifier for telemetry/feedback correlation.
        corpus_version: Version string of the indexed corpus.
        service: Name of the service emitting the log (e.g., "api", "cli", "embed").
        **extra: Additional context fields to bind to all log events.

    Returns:
        A structlog PrintLogger with bound context variables.
    """
    logger = structlog.get_logger(name)

    # Bind standard correlation fields
    if query_id is not None:
        logger = logger.bind(query_id=query_id)
    if corpus_version is not None:
        logger = logger.bind(corpus_version=corpus_version)
    if service is not None:
        logger = logger.bind(service=service)

    # Bind any extra fields
    for key, value in extra.items():
        logger = logger.bind(**{key: value})

    return logger
