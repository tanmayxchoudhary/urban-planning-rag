"""Langfuse tracing for Urban Planning RAG.

Per PLAN.md Part XIII §13.1:
    Every query is a trace with these spans, in order:

    trace: query                    (root, attrs: query_id, mode, user_hash, corpus_version)
      ├─ span: retrieval
      │   ├─ span: query_expansion  (prompt, expanded_queries)
      │   ├─ span: visual_search    (qdrant, candidates, latency)
      │   ├─ span: text_search      (qdrant, candidates, latency)
      │   ├─ span: sparse_search    (bm25, candidates, latency)
      │   ├─ span: fusion           (rrf inputs/output)
      │   └─ span: vlm_rerank       (model, scored_candidates, latency, cost)
      ├─ span: generation
      │   ├─ span: prompt_build     (template_id, template_version, tokens)
      │   └─ span: vlm_generate     (model, prompt_tokens, output_tokens, latency, cost)
      └─ span: post_process         (citation_validation, schema_check)

Usage:
    from urban_rag.telemetry.tracing import (
        get_tracer,
        make_span,
        get_current_trace_url,
        get_current_trace_id,
    )

    # In API endpoint or background task:
    tracer = get_tracer()
    with tracer.start_as_current_span("query", query_id=query_id, ...) as span:
        span.set_attribute("query_id", query_id)
        span.set_attribute("mode", mode)
        ...
        # Within retrieval orchestrator:
        with make_span("retrieval") as retrieval_span:
            with make_span("query_expansion") as expansion_span:
                ...
            with make_span("visual_search") as visual_span:
                ...

The Langfuse client is initialized lazily on first use, so missing credentials
only cause errors at trace time (not at startup).
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = structlog.get_logger(__name__, service="telemetry-tracing")

# ---------------------------------------------------------------------------
# Global Langfuse client (lazy singleton)
# ---------------------------------------------------------------------------

_LANGFUSE_CLIENT: Langfuse | None = None  # type: ignore[valid-type]
_LANGFUSE_LOCK = threading.Lock()


def _get_langfuse_client() -> Langfuse | None:  # type: ignore[valid-type]
    """Get or create the global Langfuse client.

    Returns None if Langfuse is not configured (credentials missing or disabled).
    The client is created once and reused for the lifetime of the process.
    """
    global _LANGFUSE_CLIENT

    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    with _LANGFUSE_LOCK:
        if _LANGFUSE_CLIENT is not None:
            return _LANGFUSE_CLIENT

        # Import here to avoid hard dependency when tracing is disabled
        try:
            from langfuse import Langfuse
        except ImportError:
            logger.warning("langfuse_not_installed_cannot_initialize_tracing")
            return None

        from urban_rag.common.settings import get_settings

        settings = get_settings()

        if not settings.langfuse_enabled:
            logger.debug("langfuse_disabled_skipping_initialization")
            return None

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.debug("langfuse_credentials_missing_skipping_initialization")
            return None

        try:
            client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            # Verify connectivity by checking client is functional
            _LANGFUSE_CLIENT = client
            logger.info("langfuse_tracing_initialized", host=settings.langfuse_host)
            return client
        except Exception as e:
            logger.warning("langfuse_initialization_failed", error=str(e))
            return None


# ---------------------------------------------------------------------------
# Tracer abstraction (wraps Langfuse or provides no-op)
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """A no-op span used when Langfuse is not available."""

    def __init__(self) -> None:
        pass

    def end(self) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass


class _NoOpTracer:
    """A no-op tracer that creates no-op spans."""

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Generator[_NoOpSpan, None, None]:
        yield _NoOpSpan()


# ---------------------------------------------------------------------------
# Public tracer API
# ---------------------------------------------------------------------------


def get_tracer() -> _NoOpTracer | Any:
    """Return the global tracer.

    Returns a no-op tracer if Langfuse is not configured, so callers
    can always use the tracer without conditional checks.
    """
    client = _get_langfuse_client()
    if client is None:
        return _NoOpTracer()

    # Return the Langfuse client directly as the tracer
    # Langfuse supports start_as_current_span which is the OTel API we use
    return client


@contextmanager
def make_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[_NoOpSpan, None, None]:
    """Create a child span with the given name.

    This is a convenience wrapper that uses the global tracer's
    start_as_current_span context manager.

    Args:
        name: The span name (e.g., "visual_search", "fusion").
        attributes: Optional attributes to set on the span.

    Yields:
        The active span (no-op if tracing is disabled).
    """
    tracer = get_tracer()
    kwargs: dict[str, Any] = {}
    if attributes:
        kwargs["attributes"] = attributes

    with tracer.start_as_current_span(name, **kwargs) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span


def get_current_trace_url() -> str | None:
    """Get the Langfuse URL for the current trace, if available.

    Returns:
        The Langfuse trace URL, or None if tracing is not active.
    """
    client = _get_langfuse_client()
    if client is None:
        return None

    try:
        return client.get_trace_url()
    except Exception as e:
        logger.debug("get_current_trace_url_failed", error=str(e))
        return None


def get_current_trace_id() -> str | None:
    """Get the current trace ID, if available.

    Returns:
        The current trace ID as a string, or None.
    """
    client = _get_langfuse_client()
    if client is None:
        return None

    try:
        return client.get_current_trace_id()
    except Exception as e:
        logger.debug("get_current_trace_id_failed", error=str(e))
        return None


# ---------------------------------------------------------------------------
# Trace helpers for specific operations
# ---------------------------------------------------------------------------


def trace_retrieval_span(
    name: str,
    candidates_count: int,
    latency_ms: int,
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    """Record a retrieval sub-span end event.

    Args:
        name: Span name (e.g., "visual_search").
        candidates_count: Number of candidates returned.
        latency_ms: Latency in milliseconds.
        extra_attrs: Additional attributes to record.
    """
    tracer = get_tracer()
    if isinstance(tracer, _NoOpTracer):
        return

    attrs: dict[str, Any] = {
        "candidates": candidates_count,
        "latency_ms": latency_ms,
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    try:
        with tracer.start_as_current_span(name, attributes=attrs) as span:
            span.end()
    except Exception as e:
        logger.warning("failed_to_record_trace_span", span_name=name, error=str(e))


def trace_generation_span(
    name: str,
    model: str,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: int | None = None,
    cost_usd: float | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    """Record a generation sub-span end event.

    Args:
        name: Span name (e.g., "vlm_generate").
        model: Model used (e.g., "gemini-2.5-flash").
        prompt_tokens: Number of prompt/input tokens.
        output_tokens: Number of output tokens generated.
        latency_ms: Latency in milliseconds.
        cost_usd: Cost in USD.
        extra_attrs: Additional attributes to record.
    """
    tracer = get_tracer()
    if isinstance(tracer, _NoOpTracer):
        return

    attrs: dict[str, Any] = {"model": model}
    if prompt_tokens is not None:
        attrs["prompt_tokens"] = prompt_tokens
    if output_tokens is not None:
        attrs["output_tokens"] = output_tokens
    if latency_ms is not None:
        attrs["latency_ms"] = latency_ms
    if cost_usd is not None:
        attrs["cost_usd"] = cost_usd
    if extra_attrs:
        attrs.update(extra_attrs)

    try:
        with tracer.start_as_current_span(name, attributes=attrs) as span:
            span.end()
    except Exception as e:
        logger.warning("failed_to_record_trace_span", span_name=name, error=str(e))


# ---------------------------------------------------------------------------
# Initialization check
# ---------------------------------------------------------------------------


def is_tracing_enabled() -> bool:
    """Check whether Langfuse tracing is enabled and configured.

    Returns:
        True if tracing is active, False otherwise.
    """
    return _get_langfuse_client() is not None


def verify_tracing_setup() -> dict[str, Any]:
    """Verify Langfuse tracing is properly set up.

    Returns a dict with the status and any issues found.
    """
    from urban_rag.common.settings import get_settings

    settings = get_settings()
    issues: list[str] = []
    is_configured = False

    if not settings.langfuse_enabled:
        issues.append("langfuse_enabled is False in settings")
    elif not settings.langfuse_public_key:
        issues.append("langfuse_public_key is not set")
    elif not settings.langfuse_secret_key:
        issues.append("langfuse_secret_key is not set")
    else:
        is_configured = True

    # Try to get the client
    client = _get_langfuse_client()
    client_status = "connected" if client is not None else "not_initialized"

    return {
        "is_configured": is_configured,
        "is_connected": client is not None,
        "client_status": client_status,
        "issues": issues,
        "langfuse_enabled": settings.langfuse_enabled,
        "langfuse_host": settings.langfuse_host,
    }
