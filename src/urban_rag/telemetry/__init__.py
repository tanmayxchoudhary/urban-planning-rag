"""OpenTelemetry traces + Prometheus metrics."""

from urban_rag.telemetry.tracing import (
    get_tracer,
    make_span,
    get_current_trace_url,
    get_current_trace_id,
    is_tracing_enabled,
    verify_tracing_setup,
    trace_retrieval_span,
    trace_generation_span,
)

__all__ = [
    "get_tracer",
    "make_span",
    "get_current_trace_url",
    "get_current_trace_id",
    "is_tracing_enabled",
    "verify_tracing_setup",
    "trace_retrieval_span",
    "trace_generation_span",
]
