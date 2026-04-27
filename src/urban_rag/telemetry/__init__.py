"""OpenTelemetry traces + Prometheus metrics."""

from urban_rag.telemetry.metrics import (
    get_content_type,
    get_metrics,
    record_cost,
    record_embed_cold_start,
    record_latency,
    record_qdrant_latency,
    record_query,
    record_retrieval_candidates,
    record_tokens,
    set_faithfulness_p50,
)
from urban_rag.telemetry.tracing import (
    get_current_trace_id,
    get_current_trace_url,
    get_tracer,
    is_tracing_enabled,
    make_span,
    trace_generation_span,
    trace_retrieval_span,
    verify_tracing_setup,
)

__all__ = [
    "get_content_type",
    "get_current_trace_id",
    "get_current_trace_url",
    "get_metrics",
    "get_tracer",
    "is_tracing_enabled",
    "make_span",
    "record_cost",
    "record_embed_cold_start",
    "record_latency",
    "record_qdrant_latency",
    "record_query",
    "record_retrieval_candidates",
    "record_tokens",
    "set_faithfulness_p50",
    "trace_generation_span",
    "trace_retrieval_span",
    "verify_tracing_setup",
]
