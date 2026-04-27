"""Prometheus metrics for Urban Planning RAG.

Per PLAN.md Part XIII §13.2:
    Core metrics exported to Prometheus for scraping:
    - query_total        Counter — total queries, labelled by status, mode
    - query_latency_seconds  Histogram — end-to-end query latency with latency buckets
    - qdrant_latency_seconds  Histogram — Qdrant query latency per channel
    - cost_usd           Counter — Gemini API cost in USD

Usage:
    from urban_rag.telemetry.metrics import (
        record_query,
        record_latency,
        record_qdrant_latency,
        record_cost,
        get_metrics,
    )

    # At query start:
    record_query(mode="fast", status="started")

    # At query end:
    record_latency(mode="fast", status="success", latency_seconds=1.23)

    # At retrieval time (per channel):
    record_qdrant_latency(channel="visual", latency_seconds=0.05)

    # At generation completion:
    record_cost(cost_usd=0.0021)
"""

from __future__ import annotations

from typing import Literal

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

#: Total number of queries submitted, labelled by mode and terminal status.
QUERY_TOTAL = Counter(
    "query_total",
    "Total number of queries submitted to the RAG API",
    ["mode", "status"],  # mode: fast|deep, status: success|error|refused
)

#: End-to-end query latency in seconds, from submission to final answer.
#: Buckets chosen to cover p50 (~1.8s), p95 (~4s), p99 (~8s) per PLAN.md §2.4.
QUERY_LATENCY_SECONDS = Histogram(
    "query_latency_seconds",
    "End-to-end query latency in seconds",
    ["mode", "status"],
    buckets=(0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0),
)

#: Qdrant retrieval latency per channel (visual, text, sparse) in seconds.
QDRANT_LATENCY_SECONDS = Histogram(
    "qdrant_latency_seconds",
    "Qdrant retrieval latency per channel in seconds",
    ["channel"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5),
)

#: Gemini API cost in USD per query.
COST_USD = Counter(
    "cost_usd_total",
    "Cumulative Gemini API cost in USD",
    ["mode"],  # mode: fast|deep
)

#: Retrieval candidate counts per channel and fused total.
RETRIEVAL_CANDIDATES = Histogram(
    "retrieval_candidates",
    "Number of retrieval candidates per channel",
    ["channel"],
    buckets=(1, 5, 10, 20, 50, 100, 200),
)

#: Number of tokens processed per query (prompt + output).
TOKENS_TOTAL = Counter(
    "tokens_total",
    "Total tokens processed (prompt + output)",
    ["mode", "token_type"],  # token_type: prompt|output
)

# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------


def record_query(mode: Literal["fast", "deep"], status: str) -> None:
    """Increment the total query counter.

    Args:
        mode: Query mode (fast or deep).
        status: Terminal status (success, error, refused).
    """
    QUERY_TOTAL.labels(mode=mode, status=status).inc()


def record_latency(
    mode: Literal["fast", "deep"],
    status: str,
    latency_seconds: float,
) -> None:
    """Record end-to-end query latency.

    Args:
        mode: Query mode (fast or deep).
        status: Terminal status (success, error, refused).
        latency_seconds: Latency in seconds.
    """
    QUERY_LATENCY_SECONDS.labels(mode=mode, status=status).observe(latency_seconds)


def record_qdrant_latency(
    channel: Literal["visual", "text", "sparse"],
    latency_seconds: float,
) -> None:
    """Record Qdrant retrieval latency for a specific channel.

    Args:
        channel: The retrieval channel (visual, text, sparse).
        latency_seconds: Latency in seconds.
    """
    QDRANT_LATENCY_SECONDS.labels(channel=channel).observe(latency_seconds)


def record_cost(mode: Literal["fast", "deep"], cost_usd: float) -> None:
    """Record Gemini API cost for a query.

    Args:
        mode: Query mode (fast or deep).
        cost_usd: Cost in USD.
    """
    COST_USD.labels(mode=mode).inc(cost_usd)


def record_retrieval_candidates(channel: str, count: int) -> None:
    """Record the number of retrieval candidates from a channel.

    Args:
        channel: The retrieval channel (visual, text, sparse, fused).
        count: Number of candidates.
    """
    RETRIEVAL_CANDIDATES.labels(channel=channel).observe(count)


def record_tokens(mode: Literal["fast", "deep"], token_type: str, count: int) -> None:
    """Record token counts for a query.

    Args:
        mode: Query mode (fast or deep).
        token_type: Type of tokens (prompt or output).
        count: Number of tokens.
    """
    TOKENS_TOTAL.labels(mode=mode, token_type=token_type).inc(count)


# ---------------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------------


def get_metrics() -> bytes:
    """Generate Prometheus metrics output in text format.

    Returns:
        Metrics in Prometheus text exposition format.
    """
    return generate_latest()


def get_content_type() -> str:
    """Return the content type for Prometheus metrics.

    Returns:
        Content type string.
    """
    return CONTENT_TYPE_LATEST
