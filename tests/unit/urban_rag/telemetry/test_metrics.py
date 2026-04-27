"""Unit tests for urban_rag/telemetry/metrics.py."""

from __future__ import annotations

import pytest

from prometheus_client import REGISTRY


class TestMetricsRecording:
    """Test that metrics can be recorded without errors."""

    def test_record_query_increments_counter(self) -> None:
        """record_query increments the query_total counter."""
        from urban_rag.telemetry.metrics import record_query, QUERY_TOTAL

        # Get initial value
        initial_value = self._get_counter_value(QUERY_TOTAL, {"mode": "fast", "status": "success"})

        # Record a query
        record_query(mode="fast", status="success")

        # Verify incremented
        new_value = self._get_counter_value(QUERY_TOTAL, {"mode": "fast", "status": "success"})
        assert new_value == initial_value + 1

    def test_record_latency_records_histogram(self) -> None:
        """record_latency observes a latency value."""
        from urban_rag.telemetry.metrics import record_latency

        # Should not raise
        record_latency(mode="fast", status="success", latency_seconds=1.5)

    def test_record_qdrant_latency_records_per_channel(self) -> None:
        """record_qdrant_latency records per channel."""
        from urban_rag.telemetry.metrics import record_qdrant_latency

        for channel in ("visual", "text", "sparse"):
            record_qdrant_latency(channel=channel, latency_seconds=0.05)

    def test_record_cost_increments_cost_counter(self) -> None:
        """record_cost increments the cost_usd counter."""
        from urban_rag.telemetry.metrics import record_cost, COST_USD

        initial_value = self._get_counter_value(COST_USD, {"mode": "fast"})

        record_cost(mode="fast", cost_usd=0.002)

        new_value = self._get_counter_value(COST_USD, {"mode": "fast"})
        assert new_value >= initial_value + 0.002

    def test_record_retrieval_candidates_records_histogram(self) -> None:
        """record_retrieval_candidates observes candidate counts."""
        from urban_rag.telemetry.metrics import record_retrieval_candidates

        for channel in ("visual", "text", "sparse", "fused"):
            record_retrieval_candidates(channel=channel, count=20)

    def test_record_tokens_increments_counter(self) -> None:
        """record_tokens increments the tokens counter."""
        from urban_rag.telemetry.metrics import record_tokens, TOKENS_TOTAL

        initial = self._get_counter_value(TOKENS_TOTAL, {"mode": "fast", "token_type": "prompt"})
        record_tokens(mode="fast", token_type="prompt", count=1000)
        new = self._get_counter_value(TOKENS_TOTAL, {"mode": "fast", "token_type": "prompt"})
        assert new >= initial + 1000

    @staticmethod
    def _get_counter_value(counter, labels: dict) -> float:
        """Get the current value of a labelled counter."""
        return counter.labels(**labels)._value.get()

    def test_get_metrics_returns_bytes(self) -> None:
        """get_metrics returns Prometheus text format bytes."""
        from urban_rag.telemetry.metrics import get_metrics, get_content_type

        output = get_metrics()
        assert isinstance(output, bytes)
        assert b"query_total" in output or b"cost_usd_total" in output
        # Verify content type
        content_type = get_content_type()
        assert "text/plain" in content_type or "openmetrics" in content_type

    def test_query_total_metric_exists(self) -> None:
        """QUERY_TOTAL counter is defined with correct labels."""
        from urban_rag.telemetry.metrics import QUERY_TOTAL

        # Should be able to access with valid labels
        c = QUERY_TOTAL.labels(mode="fast", status="success")
        assert c is not None

    def test_query_latency_histogram_exists(self) -> None:
        """QUERY_LATENCY_SECONDS histogram is defined."""
        from urban_rag.telemetry.metrics import QUERY_LATENCY_SECONDS

        h = QUERY_LATENCY_SECONDS.labels(mode="fast", status="success")
        assert h is not None
        # Should be able to observe a value
        h.observe(1.5)

    def test_qdrant_latency_histogram_exists(self) -> None:
        """QDRANT_LATENCY_SECONDS histogram is defined."""
        from urban_rag.telemetry.metrics import QDRANT_LATENCY_SECONDS

        h = QDRANT_LATENCY_SECONDS.labels(channel="visual")
        assert h is not None
        h.observe(0.05)

    def test_cost_counter_exists(self) -> None:
        """COST_USD counter is defined."""
        from urban_rag.telemetry.metrics import COST_USD

        c = COST_USD.labels(mode="deep")
        assert c is not None
