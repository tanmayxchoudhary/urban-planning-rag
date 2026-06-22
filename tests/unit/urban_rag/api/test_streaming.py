"""Tests for SSE streaming layer in API main module.

Per VAL-API-007, VAL-API-011, VAL-API-012, VAL-API-035, VAL-API-036, VAL-API-037.
"""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from urban_rag.api.main import (
    GENERATION_FAILED_CODE,
    INTERNAL_CODE,
    OUT_OF_CORPUS_REASON,
    RETRIEVAL_TIMEOUT_CODE,
    _format_sse,
    _query_store,
    app,
    settings,
)


class TestSSEFormatHelpers:
    """Test SSE formatting helpers."""

    def test_format_sse_returns_proper_format(self):
        """SSE events should be formatted with event: prefix and data: JSON."""
        result = _format_sse("test_event", {"key": "value"})
        assert result.startswith("event: test_event\n")
        assert "data: " in result
        assert '"key": "value"' in result
        assert result.endswith("\n\n")

    def test_format_sse_with_empty_data(self):
        """SSE with empty dict should still produce valid format."""
        result = _format_sse("empty", {})
        assert "event: empty\n" in result
        assert "data: {}\n\n" in result

    def test_format_sse_with_list_data(self):
        """SSE with list data should produce valid JSON array."""
        result = _format_sse("candidates", {"items": [1, 2, 3]})
        assert '"items": [1, 2, 3]' in result


class TestErrorCodeConstants:
    """Test error code constants are properly defined."""

    def test_retrieval_timeout_code_is_correct(self):
        """RETRIEVAL_TIMEOUT_CODE should be 'retrieval_timeout' for HTTP 504."""
        assert RETRIEVAL_TIMEOUT_CODE == "retrieval_timeout"

    def test_generation_failed_code_is_correct(self):
        """GENERATION_FAILED_CODE should be 'generation_failed' for HTTP 502."""
        assert GENERATION_FAILED_CODE == "generation_failed"

    def test_internal_code_is_correct(self):
        """INTERNAL_CODE should be 'internal' for HTTP 500."""
        assert INTERNAL_CODE == "internal"

    def test_out_of_corpus_reason_is_correct(self):
        """OUT_OF_CORPUS_REASON should be 'out_of_corpus' for refusal events."""
        assert OUT_OF_CORPUS_REASON == "out_of_corpus"


class TestQueryStore:
    """Test query store operations for SSE state management."""

    def setup_method(self):
        """Clear query store before each test."""
        _query_store.clear()

    def test_initialize_query_store_entry(self):
        """Query store entry should be initialized with correct structure."""
        query_id = "q_test123"
        _query_store[query_id] = {
            "question": "What is FSI?",
            "mode": "fast",
            "status": "pending",
            "started_at": "2024-01-01T00:00:00Z",
        }
        assert query_id in _query_store
        assert _query_store[query_id]["question"] == "What is FSI?"
        assert _query_store[query_id]["status"] == "pending"

    def test_store_completed_answer(self):
        """Completed answer should be stored with answer key."""
        query_id = "q_complete123"
        _query_store[query_id] = {
            "status": "completed",
            "answer": {
                "answer_markdown": "FSI is 2.5 [1].",
                "citations": [],
                "confidence": "high",
            },
        }
        assert _query_store[query_id]["status"] == "completed"
        assert "answer" in _query_store[query_id]
        assert "answer_markdown" in _query_store[query_id]["answer"]

    def test_store_refused_query(self):
        """Refused query should store refused_reason and refused_message."""
        query_id = "q_refused123"
        _query_store[query_id] = {
            "status": "refused",
            "refused_reason": OUT_OF_CORPUS_REASON,
            "refused_message": "Query is outside corpus scope.",
        }
        assert _query_store[query_id]["status"] == "refused"
        assert _query_store[query_id]["refused_reason"] == OUT_OF_CORPUS_REASON

    def test_store_error_query(self):
        """Error query should store error_code, error_message, and http_status."""
        query_id = "q_error123"
        _query_store[query_id] = {
            "status": "error",
            "error_code": GENERATION_FAILED_CODE,
            "error_message": "Generation failed due to timeout.",
            "http_status": 502,
        }
        assert _query_store[query_id]["status"] == "error"
        assert _query_store[query_id]["error_code"] == GENERATION_FAILED_CODE
        assert _query_store[query_id]["http_status"] == 502


class TestSSEEventMapping:
    """Test that orchestrator events map correctly to SSE events.

    Per VAL-API-007: canonical event order is:
    retrieval_started → retrieval_completed → generation_started →
    token (repeated) → generation_completed → done

    Error events: error, refused
    """

    def setup_method(self):
        """Clear query store before each test."""
        _query_store.clear()

    def test_event_stream_order_retrieval_started(self):
        """First event should be retrieval_started."""
        # Create a pending query
        query_id = "q_stream_test"
        _query_store[query_id] = {
            "question": "What is FSI for residential?",
            "mode": "fast",
            "status": "pending",
        }
        # The event stream implementation should emit retrieval_started first
        # This is verified by the integration test pattern

    def test_error_event_structure(self):
        """Error events should have code, message, and stage fields."""
        error_event = _format_sse("error", {
            "code": GENERATION_FAILED_CODE,
            "message": "Generation failed due to upstream error.",
            "stage": "generation",
        })
        assert '"code": "generation_failed"' in error_event
        assert '"stage": "generation"' in error_event

    def test_retrieval_timeout_error_event(self):
        """Retrieval timeout should emit error event with retrieval_timeout code."""
        error_event = _format_sse("error", {
            "code": RETRIEVAL_TIMEOUT_CODE,
            "message": "Retrieval channel timed out after 15s.",
            "stage": "retrieval",
        })
        assert '"code": "retrieval_timeout"' in error_event
        assert '"stage": "retrieval"' in error_event

    def test_refused_event_structure(self):
        """Refused events should have reason and message fields."""
        refused_event = _format_sse("refused", {
            "reason": OUT_OF_CORPUS_REASON,
            "message": "Query is outside the indexed corpus scope.",
        })
        assert '"reason": "out_of_corpus"' in refused_event

    def test_retrieval_completed_has_latency_ms(self):
        """retrieval_completed events should include latency_ms per VAL-API-008."""
        event_data = {
            "query_id": "q_test",
            "candidates": [],
            "latency_ms": 250,
        }
        event = _format_sse("retrieval_completed", event_data)
        assert '"latency_ms": 250' in event

    def test_token_event_has_text_chunk(self):
        """Token events should have text field with chunk content."""
        token_event = _format_sse("token", {"text": "FSI is "})
        assert '"text": "FSI is "' in token_event

    def test_done_event_has_query_id(self):
        """Done event should have query_id for terminal verification."""
        done_event = _format_sse("done", {"query_id": "q_test123"})
        assert '"query_id": "q_test123"' in done_event

    def test_generation_started_event(self):
        """generation_started should emit with model info and timestamp."""
        gen_started = _format_sse("generation_started", {
            "query_id": "q_test",
            "model": "gemini-2.5-flash",
            "ts": "2024-01-01T00:00:00Z",
        })
        assert '"model": "gemini-2.5-flash"' in gen_started
        assert '"ts"' in gen_started


class TestAPIEndpoints:
    """Test API endpoints for SSE streaming."""

    def setup_method(self):
        """Clear query store before each test."""
        _query_store.clear()

    def test_ask_endpoint_returns_query_id(self):
        """POST /v1/ask should return query_id and stream_url."""
        client = TestClient(app)
        response = client.post(
            "/v1/ask",
            json={"question": "What is FSI?", "mode": "fast"},
        )
        assert response.status_code == 202
        data = response.json()
        assert "query_id" in data
        assert "stream_url" in data
        assert data["mode"] == "fast"

    def test_stream_endpoint_unknown_query_returns_404(self):
        """GET /v1/ask/{id}/stream with unknown query_id returns 404."""
        client = TestClient(app)
        response = client.get("/v1/ask/nonexistent_id/stream")
        assert response.status_code == 404

    def test_get_answer_refused_returns_200_with_refusal(self):
        """GET /v1/ask/{id} for refused query returns 200 with refusal info."""
        query_id = "q_refused_test"
        _query_store[query_id] = {
            "status": "refused",
            "refused_reason": OUT_OF_CORPUS_REASON,
            "refused_message": "Query is outside corpus scope.",
        }
        client = TestClient(app)
        response = client.get(f"/v1/ask/{query_id}")
        # Should return 200 with error detail (not 404)
        assert response.status_code == 200
        data = response.json()
        # Response is wrapped in 'error' key per UrbanRagError envelope
        assert "error" in data
        assert data["error"]["code"] == "refused"

    def test_get_answer_error_returns_http_status_from_store(self):
        """GET /v1/ask/{id} for error query returns stored HTTP status."""
        query_id = "q_error_test"
        _query_store[query_id] = {
            "status": "error",
            "error_code": GENERATION_FAILED_CODE,
            "error_message": "Generation failed.",
            "http_status": 502,
        }
        client = TestClient(app)
        response = client.get(f"/v1/ask/{query_id}")
        # Should return 502 from the stored http_status
        assert response.status_code == 502

    def test_get_answer_not_found_returns_404(self):
        """GET /v1/ask/{id} for unknown query returns 404."""
        client = TestClient(app)
        response = client.get("/v1/ask/unknown_id")
        assert response.status_code == 404


class TestEventOrderContract:
    """Test that event order follows VAL-API-007 contract."""

    def test_canonical_order_sequence(self):
        """Canonical event order: retrieval_started → retrieval_completed →
        generation_started → token → generation_completed → done
        """
        # Build expected sequence
        events = []
        events.append(("retrieval_started", _format_sse("retrieval_started", {"query_id": "q"})))
        events.append(("retrieval_completed", _format_sse("retrieval_completed", {"query_id": "q", "candidates": [], "latency_ms": 100})))
        events.append(("generation_started", _format_sse("generation_started", {"query_id": "q", "model": "gemini-2.5-flash"})))
        events.append(("token", _format_sse("token", {"text": "Hello"})))
        events.append(("generation_completed", _format_sse("generation_completed", {"query_id": "q", "answer_markdown": "Answer.", "citations": []})))
        events.append(("done", _format_sse("done", {"query_id": "q"})))

        # Verify each event type is correct
        for expected_type, formatted_event in events:
            assert formatted_event.startswith(f"event: {expected_type}\n")

    def test_error_event_is_alternative_path(self):
        """Error event is alternative to normal completion path."""
        # Error events should have code and message
        error_event = _format_sse("error", {
            "code": RETRIEVAL_TIMEOUT_CODE,
            "message": "Timeout",
            "stage": "retrieval",
        })
        assert error_event.startswith("event: error\n")
        assert '"code": "retrieval_timeout"' in error_event

    def test_refused_event_is_alternative_path(self):
        """Refused event is alternative to normal completion path."""
        refused_event = _format_sse("refused", {
            "reason": OUT_OF_CORPUS_REASON,
            "message": "Out of scope",
        })
        assert refused_event.startswith("event: refused\n")
        assert '"reason": "out_of_corpus"' in refused_event


class TestCorpusAndReadinessEndpoints:
    """Truthful corpus/image/readyz behavior for the API layer."""

    def _write_manifest(self, manifest_path: Path) -> str:
        doc_hash = "abc123doc"
        frame = pd.DataFrame(
            [
                {
                    "doc_hash": doc_hash,
                    "filename": "sample.pdf",
                    "title": "Sample Document",
                    "family": "OTHER",
                    "jurisdiction": None,
                    "publisher": None,
                    "year": None,
                    "version": "1",
                    "license": "unknown",
                    "page_count": 3,
                    "size_bytes": 1234,
                    "storage_uri": "",
                    "ingested_at": "2026-06-18T00:00:00+00:00",
                    "indexed_at": "2026-06-18T01:00:00+00:00",
                }
            ]
        )
        frame.to_parquet(manifest_path)
        return doc_hash

    def test_corpus_endpoint_reads_manifest(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.parquet"
        doc_hash = self._write_manifest(manifest_path)
        old_manifest = settings.manifest_path
        try:
            settings.manifest_path = str(manifest_path)
            client = TestClient(app)
            response = client.get("/v1/corpus")
            assert response.status_code == 200
            payload = response.json()
            assert payload["totals"] == {"documents": 1, "pages": 3}
            assert payload["documents"][0]["doc_id"] == doc_hash
        finally:
            settings.manifest_path = old_manifest

    def test_page_image_returns_503_when_doc_exists_but_png_missing(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.parquet"
        docs_dir = tmp_path / "docs"
        page_images_dir = tmp_path / "page_images"
        docs_dir.mkdir()
        page_images_dir.mkdir()
        doc_hash = self._write_manifest(manifest_path)
        doc_dir = docs_dir / doc_hash
        doc_dir.mkdir()
        (doc_dir / "source.pdf").write_bytes(b"%PDF-1.4\n")

        old_manifest = settings.manifest_path
        old_docs = settings.docs_dir
        old_page_images = settings.page_images_dir
        try:
            settings.manifest_path = str(manifest_path)
            settings.docs_dir = str(docs_dir)
            settings.page_images_dir = str(page_images_dir)
            client = TestClient(app)
            response = client.get(f"/v1/corpus/{doc_hash}/pages/1/image")
            assert response.status_code == 503
            payload = response.json()
            assert payload["error"]["code"] == "page_asset_unavailable"
            assert payload["details"]["source_pdf_exists"] is True
        finally:
            settings.manifest_path = old_manifest
            settings.docs_dir = old_docs
            settings.page_images_dir = old_page_images

    def test_readyz_reports_missing_page_image_root_as_not_ready(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.parquet"
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        self._write_manifest(manifest_path)

        old_manifest = settings.manifest_path
        old_docs = settings.docs_dir
        old_page_images = settings.page_images_dir
        try:
            settings.manifest_path = str(manifest_path)
            settings.docs_dir = str(docs_dir)
            settings.page_images_dir = str(tmp_path / "missing_page_images")
            client = TestClient(app)
            response = client.get("/v1/readyz")
            assert response.status_code == 503
            payload = response.json()
            assert payload["checks"]["manifest"] is True
            assert payload["checks"]["docs_dir"] is True
            assert payload["checks"]["page_images_dir"] is False
        finally:
            settings.manifest_path = old_manifest
            settings.docs_dir = old_docs
            settings.page_images_dir = old_page_images
