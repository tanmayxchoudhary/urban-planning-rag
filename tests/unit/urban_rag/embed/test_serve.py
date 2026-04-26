"""Unit tests for src/urban_rag/embed/serve.py"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from urban_rag.embed.colqwen import VisualEmbedder


class TestEmbedServiceRequests:
    """Tests for request/response models."""

    def test_embed_request_valid(self) -> None:
        """Test EmbedRequest with valid data."""
        from urban_rag.embed.serve import EmbedRequest

        req = EmbedRequest(image_paths=["/path/to/image.png"])
        assert req.image_paths == ["/path/to/image.png"]
        assert req.batch_size is None

    def test_embed_request_with_batch_size(self) -> None:
        """Test EmbedRequest with explicit batch_size."""
        from urban_rag.embed.serve import EmbedRequest

        req = EmbedRequest(image_paths=["/a.png", "/b.png"], batch_size=10)
        assert req.batch_size == 10

    def test_embed_request_empty_rejected(self) -> None:
        """Test EmbedRequest rejects empty list."""
        from urban_rag.embed.serve import EmbedRequest

        with pytest.raises(Exception):
            EmbedRequest(image_paths=[])

    def test_embed_query_request_valid(self) -> None:
        """Test EmbedQueryRequest with valid query."""
        from urban_rag.embed.serve import EmbedQueryRequest

        req = EmbedQueryRequest(query="What is FSI?")
        assert req.query == "What is FSI?"
        assert req.max_length is None

    def test_embed_query_request_empty_rejected(self) -> None:
        """Test EmbedQueryRequest rejects empty query."""
        from urban_rag.embed.serve import EmbedQueryRequest

        with pytest.raises(Exception):
            EmbedQueryRequest(query="")

    def test_embed_response_structure(self) -> None:
        """Test EmbedResponse model structure."""
        from urban_rag.embed.serve import EmbedResponse

        resp = EmbedResponse(
            embeddings=[[[1.0, 2.0], [3.0, 4.0]]],
            model_id="test-model",
            device="cpu",
            batch_size=1,
            latency_ms=100,
        )
        assert resp.model_id == "test-model"
        assert resp.device == "cpu"
        assert resp.batch_size == 1
        assert resp.latency_ms == 100


class TestEmbedServiceEndpoints:
    """Tests for FastAPI endpoints using mocked embedder."""

    def test_health_endpoint_no_model_load(self) -> None:
        """Test /health does not trigger model loading."""
        from urban_rag.embed import serve

        # Reset global state
        serve._embedder = None

        # Mock get_embedder to verify it's called
        mock_embedder = MagicMock(spec=VisualEmbedder)
        mock_embedder.is_loaded = False
        mock_embedder.model_id = "test-model"
        mock_embedder.device = "cpu"
        mock_embedder.embedding_dim = 128

        with patch.object(serve, "get_embedder_instance", return_value=mock_embedder):
            client = TestClient(serve.app)
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False
        assert data["model_id"] == "test-model"
        assert data["device"] == "cpu"
        assert data["embedding_dim"] == 128

        # Reset global state
        serve._embedder = None

    def test_embed_endpoint_invalid_path(self) -> None:
        """Test POST /embed with nonexistent paths returns 400."""
        from urban_rag.embed import serve

        serve._embedder = None

        client = TestClient(serve.app)
        response = client.post(
            "/embed",
            json={"image_paths": ["/nonexistent/path.png"]},
        )

        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

        serve._embedder = None

    def test_embed_endpoint_empty_request(self) -> None:
        """Test POST /embed with empty paths list returns 400."""
        from urban_rag.embed import serve

        serve._embedder = None

        client = TestClient(serve.app)
        response = client.post(
            "/embed",
            json={"image_paths": []},
        )

        # FastAPI validation rejects empty list via min_length
        assert response.status_code == 422

        serve._embedder = None

    def test_embed_query_endpoint_empty_query(self) -> None:
        """Test POST /embed_query with empty query returns 400."""
        from urban_rag.embed import serve

        serve._embedder = None

        client = TestClient(serve.app)
        response = client.post(
            "/embed_query",
            json={"query": "   "},
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

        serve._embedder = None

    def test_embed_endpoint_success_with_mock(self) -> None:
        """Test POST /embed success path with mocked embedder."""
        from urban_rag.embed import serve
        import torch

        serve._embedder = None

        # Create a real-looking mock embedder
        mock_embedder = MagicMock(spec=VisualEmbedder)
        mock_embedder.is_loaded = True
        mock_embedder.model_id = "vidore/colqwen2.5-v0.2"
        mock_embedder.device = "cpu"
        mock_embedder.embedding_dim = 128
        # Return a proper-shaped tensor: (1, num_patches, 128)
        mock_embedder.embed_pages.return_value = torch.randn(1, 5, 128)

        with patch.object(serve, "get_embedder_instance", return_value=mock_embedder):
            with patch.object(Path, "exists", return_value=True):
                client = TestClient(serve.app)
                response = client.post(
                    "/embed",
                    json={"image_paths": ["/test/page.png"]},
                )

        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert data["model_id"] == "vidore/colqwen2.5-v0.2"
        assert data["device"] == "cpu"
        assert data["batch_size"] == 1
        assert "latency_ms" in data

        serve._embedder = None

    def test_embed_query_endpoint_success_with_mock(self) -> None:
        """Test POST /embed_query success path with mocked embedder."""
        from urban_rag.embed import serve
        import torch

        serve._embedder = None

        mock_embedder = MagicMock(spec=VisualEmbedder)
        mock_embedder.is_loaded = True
        mock_embedder.model_id = "vidore/colqwen2.5-v0.2"
        mock_embedder.device = "cpu"
        mock_embedder.embedding_dim = 128
        # Return (1, query_tokens, 128) -> squeeze to (query_tokens, 128)
        mock_embedder.embed_query.return_value = torch.randn(1, 8, 128)

        with patch.object(serve, "get_embedder_instance", return_value=mock_embedder):
            client = TestClient(serve.app)
            response = client.post(
                "/embed_query",
                json={"query": "What is the FSI for residential?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "query_embedding" in data
        assert data["model_id"] == "vidore/colqwen2.5-v0.2"
        assert data["device"] == "cpu"
        assert "latency_ms" in data

        serve._embedder = None


class TestEmbedServiceLifecycle:
    """Tests for application lifecycle management."""

    def test_embedder_released_on_shutdown(self) -> None:
        """Test that embedder is released when app shuts down."""
        from urban_rag.embed import serve
        import urban_rag.embed.colqwen as colqwen_module

        serve._embedder = None
        colqwen_module._embedder_instance = None

        mock_embedder = MagicMock(spec=VisualEmbedder)
        serve._embedder = mock_embedder
        colqwen_module._embedder_instance = mock_embedder

        import asyncio
        asyncio.run(serve.shutdown_event())

        assert serve._embedder is None
        assert colqwen_module._embedder_instance is None
        mock_embedder.unload.assert_called_once()

        # Cleanup
        serve._embedder = None
        colqwen_module._embedder_instance = None

    def test_get_embedder_instance_creates_once(self) -> None:
        """Test get_embedder_instance creates singleton once."""
        from urban_rag.embed import serve

        serve._embedder = None

        mock_embedder = MagicMock(spec=VisualEmbedder)
        mock_embedder.is_loaded = False

        with patch.object(serve, "get_embedder", return_value=mock_embedder):
            result1 = serve.get_embedder_instance()
            result2 = serve.get_embedder_instance()

        assert result1 is result2
        assert result1 is mock_embedder

        serve._embedder = None
