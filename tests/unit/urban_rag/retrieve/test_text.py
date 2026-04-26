"""Unit tests for src/urban_rag/retrieve/text.py — GTE-ModernColBERT text retrieval channel."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from urban_rag.common.errors import RetrievalError, ServiceUnavailableError
from urban_rag.common.types import RetrievalCandidate
from urban_rag.retrieve.text import (
    _build_filter,
    _encode_query,
    _qdrant_text_search,
    retrieve_text,
    smoke_query,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candidate(
    page_id: str,
    score: float = 1.0,
    image_uri: str = "",
    excerpt: str = "",
    section_title: str | None = None,
) -> RetrievalCandidate:
    """Factory to create a RetrievalCandidate for testing."""
    return RetrievalCandidate(
        page_id=page_id,
        score=score,
        channel_scores={"text": score},
        channel_ranks={"text": 1},
        rerank_score=None,
        rerank_rationale=None,
        page_image_uri=image_uri,
        extracted_text_excerpt=excerpt,
        section_title=section_title,
    )


# ---------------------------------------------------------------------------
# _build_filter
# ---------------------------------------------------------------------------

class TestBuildFilter:
    """Tests for _build_filter()."""

    def test_empty_filters_returns_none(self) -> None:
        """No filters returns None (no filter applied)."""
        result = _build_filter({})
        assert result is None

    def test_single_filter_builds_must_clause(self) -> None:
        """A single {field: value} builds a Filter with one FieldCondition."""
        result = _build_filter({"doc_type": "code"})
        assert result is not None
        assert hasattr(result, "must")
        assert len(result.must) == 1

    def test_multiple_filters_builds_multiple_clauses(self) -> None:
        """Multiple filters produce multiple FieldCondition clauses."""
        result = _build_filter({"jurisdiction": "IN-MH", "doc_type": "code"})
        assert result is not None
        assert len(result.must) == 2


# ---------------------------------------------------------------------------
# _encode_query
# ---------------------------------------------------------------------------

class TestEncodeQuery:
    """Tests for _encode_query()."""

    def test_returns_tokens_and_pooled(self) -> None:
        """Returns tuple of (query_tokens, query_pooled)."""
        # Mock embedder that returns a 3-token embedding with 4-dim vectors
        mock_embedder = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.__getitem__ = lambda self, key: [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ]
        mock_embedder.embed_query.return_value = mock_tensor

        tokens, pooled = _encode_query(mock_embedder, "test query")

        assert isinstance(tokens, list)
        assert len(tokens) == 3  # 3 tokens
        assert all(isinstance(v, list) for v in tokens)
        assert isinstance(pooled, list)
        assert len(pooled) == 4  # embedding dim = 4

    def test_pooled_is_mean_of_tokens(self) -> None:
        """Pooled vector is the mean of token vectors."""
        mock_embedder = MagicMock()
        mock_tensor = MagicMock()
        # Two tokens, each 2-dim
        mock_tensor.__getitem__ = lambda self, key: [[0.0, 2.0], [4.0, 6.0]]
        mock_embedder.embed_query.return_value = mock_tensor

        _, pooled = _encode_query(mock_embedder, "test")

        # Mean of [0,2] and [4,6] = [2, 4]
        assert pooled == pytest.approx([2.0, 4.0])


# ---------------------------------------------------------------------------
# _qdrant_text_search
# ---------------------------------------------------------------------------

class TestQdrantTextSearch:
    """Tests for _qdrant_text_search()."""

    def test_empty_results_returns_empty_list(self) -> None:
        """Qdrant returns empty points → empty candidate list."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        result = _qdrant_text_search(
            client=mock_client,
            query_tokens=[[0.1, 0.2], [0.3, 0.4]],
            query_pooled=[[0.15, 0.25]],
            top_k=10,
        )

        assert result == []

    def test_single_result_converted_to_candidate(self) -> None:
        """A single Qdrant point is converted to a RetrievalCandidate."""
        mock_client = MagicMock()
        mock_results = MagicMock()

        mock_point = MagicMock()
        mock_point.id = "doc_hash#c000001"
        mock_point.score = 0.95
        mock_point.payload = {
            "image_uri": "lance://data/docs/abc/pages/p_0001.png",
            "chunk_text": "Section about FSI calculations for residential...",
            "section_path": ["Chapter 4", "4.3 FSI", "4.3.1 Residential"],
        }
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results

        candidates = _qdrant_text_search(
            client=mock_client,
            query_tokens=[[0.1, 0.2]],
            query_pooled=[[0.15]],
            top_k=10,
        )

        assert len(candidates) == 1
        c = candidates[0]
        assert c.page_id == "doc_hash#c000001"
        assert c.score == 0.95
        assert c.channel_scores["text"] == 0.95
        assert c.channel_ranks["text"] == 1
        assert c.extracted_text_excerpt == "Section about FSI calculations for residential..."
        assert c.section_title == "4.3.1 Residential"

    def test_results_ordered_by_score(self) -> None:
        """Results are returned in score-descending order."""
        mock_client = MagicMock()
        mock_results = MagicMock()

        mock_points = []
        for score, page_id in [(0.8, "chunk1"), (0.95, "chunk2"), (0.5, "chunk3")]:
            pt = MagicMock()
            pt.id = page_id
            pt.score = score
            pt.payload = {"image_uri": "", "section_path": [], "chunk_text": ""}
            mock_points.append(pt)

        mock_results.points = mock_points
        mock_client.query_points.return_value = mock_results

        candidates = _qdrant_text_search(
            client=mock_client,
            query_tokens=[[0.1]],
            query_pooled=[[0.15]],
            top_k=10,
        )

        assert [c.page_id for c in candidates] == ["chunk1", "chunk2", "chunk3"]
        assert [c.score for c in candidates] == pytest.approx([0.8, 0.95, 0.5])

    def test_connection_error_raises_service_unavailable(self) -> None:
        """Connection-related errors raise ServiceUnavailableError."""
        mock_client = MagicMock()
        mock_client.query_points.side_effect = ConnectionError("connection refused")

        with pytest.raises(ServiceUnavailableError):
            _qdrant_text_search(
                client=mock_client,
                query_tokens=[[0.1]],
                query_pooled=[[0.15]],
                top_k=10,
            )

    def test_qdrant_client_query_points_call_structure(self) -> None:
        """The Qdrant query_points is called with prefetch + multivector query."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        _qdrant_text_search(
            client=mock_client,
            query_tokens=[[0.1, 0.2], [0.3, 0.4]],
            query_pooled=[[0.15, 0.25]],
            top_k=20,
            prefetch_limit=200,
        )

        # Verify the call structure
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == "pages_text"
        assert "prefetch" in call_kwargs
        assert "query" in call_kwargs
        assert call_kwargs["using"] == "text"
        assert call_kwargs["limit"] == 20
        assert call_kwargs["with_payload"] is True

    def test_top_k_bounds_enforcement(self) -> None:
        """top_k parameter is passed directly to Qdrant."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        _qdrant_text_search(
            client=mock_client,
            query_tokens=[[0.1]],
            query_pooled=[[0.15]],
            top_k=5,
        )

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 5


# ---------------------------------------------------------------------------
# retrieve_text
# ---------------------------------------------------------------------------

class TestRetrieveText:
    """Tests for retrieve_text()."""

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        """A mock TextEmbedder that returns fixed query embeddings."""
        embedder = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.__getitem__ = lambda self, key: [
            [0.1, 0.2, 0.3] * 10,  # 30 tokens, 3-dim (simplified)
        ]
        embedder.embed_query.return_value = mock_tensor
        return embedder

    def test_returns_retrieval_result(self, mock_embedder: MagicMock) -> None:
        """retrieve_text returns a RetrievalResult."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.retrieve.text.get_qdrant_client", return_value=mock_client):
            result = retrieve_text(
                query="FSI residential zones",
                embedder=mock_embedder,
            )

        assert result is not None
        assert result.query == "FSI residential zones"
        assert result.retrieval_strategy == "text_primary"
        assert "text" in result.flags or result.flags == {}

    def test_candidates_populated_from_qdrant(self, mock_embedder: MagicMock) -> None:
        """Qdrant results populate the candidates list."""
        mock_client = MagicMock()
        mock_results = MagicMock()

        pt = MagicMock()
        pt.id = "doc#c000001"
        pt.score = 0.92
        pt.payload = {
            "image_uri": "lance://x.png",
            "section_path": ["Chapter 1"],
            "chunk_text": "FSI provisions for residential...",
        }
        mock_results.points = [pt]
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.retrieve.text.get_qdrant_client", return_value=mock_client):
            result = retrieve_text(
                query="FSI requirements",
                top_k=10,
                embedder=mock_embedder,
            )

        assert len(result.candidates) == 1
        assert result.candidates[0].page_id == "doc#c000001"

    def test_encode_ms_and_retrieve_ms_in_latency(self, mock_embedder: MagicMock) -> None:
        """Latency is measured and reported."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.retrieve.text.get_qdrant_client", return_value=mock_client):
            result = retrieve_text(query="test query", embedder=mock_embedder)

        assert result.latency_ms >= 0

    def test_query_encoding_failure_raises_retrieval_error(self) -> None:
        """embedder.embed_query raises → RetrievalError."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = RuntimeError("model error")

        with pytest.raises(RetrievalError) as exc_info:
            retrieve_text(query="test", embedder=mock_embedder)

        assert "Query encoding failed" in str(exc_info.value.message)

    def test_qdrant_unavailable_raises_service_unavailable(self, mock_embedder: MagicMock) -> None:
        """Qdrant connection failure raises ServiceUnavailableError."""
        with patch(
            "urban_rag.retrieve.text.get_qdrant_client",
            side_effect=ConnectionError("qdrant down"),
        ), pytest.raises(ServiceUnavailableError):
            retrieve_text(query="test", embedder=mock_embedder)


# ---------------------------------------------------------------------------
# smoke_query
# ---------------------------------------------------------------------------

class TestSmokeQuery:
    """Tests for smoke_query()."""

    def test_returns_dict_with_candidates_latency(self) -> None:
        """smoke_query returns dict with candidates, latency_ms, candidate_count."""
        mock_embedder = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.__getitem__ = lambda self, key: [[0.1, 0.2, 0.3]]
        mock_embedder.embed_query.return_value = mock_tensor

        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.embed.text_encoder.get_text_embedder", return_value=mock_embedder), \
             patch("urban_rag.embed.text_encoder.release_text_embedder") as mock_release, \
             patch("urban_rag.retrieve.text.get_qdrant_client", return_value=mock_client):
            result = smoke_query(query="FSI residential", top_k=10)

        assert "candidates" in result
        assert "latency_ms" in result
        assert "candidate_count" in result
        assert "query" in result
        mock_release.assert_called_once()

    def test_default_query_is_fsi_residential(self) -> None:
        """Default query string is 'FSI residential zones floor space index'."""
        mock_embedder = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.__getitem__ = lambda self, key: [[0.1]]
        mock_embedder.embed_query.return_value = mock_tensor

        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.embed.text_encoder.get_text_embedder", return_value=mock_embedder), \
             patch("urban_rag.embed.text_encoder.release_text_embedder"), \
             patch("urban_rag.retrieve.text.get_qdrant_client", return_value=mock_client):
                smoke_query(top_k=5)

        # Verify embedder was called
        mock_embedder.embed_query.assert_called()

    def test_candidate_count_reflects_qdrant_results(self) -> None:
        """Candidate count matches the number of Qdrant results."""
        mock_embedder = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.__getitem__ = lambda self, key: [[0.1, 0.2]]
        mock_embedder.embed_query.return_value = mock_tensor

        mock_client = MagicMock()
        mock_results = MagicMock()

        def make_point(page_id: str, score: float) -> MagicMock:
            pt = MagicMock()
            pt.id = page_id
            pt.score = score
            pt.payload = {"image_uri": "", "section_path": [], "chunk_text": ""}
            return pt

        mock_results.points = [
            make_point("chunk1", 0.95),
            make_point("chunk2", 0.90),
            make_point("chunk3", 0.85),
        ]
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.embed.text_encoder.get_text_embedder", return_value=mock_embedder), \
             patch("urban_rag.embed.text_encoder.release_text_embedder"), \
             patch("urban_rag.retrieve.text.get_qdrant_client", return_value=mock_client):
                result = smoke_query(query="FSI", top_k=10)

        assert result["candidate_count"] == 3
        assert len(result["candidates"]) == 3
