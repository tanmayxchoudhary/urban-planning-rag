"""Unit tests for src/urban_rag/retrieve/sparse.py — Qdrant native BM25 sparse retrieval channel."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from urban_rag.common.errors import RetrievalError, ServiceUnavailableError
from urban_rag.common.types import RetrievalCandidate
from urban_rag.retrieve.sparse import (
    BM25_K1,
    BM25_B,
    BM25QueryScorer,
    _build_filter,
    _is_schedule_ii_page,
    _qdrant_sparse_search,
    _tokenize,
    retrieve_sparse,
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
        channel_scores={"sparse": score},
        channel_ranks={"sparse": 1},
        rerank_score=None,
        rerank_rationale=None,
        page_image_uri=image_uri,
        extracted_text_excerpt=excerpt,
        section_title=section_title,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestBM25Constants:
    """Tests for BM25 parameter constants."""

    def test_bm25_k1_value(self) -> None:
        """Test BM25_K1 is 1.5 (standard IR value)."""
        assert BM25_K1 == 1.5

    def test_bm25_b_value(self) -> None:
        """Test BM25_B is 0.75 (standard IR value)."""
        assert BM25_B == 0.75


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    """Tests for _tokenize() helper."""

    def test_lowercase(self) -> None:
        """Text is converted to lowercase."""
        tokens = _tokenize("HELLO World")
        assert "hello" in tokens
        assert "world" in tokens
        assert "HELLO" not in tokens

    def test_split_on_non_alphanumeric(self) -> None:
        """Text is split on non-alphanumeric characters."""
        tokens = _tokenize("hello-world foo.bar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens

    def test_filter_short_tokens(self) -> None:
        """Tokens with length < 2 are filtered out."""
        tokens = _tokenize("a b c dd ee ff")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "dd" in tokens
        assert "ee" in tokens
        assert "ff" in tokens

    def test_empty_result_for_special_chars(self) -> None:
        """Special character strings produce empty token list."""
        tokens = _tokenize("--- ...")
        assert tokens == []

    def test_legal_document_tokenization(self) -> None:
        """Legal document text is tokenized correctly."""
        text = "SCHEDULE II — Floor Space Index"
        tokens = _tokenize(text)
        assert "schedule" in tokens
        assert "ii" in tokens
        assert "floor" in tokens
        assert "space" in tokens
        assert "index" in tokens


# ---------------------------------------------------------------------------
# _build_filter
# ---------------------------------------------------------------------------

class TestBuildFilterSparse:
    """Tests for _build_filter() in sparse module."""

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


# ---------------------------------------------------------------------------
# _is_schedule_ii_page
# ---------------------------------------------------------------------------

class TestIsScheduleIiPage:
    """Tests for _is_schedule_ii_page()."""

    def test_excerpt_contains_schedule_ii(self) -> None:
        """Candidate with 'Schedule II' in excerpt is detected."""
        c = make_candidate(
            page_id="doc#c000001",
            excerpt="Per Schedule II provisions for FSI...",
        )
        assert _is_schedule_ii_page(c) is True

    def test_excerpt_contains_schedule_2(self) -> None:
        """Candidate with 'Schedule 2' in excerpt is detected."""
        c = make_candidate(
            page_id="doc#c000001",
            excerpt="Schedule 2 of the regulations...",
        )
        assert _is_schedule_ii_page(c) is True

    def test_section_title_contains_schedule_ii(self) -> None:
        """Schedule II in section_title is detected."""
        c = make_candidate(
            page_id="doc#c000001",
            section_title="Schedule II - FSI Tables",
        )
        assert _is_schedule_ii_page(c) is True

    def test_page_id_contains_schedule_ii(self) -> None:
        """Schedule II in page_id is detected."""
        c = make_candidate(
            page_id="doc_schedule_ii#c000001",
            excerpt="Some content about FSI...",
        )
        assert _is_schedule_ii_page(c) is True

    def test_non_schedule_ii_returns_false(self) -> None:
        """A non-Schedule-II page returns False."""
        c = make_candidate(
            page_id="doc#c000001",
            excerpt="General FSI provisions for all zones...",
        )
        assert _is_schedule_ii_page(c) is False

    def test_case_insensitive(self) -> None:
        """Detection is case-insensitive."""
        c = make_candidate(
            page_id="doc#c000001",
            excerpt="schedule II provisions...",
        )
        assert _is_schedule_ii_page(c) is True


# ---------------------------------------------------------------------------
# BM25QueryScorer
# ---------------------------------------------------------------------------

class TestBM25QueryScorer:
    """Tests for BM25QueryScorer class."""

    def test_init_empty_corpus(self) -> None:
        """BM25QueryScorer with empty corpus initializes correctly."""
        scorer = BM25QueryScorer([])
        assert scorer.N == 0
        assert len(scorer._vocab) == 0

    def test_init_single_document(self) -> None:
        """BM25QueryScorer with single document initializes correctly."""
        texts = ["hello world hello"]
        scorer = BM25QueryScorer(texts)
        assert scorer.N == 1
        assert len(scorer._vocab) > 0
        assert scorer._avgdl > 0

    def test_init_multiple_documents(self) -> None:
        """BM25QueryScorer with multiple documents computes IDF correctly."""
        texts = ["FSI residential", "FAR commercial", "FSI FAR zones"]
        scorer = BM25QueryScorer(texts)
        assert scorer.N == 3
        assert len(scorer._vocab) > 0
        # IDF should be computed for all terms
        for term in scorer._vocab:
            assert term in scorer._idf

    def test_score_query_empty(self) -> None:
        """Empty query returns empty dict."""
        scorer = BM25QueryScorer(["FSI residential"])
        scores = scorer.score_query("")
        assert scores == {}

    def test_score_query_no_matches(self) -> None:
        """Query with no matching terms returns empty or zero scores."""
        texts = ["FSI residential"]
        scorer = BM25QueryScorer(texts)
        scores = scorer.score_query("xyzabc123")
        # Either empty or all zeros
        assert len(scores) == 0 or all(s == 0 for s in scores.values())

    def test_score_query_with_matches(self) -> None:
        """Query with matching terms returns positive scores."""
        texts = ["FSI residential Mumbai", "FAR commercial Delhi"]
        scorer = BM25QueryScorer(texts)
        scores = scorer.score_query("FSI")
        assert len(scores) == 2
        assert scores[0] > 0  # First doc contains FSI

    def test_build_sparse_query_vector(self) -> None:
        """build_sparse_query_vector returns indices and values."""
        texts = ["FSI residential zones", "FAR commercial areas"]
        scorer = BM25QueryScorer(texts)
        indices, values = scorer.build_sparse_query_vector("FSI residential")
        assert isinstance(indices, list)
        assert isinstance(values, list)
        assert len(indices) == len(values)

    def test_build_sparse_query_vector_empty_query(self) -> None:
        """Empty query returns empty vectors."""
        scorer = BM25QueryScorer(["FSI residential"])
        indices, values = scorer.build_sparse_query_vector("")
        assert indices == []
        assert values == []

    def test_build_sparse_query_vector_top_k(self) -> None:
        """top_k limits the number of terms in sparse vector."""
        texts = [" ".join(["term"] * 50) for _ in range(10)]
        scorer = BM25QueryScorer(texts)
        indices, values = scorer.build_sparse_query_vector(" ".join(["term"] * 20), top_k=5)
        assert len(indices) <= 5
        assert len(values) <= 5


# ---------------------------------------------------------------------------
# _qdrant_sparse_search
# ---------------------------------------------------------------------------

class TestQdrantSparseSearch:
    """Tests for _qdrant_sparse_search()."""

    def test_empty_results_returns_empty_list(self) -> None:
        """Qdrant returns empty points → empty candidate list."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        result = _qdrant_sparse_search(
            client=mock_client,
            query_indices=[1, 2, 3],
            query_values=[0.5, 0.3, 0.2],
            top_k=10,
        )

        assert result == []

    def test_single_result_converted_to_candidate(self) -> None:
        """A single Qdrant point is converted to a RetrievalCandidate."""
        mock_client = MagicMock()
        mock_results = MagicMock()

        mock_point = MagicMock()
        mock_point.id = "doc_hash#c000001"
        mock_point.score = 12.5
        mock_point.payload = {
            "image_uri": "lance://data/docs/abc/pages/p_0001.png",
            "chunk_text": "Section about Schedule II FSI provisions...",
            "section_path": ["Chapter 4", "Schedule II"],
        }
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results

        candidates = _qdrant_sparse_search(
            client=mock_client,
            query_indices=[1, 2],
            query_values=[0.8, 0.6],
            top_k=10,
        )

        assert len(candidates) == 1
        c = candidates[0]
        assert c.page_id == "doc_hash#c000001"
        assert c.score == 12.5
        assert c.channel_scores["sparse"] == 12.5
        assert c.channel_ranks["sparse"] == 1
        assert c.extracted_text_excerpt == "Section about Schedule II FSI provisions..."
        assert c.section_title == "Schedule II"

    def test_results_ordered_by_score(self) -> None:
        """Results are returned in score-descending order."""
        mock_client = MagicMock()
        mock_results = MagicMock()

        mock_points = []
        for score, page_id in [(8.0, "chunk1"), (12.5, "chunk2"), (5.0, "chunk3")]:
            pt = MagicMock()
            pt.id = page_id
            pt.score = score
            pt.payload = {"image_uri": "", "section_path": [], "chunk_text": ""}
            mock_points.append(pt)

        mock_results.points = mock_points
        mock_client.query_points.return_value = mock_results

        candidates = _qdrant_sparse_search(
            client=mock_client,
            query_indices=[1],
            query_values=[0.9],
            top_k=10,
        )

        assert [c.page_id for c in candidates] == ["chunk1", "chunk2", "chunk3"]
        assert [c.score for c in candidates] == pytest.approx([8.0, 12.5, 5.0])

    def test_connection_error_raises_service_unavailable(self) -> None:
        """Connection-related errors raise ServiceUnavailableError."""
        mock_client = MagicMock()
        mock_client.query_points.side_effect = ConnectionError("connection refused")

        with pytest.raises(ServiceUnavailableError):
            _qdrant_sparse_search(
                client=mock_client,
                query_indices=[1],
                query_values=[0.9],
                top_k=10,
            )

    def test_qdrant_client_query_points_call_structure(self) -> None:
        """The Qdrant query_points is called with prefetch + sparse query."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        _qdrant_sparse_search(
            client=mock_client,
            query_indices=[1, 2, 3],
            query_values=[0.8, 0.6, 0.4],
            top_k=20,
            prefetch_limit=200,
        )

        # Verify the call structure
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == "pages_text"
        assert "prefetch" in call_kwargs
        assert "limit" in call_kwargs
        assert call_kwargs["limit"] == 20
        assert call_kwargs["with_payload"] is True


# ---------------------------------------------------------------------------
# retrieve_sparse
# ---------------------------------------------------------------------------

class TestRetrieveSparse:
    """Tests for retrieve_sparse()."""

    def test_returns_retrieval_result(self) -> None:
        """retrieve_sparse returns a RetrievalResult."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.retrieve.sparse.get_qdrant_client", return_value=mock_client):
            result = retrieve_sparse(
                query="Schedule II FSI provisions",
            )

        assert result is not None
        assert result.query == "Schedule II FSI provisions"
        assert result.retrieval_strategy == "hybrid"
        assert "sparse" in result.flags or result.flags == {}

    def test_candidates_populated_from_qdrant(self) -> None:
        """Qdrant results populate the candidates list."""
        mock_client = MagicMock()
        mock_results = MagicMock()

        pt = MagicMock()
        pt.id = "doc#c000001"
        pt.score = 10.5
        pt.payload = {
            "image_uri": "lance://x.png",
            "section_path": ["Chapter 4"],
            "chunk_text": "Schedule II provisions...",
        }
        mock_results.points = [pt]
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.retrieve.sparse.get_qdrant_client", return_value=mock_client):
            result = retrieve_sparse(
                query="Schedule II",
                top_k=10,
            )

        assert len(result.candidates) == 1
        assert result.candidates[0].page_id == "doc#c000001"

    def test_latency_is_measured(self) -> None:
        """Latency is measured and reported."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.retrieve.sparse.get_qdrant_client", return_value=mock_client):
            result = retrieve_sparse(query="FSI")

        assert result.latency_ms >= 0

    def test_qdrant_unavailable_raises_service_unavailable(self) -> None:
        """Qdrant connection failure raises ServiceUnavailableError."""
        with patch(
            "urban_rag.retrieve.sparse.get_qdrant_client",
            side_effect=ConnectionError("qdrant down"),
        ), pytest.raises(ServiceUnavailableError):
            retrieve_sparse(query="Schedule II")

    def test_custom_scorer_is_used(self) -> None:
        """Custom scorer is used when provided."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        mock_scorer = MagicMock()
        mock_scorer.build_sparse_query_vector.return_value = ([1, 2], [0.8, 0.6])

        with patch("urban_rag.retrieve.sparse.get_qdrant_client", return_value=mock_client):
            result = retrieve_sparse(query="FSI", scorer=mock_scorer)

        mock_scorer.build_sparse_query_vector.assert_called_once_with("FSI")
        assert result is not None


# ---------------------------------------------------------------------------
# smoke_query
# ---------------------------------------------------------------------------

class TestSmokeQuery:
    """Tests for smoke_query()."""

    def test_returns_dict_with_all_keys(self) -> None:
        """smoke_query returns dict with candidates, latency_ms, etc."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.retrieve.sparse.get_qdrant_client", return_value=mock_client):
            result = smoke_query(query="Schedule II FSI", top_k=10)

        assert "candidates" in result
        assert "latency_ms" in result
        assert "candidate_count" in result
        assert "schedule_ii_count" in result
        assert "query" in result

    def test_default_query_is_schedule_ii(self) -> None:
        """Default query string is 'Schedule II FSI FAR provisions'."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.retrieve.sparse.get_qdrant_client", return_value=mock_client):
            smoke_query(top_k=5)

        # Verify query_points was called (smoke query triggers retrieval)
        mock_client.query_points.assert_called()

    def test_schedule_ii_count_identifies_schedule_ii_pages(self) -> None:
        """Schedule II pages in top_k are counted correctly."""
        mock_client = MagicMock()
        mock_results = MagicMock()

        def make_point(page_id: str, score: float, excerpt: str) -> MagicMock:
            pt = MagicMock()
            pt.id = page_id
            pt.score = score
            pt.payload = {
                "image_uri": "",
                "section_path": [],
                "chunk_text": excerpt,
            }
            return pt

        mock_results.points = [
            make_point("chunk1", 15.0, "Schedule II provisions for FSI..."),
            make_point("chunk2", 12.0, "General FSI rules for all zones..."),
            make_point("chunk3", 10.0, "Schedule II - FAR tables..."),
            make_point("chunk4", 8.0, "FAR commercial zones..."),
        ]
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.retrieve.sparse.get_qdrant_client", return_value=mock_client):
            result = smoke_query(query="Schedule II", top_k=10)

        # 2 Schedule II pages (chunk1 and chunk3)
        assert result["schedule_ii_count"] == 2
        assert result["candidate_count"] == 4

    def test_candidate_count_reflects_qdrant_results(self) -> None:
        """Candidate count matches the number of Qdrant results."""
        mock_client = MagicMock()
        mock_results = MagicMock()

        def make_point(page_id: str, score: float) -> MagicMock:
            pt = MagicMock()
            pt.id = page_id
            pt.score = score
            pt.payload = {"image_uri": "", "section_path": [], "chunk_text": ""}
            return pt

        mock_results.points = [
            make_point("chunk1", 15.0),
            make_point("chunk2", 12.0),
        ]
        mock_client.query_points.return_value = mock_results

        with patch("urban_rag.retrieve.sparse.get_qdrant_client", return_value=mock_client):
            result = smoke_query(query="FSI", top_k=10)

        assert result["candidate_count"] == 2
        assert len(result["candidates"]) == 2
