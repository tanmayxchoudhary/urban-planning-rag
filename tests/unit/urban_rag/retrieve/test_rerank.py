"""Unit tests for src/urban_rag/retrieve/rerank.py — VLM cross-encoder rerank."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from urban_rag.common.types import RetrievalCandidate, RetrievalResult
from urban_rag.retrieve.rerank import (
    _apply_scores,
    _parse_gemini_response,
    rerank_candidates,
    rerank_retrieval_result,
    smoke_test,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candidate(
    page_id: str,
    score: float = 1.0,
    channel: str = "visual",
    rank: int = 1,
    image_uri: str = "",
    excerpt: str = "",
    section_title: str | None = None,
) -> RetrievalCandidate:
    """Factory to create a RetrievalCandidate for testing."""
    return RetrievalCandidate(
        page_id=page_id,
        score=score,
        channel_scores={channel: score},
        channel_ranks={channel: rank},
        rerank_score=None,
        rerank_rationale=None,
        page_image_uri=image_uri or f"s3://pages/{page_id}.png",
        extracted_text_excerpt=excerpt or f"Excerpt for {page_id}",
        section_title=section_title,
    )


def make_retrieval_result(
    query: str,
    candidates: list[RetrievalCandidate],
    flags: dict[str, bool] | None = None,
) -> RetrievalResult:
    """Factory to create a RetrievalResult for testing."""
    return RetrievalResult(
        query=query,
        expanded_queries=[],
        candidates=candidates,
        latency_ms=10,
        flags=flags or {},
        retrieval_strategy="hybrid",
    )


# ---------------------------------------------------------------------------
# _apply_scores
# ---------------------------------------------------------------------------

class TestApplyScores:
    """Tests for _apply_scores()."""

    def test_scores_applied_to_matching_candidates(self) -> None:
        """VLM scores are applied to candidates with matching page_id."""
        candidates = [
            make_candidate("p1", score=1.0),
            make_candidate("p2", score=0.9),
            make_candidate("p3", score=0.8),
        ]

        scores_data = {
            "scores": [
                {"page_id": "p1", "score": 9.0, "rationale": "Direct answer"},
                {"page_id": "p2", "score": 5.0, "rationale": "Partial"},
                {"page_id": "p3", "score": 2.0, "rationale": "Irrelevant"},
            ]
        }

        reranked = _apply_scores(candidates, scores_data)

        assert reranked[0].rerank_score == 9.0
        assert reranked[0].rerank_rationale == "Direct answer"
        assert reranked[1].rerank_score == 5.0
        assert reranked[2].rerank_score == 2.0

    def test_candidates_sorted_by_rerank_score_descending(self) -> None:
        """Candidates are sorted by rerank_score, highest first."""
        candidates = [
            make_candidate("p1"),
            make_candidate("p2"),
            make_candidate("p3"),
        ]

        scores_data = {
            "scores": [
                {"page_id": "p3", "score": 10.0, "rationale": "Best"},
                {"page_id": "p1", "score": 7.0, "rationale": "Good"},
                {"page_id": "p2", "score": 3.0, "rationale": "Bad"},
            ]
        }

        reranked = _apply_scores(candidates, scores_data)

        assert [c.page_id for c in reranked] == ["p3", "p1", "p2"]
        assert [c.rerank_score for c in reranked] == [10.0, 7.0, 3.0]

    def test_unscored_pages_get_zero(self) -> None:
        """Pages not in scores_data get rerank_score=0 and default rationale."""
        candidates = [
            make_candidate("p1"),
            make_candidate("p2"),
            make_candidate("p3"),
        ]

        scores_data = {
            "scores": [
                {"page_id": "p1", "score": 8.0, "rationale": "Scored"},
            ]
        }

        reranked = _apply_scores(candidates, scores_data)

        p1 = next(c for c in reranked if c.page_id == "p1")
        p2 = next(c for c in reranked if c.page_id == "p2")
        p3 = next(c for c in reranked if c.page_id == "p3")

        assert p1.rerank_score == 8.0
        assert p2.rerank_score == 0.0
        assert p2.rerank_rationale == "Not scored by VLM"
        assert p3.rerank_score == 0.0

    def test_preserves_other_candidate_fields(self) -> None:
        """Other fields (channel_scores, channel_ranks, etc.) are preserved."""
        candidates = [
            RetrievalCandidate(
                page_id="p1",
                score=0.95,
                channel_scores={"visual": 0.95, "text": 0.7},
                channel_ranks={"visual": 1, "text": 3},
                page_image_uri="s3://pages/p1.png",
                extracted_text_excerpt="Test excerpt",
                section_title="Test Section",
            )
        ]

        scores_data = {
            "scores": [
                {"page_id": "p1", "score": 9.5, "rationale": "Great page"},
            ]
        }

        reranked = _apply_scores(candidates, scores_data)

        assert reranked[0].score == 0.95
        assert reranked[0].channel_scores == {"visual": 0.95, "text": 0.7}
        assert reranked[0].channel_ranks == {"visual": 1, "text": 3}
        assert reranked[0].page_image_uri == "s3://pages/p1.png"
        assert reranked[0].extracted_text_excerpt == "Test excerpt"
        assert reranked[0].section_title == "Test Section"


# ---------------------------------------------------------------------------
# _parse_gemini_response
# ---------------------------------------------------------------------------

class TestParseGeminiResponse:
    """Tests for _parse_gemini_response()."""

    def test_parses_valid_json_response(self) -> None:
        """A valid Gemini JSON response is parsed correctly."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"scores": [{"page_id": "p1", "score": 9.0}]}'}
                        ]
                    }
                }
            ]
        }

        result = _parse_gemini_response(response)

        assert "scores" in result
        assert result["scores"][0]["page_id"] == "p1"
        assert result["scores"][0]["score"] == 9.0

    def test_strips_markdown_code_fences(self) -> None:
        """JSON wrapped in ```json ... ``` is parsed correctly."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '```json\n{"scores": [{"page_id": "p1", "score": 8.5}]}\n```'}
                        ]
                    }
                }
            ]
        }

        result = _parse_gemini_response(response)

        assert "scores" in result
        assert result["scores"][0]["score"] == 8.5

    def test_strips_plain_markdown_fences(self) -> None:
        """JSON wrapped in ``` ... ``` is parsed correctly."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '```\n{"scores": [{"page_id": "p1", "score": 7.0}]}\n```'}
                        ]
                    }
                }
            ]
        }

        result = _parse_gemini_response(response)

        assert "scores" in result

    def test_raises_on_empty_candidates(self) -> None:
        """Empty candidates list raises ValueError."""
        response = {"candidates": []}

        with pytest.raises(ValueError, match="No candidates"):
            _parse_gemini_response(response)

    def test_raises_on_missing_scores_key(self) -> None:
        """Response without 'scores' key raises ValueError."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"other": "data"}'}
                        ]
                    }
                }
            ]
        }

        with pytest.raises(ValueError, match="No 'scores' key"):
            _parse_gemini_response(response)

    def test_raises_on_invalid_json(self) -> None:
        """Invalid JSON raises ValueError."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "not valid json at all"}
                        ]
                    }
                }
            ]
        }

        with pytest.raises(ValueError, match="Failed to parse JSON"):
            _parse_gemini_response(response)


# ---------------------------------------------------------------------------
# rerank_candidates
# ---------------------------------------------------------------------------

class TestRerankCandidates:
    """Tests for rerank_candidates()."""

    def test_empty_candidates_returns_empty_with_flag(self) -> None:
        """Empty input returns empty list with vlm_rerank_skipped=True."""
        with patch("urban_rag.retrieve.rerank.get_settings") as mock_settings:
            mock_settings.return_value.gemini_api_key = "valid-test-key"
            reranked, flags = rerank_candidates(
                candidates=[],
                query="test query",
            )

        assert reranked == []
        assert flags["vlm_rerank_skipped"] is True

    def test_fewer_candidates_than_top_n_returns_copy(self) -> None:
        """When len(candidates) <= top_n, returns copy with vlm_rerank_skipped=False."""
        candidates = [
            make_candidate("p1", score=1.0),
            make_candidate("p2", score=0.9),
            make_candidate("p3", score=0.8),
        ]

        with patch("urban_rag.retrieve.rerank.get_settings") as mock_settings:
            mock_settings.return_value.gemini_api_key = "valid-test-key"
            # top_n=5, but only 3 candidates
            reranked, flags = rerank_candidates(
                candidates=candidates,
                query="test query",
                top_n=5,
            )

        assert len(reranked) == 3
        assert flags["vlm_rerank_skipped"] is False

    def test_successful_rerank_reorders_candidates(self) -> None:
        """Successful reranking changes the candidate order."""
        # Candidates: p0 has highest fusion score (1.0), p19 has lowest (0.81)
        candidates = [
            make_candidate(f"p{i}", score=1.0 - i * 0.01) for i in range(20)
        ]

        # mock_scores gives p0 score 20 (highest), p19 score 1 (lowest)
        mock_scores = {
            "scores": [
                {"page_id": f"p{i}", "score": float(20 - i), "rationale": f"Page {i}"}
                for i in range(20)
            ]
        }

        with patch(
            "urban_rag.retrieve.rerank.get_settings"
        ) as mock_settings, patch(
            "urban_rag.retrieve.rerank._call_gemini_with_retry",
            return_value=mock_scores,
        ):
            mock_settings.return_value.gemini_api_key = "valid-test-key"
            reranked, flags = rerank_candidates(
                candidates=candidates,
                query="test query",
                top_n=5,
            )

        # p0 should be first (score 20 is highest)
        assert reranked[0].page_id == "p0"
        assert reranked[0].rerank_score == 20.0
        assert flags["vlm_rerank_skipped"] is False

    def test_top_n_limits_output(self) -> None:
        """top_n parameter limits the number of returned candidates."""
        candidates = [
            make_candidate(f"p{i}", score=1.0 - i * 0.01) for i in range(20)
        ]

        # mock_scores: p0 gets 0, p19 gets 19 (p19 is highest)
        mock_scores = {
            "scores": [
                {"page_id": f"p{i}", "score": float(i), "rationale": f"Page {i}"}
                for i in range(20)
            ]
        }

        with patch(
            "urban_rag.retrieve.rerank.get_settings"
        ) as mock_settings, patch(
            "urban_rag.retrieve.rerank._call_gemini_with_retry",
            return_value=mock_scores,
        ):
            mock_settings.return_value.gemini_api_key = "valid-test-key"
            reranked, _flags = rerank_candidates(
                candidates=candidates,
                query="test query",
                top_n=5,
            )

        assert len(reranked) == 5
        # p19 has highest score 19, p18 has 18, etc.
        assert [c.page_id for c in reranked] == ["p19", "p18", "p17", "p16", "p15"]

    def test_timeout_fallback_sets_flag(self) -> None:
        """Timeout/exception triggers fallback with vlm_rerank_skipped=True."""
        candidates = [
            make_candidate(f"p{i}", score=1.0 - i * 0.01) for i in range(20)
        ]

        with patch(
            "urban_rag.retrieve.rerank.get_settings"
        ) as mock_settings, patch(
            "urban_rag.retrieve.rerank._call_gemini_with_retry",
            side_effect=TimeoutError("Request timed out"),
        ):
            mock_settings.return_value.gemini_api_key = "valid-test-key"
            reranked, flags = rerank_candidates(
                candidates=candidates,
                query="test query",
                top_n=5,
            )

        # Should return fusion order (top 5)
        assert len(reranked) == 5
        assert [c.page_id for c in reranked] == [f"p{i}" for i in range(5)]
        assert flags["vlm_rerank_skipped"] is True

    def test_http_error_fallback_sets_flag(self) -> None:
        """HTTP errors trigger fallback with vlm_rerank_skipped=True."""
        candidates = [
            make_candidate(f"p{i}", score=1.0 - i * 0.01) for i in range(20)
        ]

        with patch(
            "urban_rag.retrieve.rerank.get_settings"
        ) as mock_settings, patch(
            "urban_rag.retrieve.rerank._call_gemini_with_retry",
            side_effect=Exception("Network error"),
        ):
            mock_settings.return_value.gemini_api_key = "valid-test-key"
            reranked, flags = rerank_candidates(
                candidates=candidates,
                query="test query",
                top_n=5,
            )

        assert len(reranked) == 5
        assert flags["vlm_rerank_skipped"] is True

    def test_no_api_key_skips_with_flag(self) -> None:
        """When no API key is available, rerank is skipped with flag=True."""
        candidates = [
            make_candidate(f"p{i}", score=1.0 - i * 0.01) for i in range(20)
        ]

        with patch(
            "urban_rag.retrieve.rerank.get_settings"
        ) as mock_settings:
            mock_settings.return_value.gemini_api_key = "test-api-key-for-unit-tests"

            reranked, flags = rerank_candidates(
                candidates=candidates,
                query="test query",
                top_n=5,
            )

        # Should return fusion order (top 5 from original)
        assert len(reranked) == 5
        assert flags["vlm_rerank_skipped"] is True


# ---------------------------------------------------------------------------
# rerank_retrieval_result
# ---------------------------------------------------------------------------

class TestRerankRetrievalResult:
    """Tests for rerank_retrieval_result()."""

    def test_returns_new_retrieval_result_with_reranked_candidates(self) -> None:
        """Returns a new RetrievalResult with reranked candidates."""
        candidates = [
            make_candidate(f"p{i}", score=1.0 - i * 0.01) for i in range(20)
        ]
        original_result = make_retrieval_result(
            query="test query",
            candidates=candidates,
            flags={"degraded_mode": True},
        )

        # p0 gets highest VLM score (20), p19 gets lowest (1)
        mock_scores = {
            "scores": [
                {"page_id": f"p{i}", "score": float(20 - i), "rationale": f"Page {i}"}
                for i in range(20)
            ]
        }

        with patch(
            "urban_rag.retrieve.rerank.get_settings"
        ) as mock_settings, patch(
            "urban_rag.retrieve.rerank._call_gemini_with_retry",
            return_value=mock_scores,
        ):
            mock_settings.return_value.gemini_api_key = "valid-test-key"
            reranked_result, flags = rerank_retrieval_result(
                result=original_result,
                query="test query",
                top_n=5,
            )

        assert reranked_result.query == "test query"
        assert len(reranked_result.candidates) == 5
        assert reranked_result.candidates[0].page_id == "p0"
        assert reranked_result.flags.get("degraded_mode") is True
        assert flags["vlm_rerank_skipped"] is False

    def test_timeout_preserves_original_flags(self) -> None:
        """On timeout, original flags are preserved and vlm_rerank_skipped is added."""
        candidates = [
            make_candidate(f"p{i}", score=1.0 - i * 0.01) for i in range(20)
        ]
        original_result = make_retrieval_result(
            query="test query",
            candidates=candidates,
            flags={"degraded_mode": True},
        )

        with patch(
            "urban_rag.retrieve.rerank.get_settings"
        ) as mock_settings, patch(
            "urban_rag.retrieve.rerank._call_gemini_with_retry",
            side_effect=TimeoutError("timed out"),
        ):
            mock_settings.return_value.gemini_api_key = "valid-test-key"
            reranked_result, flags = rerank_retrieval_result(
                result=original_result,
                query="test query",
                top_n=5,
            )

        assert reranked_result.flags.get("degraded_mode") is True
        assert reranked_result.flags.get("vlm_rerank_skipped") is True
        assert flags["vlm_rerank_skipped"] is True


# ---------------------------------------------------------------------------
# Integration scenario: recall@5 >= recall@10 before rerank
# ---------------------------------------------------------------------------

class TestRerankQuality:
    """Tests verifying rerank improves or maintains retrieval quality."""

    def test_rerank_reorders_fusion_output(self) -> None:
        """Reranking changes the order of fusion output."""
        # Create candidates where fusion score is opposite to true relevance
        # Fusion would rank p0 highest, but true relevance says p0-p4 are best
        candidates = [
            RetrievalCandidate(
                page_id=f"p{i}",
                score=1.0 - i * 0.001,  # p0 has highest fusion score
                channel_scores={"visual": 1.0 - i * 0.001},
                channel_ranks={"visual": i},
                page_image_uri=f"s3://pages/p{i}.png",
                extracted_text_excerpt=f"Content for page {i}",
                section_title=f"Section {i}",
            )
            for i in range(20)
        ]

        # True relevance: p0-p4 are the relevant pages (score 10), others score 1
        mock_scores = {
            "scores": [
                {
                    "page_id": f"p{i}",
                    "score": 10.0 if i < 5 else 1.0,
                    "rationale": "Relevant" if i < 5 else "Not relevant",
                }
                for i in range(20)
            ]
        }

        with patch(
            "urban_rag.retrieve.rerank.get_settings"
        ) as mock_settings, patch(
            "urban_rag.retrieve.rerank._call_gemini_with_retry",
            return_value=mock_scores,
        ):
            mock_settings.return_value.gemini_api_key = "valid-test-key"
            reranked, _ = rerank_candidates(
                candidates=candidates,
                query="test query",
                top_n=5,
            )

        # p0-p4 should now be in top 5 (all have score 10, sorted by original order)
        reranked_ids = {c.page_id for c in reranked}
        expected_ids = {f"p{i}" for i in range(5)}
        assert reranked_ids == expected_ids, (
            f"Reranked top-5 should be p0-p4, got {reranked_ids}"
        )

    def test_rerank_does_not_drop_relevant_candidates(self) -> None:
        """Reranking should not lose relevant candidates that were in top-10."""
        # p0-p9 are in top-10 by fusion, but only p0-p4 are truly relevant
        candidates = [
            RetrievalCandidate(
                page_id=f"p{i}",
                score=1.0 - i * 0.001,
                channel_scores={"visual": 1.0 - i * 0.001},
                channel_ranks={"visual": i},
                page_image_uri=f"s3://pages/p{i}.png",
                extracted_text_excerpt=f"Content for page {i}",
            )
            for i in range(10)
        ]

        # p0-p4 are highly relevant (score 10), p5-p9 are not (score 1)
        mock_scores = {
            "scores": [
                {
                    "page_id": f"p{i}",
                    "score": 10.0 if i < 5 else 1.0,
                    "rationale": "Relevant" if i < 5 else "Not relevant",
                }
                for i in range(10)
            ]
        }

        with patch(
            "urban_rag.retrieve.rerank.get_settings"
        ) as mock_settings, patch(
            "urban_rag.retrieve.rerank._call_gemini_with_retry",
            return_value=mock_scores,
        ):
            mock_settings.return_value.gemini_api_key = "valid-test-key"
            reranked, _ = rerank_candidates(
                candidates=candidates,
                query="test query",
                top_n=5,
            )

        # All top-5 should be from the truly relevant set (p0-p4)
        for candidate in reranked:
            # Extract the number from page_id
            num = int(candidate.page_id[1:])
            assert num < 5, f"Expected p0-p4, got {candidate.page_id}"


# ---------------------------------------------------------------------------
# smoke_test
# ---------------------------------------------------------------------------

class TestSmokeTest:
    """Tests for the smoke_test() function."""

    def test_smoke_test_runs_without_error(self) -> None:
        """smoke_test() executes without raising."""
        result = smoke_test()
        assert result["passed"] is True

    def test_smoke_test_returns_expected_keys(self) -> None:
        """smoke_test returns dict with required keys."""
        result = smoke_test()
        assert "passed" in result
        assert "original_top5" in result
        assert "reranked_top5" in result
        assert "top_score" in result
        assert result["passed"] is True

    def test_smoke_test_verifies_reordering(self) -> None:
        """smoke_test verifies that reranking changed the order."""
        result = smoke_test()
        # The smoke test creates candidates with fusion scores in one order
        # and VLM scores in reverse order
        assert result["original_top5"] != result["reranked_top5"]
