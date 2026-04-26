"""Unit tests for src/urban_rag/retrieve/fusion.py — RRF fusion."""

from __future__ import annotations

from urban_rag.common.types import RetrievalCandidate, RetrievalResult
from urban_rag.retrieve.fusion import RRF_K, fuse_candidates, fuse_results, smoke_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candidate(
    page_id: str,
    score: float,
    channel: str,
    rank: int,
) -> RetrievalCandidate:
    """Make a RetrievalCandidate with channel_scores and channel_ranks."""
    return RetrievalCandidate(
        page_id=page_id,
        score=score,
        channel_scores={channel: score},
        channel_ranks={channel: rank},
        page_image_uri=f"s3://pages/{page_id}.png",
        extracted_text_excerpt=f"Excerpt for {page_id}",
    )


# ---------------------------------------------------------------------------
# Tests for fuse_candidates
# ---------------------------------------------------------------------------

class TestFuseCandidates:
    """Tests for the core RRF fusion function."""

    def test_empty_inputs_returns_empty(self):
        """No input channels → empty output."""
        result = fuse_candidates(
            visual_candidates=None,
            text_candidates=None,
            sparse_candidates=None,
        )
        assert result == []

    def test_single_channel_all_candidates_preserved(self):
        """With one channel, all its candidates appear in output."""
        visual = [
            make_candidate("p1", 10.0, "visual", 1),
            make_candidate("p2", 8.0, "visual", 2),
            make_candidate("p3", 6.0, "visual", 3),
        ]
        result = fuse_candidates(visual_candidates=visual, top_n=5)
        assert len(result) == 3
        page_ids = [c.page_id for c in result]
        assert page_ids == ["p1", "p2", "p3"]

    def test_two_channels_both_contribute(self):
        """Candidates from both channels appear, deduplicated by page_id."""
        visual = [
            make_candidate("p1", 10.0, "visual", 1),
            make_candidate("p2", 8.0, "visual", 2),
        ]
        text = [
            make_candidate("p1", 0.9, "text", 1),  # p1 also in text
            make_candidate("p3", 0.8, "text", 2),
        ]
        result = fuse_candidates(
            visual_candidates=visual,
            text_candidates=text,
            top_n=5,
        )

        page_ids = [c.page_id for c in result]
        assert "p1" in page_ids
        assert "p2" in page_ids
        assert "p3" in page_ids

        # p1 has contributions from both channels
        p1 = next(c for c in result if c.page_id == "p1")
        assert "visual" in p1.channel_scores
        assert "text" in p1.channel_scores
        assert p1.channel_ranks["visual"] == 1
        assert p1.channel_ranks["text"] == 1

    def test_rrf_score_computed_correctly(self):
        """RRF = sum of 1/(k+rank) per channel."""
        # p1 appears at visual rank 1 and text rank 1 (both positions 0 → rank 1)
        visual = [make_candidate("p1", 10.0, "visual", 1)]
        text = [make_candidate("p1", 0.9, "text", 1)]

        result = fuse_candidates(
            visual_candidates=visual,
            text_candidates=text,
        )
        p1 = result[0]
        expected = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 1)
        assert abs(p1.score - expected) < 1e-9

    def test_rrf_k_constant_is_60(self):
        """Verify the RRF constant k=60 is used."""
        assert RRF_K == 60

    def test_top_n_limits_output(self):
        """top_n parameter caps the number of returned candidates."""
        visual = [make_candidate(f"p{i}", 10.0 - i, "visual", i + 1) for i in range(10)]
        result = fuse_candidates(visual_candidates=visual, top_n=3)
        assert len(result) == 3

    def test_channel_scores_and_ranks_populated(self):
        """Each candidate has channel_scores and channel_ranks dicts."""
        visual = [make_candidate("p1", 10.0, "visual", 1)]
        sparse = [make_candidate("p1", 4.0, "sparse", 1)]  # p1 at SPARSE rank 1 (position 0)

        result = fuse_candidates(
            visual_candidates=visual,
            sparse_candidates=sparse,
        )
        p1 = result[0]
        assert "visual" in p1.channel_scores
        assert "sparse" in p1.channel_scores
        assert p1.channel_scores["visual"] == 10.0
        assert p1.channel_scores["sparse"] == 4.0
        assert p1.channel_ranks["visual"] == 1
        assert p1.channel_ranks["sparse"] == 1

    def test_none_channel_skipped(self):
        """None channel is treated as empty (graceful degradation)."""
        visual = [make_candidate("p1", 10.0, "visual", 1)]
        result = fuse_candidates(
            visual_candidates=visual,
            text_candidates=None,
            sparse_candidates=None,
        )
        assert len(result) == 1
        assert result[0].page_id == "p1"

    def test_candidates_sorted_by_rrf_score_descending(self):
        """Fused candidates are sorted by RRF score, highest first."""
        # p1 appears in both channels; p2 only in visual
        # p1 should be ranked above p2
        visual = [
            make_candidate("p1", 10.0, "visual", 1),
            make_candidate("p2", 9.0, "visual", 2),
        ]
        text = [make_candidate("p1", 0.9, "text", 2)]

        result = fuse_candidates(
            visual_candidates=visual,
            text_candidates=text,
        )
        # p1: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0159 = 0.0323
        # p2: 1/(60+2) = 0.0161
        assert result[0].page_id == "p1"
        assert result[1].page_id == "p2"

    def test_all_three_channels_contribute(self):
        """With all three channels, candidates can come from any channel."""
        visual = [make_candidate("p_visual", 10.0, "visual", 1)]
        text = [make_candidate("p_text", 10.0, "text", 1)]
        sparse = [make_candidate("p_sparse", 10.0, "sparse", 1)]

        result = fuse_candidates(
            visual_candidates=visual,
            text_candidates=text,
            sparse_candidates=sparse,
        )
        page_ids = {c.page_id for c in result}
        assert "p_visual" in page_ids
        assert "p_text" in page_ids
        assert "p_sparse" in page_ids

    def test_same_page_in_all_channels_highest_rank(self):
        """A page appearing in all three channels ranks above single-channel."""
        visual = [make_candidate("p_all", 10.0, "visual", 1)]
        text = [make_candidate("p_all", 0.9, "text", 1)]
        sparse = [make_candidate("p_all", 4.0, "sparse", 1)]
        single = [make_candidate("p_one", 9.0, "visual", 2)]

        result = fuse_candidates(
            visual_candidates=visual + single,
            text_candidates=text,
            sparse_candidates=sparse,
            top_n=5,
        )
        page_ids = [c.page_id for c in result]
        assert page_ids[0] == "p_all"
        assert "p_one" in page_ids


# ---------------------------------------------------------------------------
# Tests for fuse_results
# ---------------------------------------------------------------------------

class TestFuseResults:
    """Tests for the RetrievalResult-level fusion."""

    def test_fuse_results_builds_retrieval_result(self):
        """fuse_results returns a RetrievalResult with all fields."""
        from urban_rag.common.types import RetrievalResult

        # Create individual results
        visual = RetrievalResult(
            query="test query",
            candidates=[make_candidate("p1", 10.0, "visual", 1)],
            latency_ms=5,
            flags={},
            retrieval_strategy="visual_primary",
        )

        fused = fuse_results(visual_result=visual)
        assert isinstance(fused, RetrievalResult)
        assert fused.query == "test query"
        assert len(fused.candidates) == 1

    def test_flags_from_input_results_preserved(self):
        """Flags from input results are carried into fused result."""
        visual = RetrievalResult(
            query="test",
            candidates=[make_candidate("p1", 10.0, "visual", 1)],
            latency_ms=5,
            flags={"degraded_mode": True},
            retrieval_strategy="visual_primary",
        )
        fused = fuse_results(visual_result=visual)
        assert fused.flags.get("degraded_mode") is True

    def test_none_input_channel_skipped(self):
        """None input results are skipped gracefully."""
        from urban_rag.common.types import RetrievalResult

        text_result = RetrievalResult(
            query="test",
            candidates=[make_candidate("p2", 0.8, "text", 1)],
            latency_ms=3,
            flags={},
            retrieval_strategy="text_primary",
        )

        fused = fuse_results(text_result=text_result, sparse_result=None)
        assert len(fused.candidates) == 1
        assert fused.candidates[0].page_id == "p2"

    def test_latency_ms_records_fusion_time(self):
        """fuse_results latency_ms reflects time spent in fusion."""
        visual = RetrievalResult(
            query="test",
            candidates=[make_candidate("p1", 10.0, "visual", i) for i in range(1, 6)],
            latency_ms=5,
            flags={},
            retrieval_strategy="visual_primary",
        )
        fused = fuse_results(visual_result=visual)
        assert fused.latency_ms >= 0

    def test_retrieval_strategy_hybrid_when_all_channels(self):
        """Strategy is 'hybrid' when all three channels contribute."""
        from urban_rag.common.types import RetrievalResult

        visual = RetrievalResult(
            query="test",
            candidates=[make_candidate("p1", 10.0, "visual", 1)],
            retrieval_strategy="visual_primary",
        )
        text = RetrievalResult(
            query="test",
            candidates=[make_candidate("p2", 0.8, "text", 1)],
            retrieval_strategy="text_primary",
        )
        sparse = RetrievalResult(
            query="test",
            candidates=[make_candidate("p3", 4.0, "sparse", 1)],
            retrieval_strategy="hybrid",
        )
        fused = fuse_results(
            visual_result=visual,
            text_result=text,
            sparse_result=sparse,
        )
        assert fused.retrieval_strategy == "hybrid"


# ---------------------------------------------------------------------------
# Tests for smoke_test
# ---------------------------------------------------------------------------

class TestSmokeTest:
    """Tests for the smoke_test utility function."""

    def test_smoke_test_runs_without_error(self):
        """smoke_test() executes without raising."""
        result = smoke_test()
        assert result["passed"] is True
        assert result["rrf_k"] == 60

    def test_smoke_test_returns_expected_keys(self):
        """smoke_test returns dict with fused_count, doc scores, rrf_k."""
        result = smoke_test()
        assert "fused_count" in result
        assert "doc1_p001_score" in result
        assert "doc1_p002_score" in result
        assert result["rrf_k"] == 60


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for RRF fusion."""

    def test_duplicate_page_id_same_channel_takes_best_rank(self):
        """If the same page appears twice in one channel, keep better rank."""
        # p1 appears at rank 3 in text channel with higher score
        # and at rank 1 in text channel with lower score
        # We should keep rank 1 (the better rank)
        text = [
            make_candidate("p1", 0.5, "text", 1),  # rank 1, score 0.5
            make_candidate("p1", 0.9, "text", 3),  # rank 3, score 0.9
        ]
        result = fuse_candidates(text_candidates=text)
        p1 = result[0]
        assert p1.channel_ranks["text"] == 1
        assert p1.channel_scores["text"] == 0.5  # score from better rank

    def test_empty_candidate_lists(self):
        """Empty candidate list is treated same as None."""
        result = fuse_candidates(
            visual_candidates=[],
            text_candidates=None,
            sparse_candidates=[],
        )
        assert result == []
