"""Unit tests for retrieval metrics (recall@k, MRR, NDCG, coverage)."""

from __future__ import annotations

import pytest

from urban_rag.common.types import RetrievalCandidate
from urban_rag.eval.metrics.retrieval import (
    RetrievalMetricsResult,
    compute_retrieval_metrics,
    coverage_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_candidates(page_ids: list[str]) -> list[RetrievalCandidate]:
    """Build ordered candidate list from page_id strings."""
    return [
        RetrievalCandidate(
            page_id=pid,
            score=1.0 / (i + 1),
            page_image_uri=f"s3://pages/{pid}.png",
        )
        for i, pid in enumerate(page_ids)
    ]


# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_full_recall_top_k_contains_all(self):
        """When all expected pages are in top-5, recall@5 == 1.0."""
        candidates = make_candidates(["p1", "p2", "p3", "p4", "p5"])
        expected = {"p1", "p2", "p3", "p4", "p5"}
        assert recall_at_k(candidates, expected, k=5) == 1.0

    def test_partial_recall(self):
        """2 of 4 expected pages in top-5 → recall@5 = 0.5."""
        candidates = make_candidates(["p1", "p2", "p99", "p98", "p97"])
        expected = {"p1", "p2", "p3", "p4"}
        assert recall_at_k(candidates, expected, k=5) == 0.5

    def test_no_recall(self):
        """No expected page in top-k → recall = 0.0."""
        candidates = make_candidates(["p99", "p98", "p97", "p96"])
        expected = {"p1", "p2", "p3"}
        assert recall_at_k(candidates, expected, k=5) == 0.0

    def test_empty_expected(self):
        """Empty ground truth → recall = 0.0 (no false positives)."""
        candidates = make_candidates(["p1", "p2"])
        assert recall_at_k(candidates, expected_pages=set(), k=5) == 0.0

    def test_k_smaller_than_expected(self):
        """k=2, expected=4 pages, 2 retrieved → recall = 0.5."""
        candidates = make_candidates(["p1", "p2", "p3", "p4"])
        expected = {"p1", "p2", "p3", "p4"}
        assert recall_at_k(candidates, expected, k=2) == 0.5

    def test_recall_at_1(self):
        """First result is relevant → recall@1 = 1.0."""
        candidates = make_candidates(["p1", "p2", "p3"])
        expected = {"p1"}
        assert recall_at_k(candidates, expected, k=1) == 1.0

    def test_recall_at_1_no_match(self):
        """First result is not relevant → recall@1 = 0.0."""
        candidates = make_candidates(["p99", "p1", "p2"])
        expected = {"p1"}
        assert recall_at_k(candidates, expected, k=1) == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


class TestMeanReciprocalRank:
    def test_mrr_first_relevant(self):
        """First candidate is relevant → MRR = 1.0."""
        candidates = make_candidates(["p1", "p2", "p3"])
        expected = {"p1"}
        assert mean_reciprocal_rank(candidates, expected, k=10) == 1.0

    def test_mrr_second_relevant(self):
        """Second candidate is first relevant → MRR = 0.5."""
        candidates = make_candidates(["p99", "p1", "p2"])
        expected = {"p1"}
        assert mean_reciprocal_rank(candidates, expected, k=10) == 0.5

    def test_mrr_third_relevant(self):
        """Third candidate is first relevant → MRR = 1/3."""
        candidates = make_candidates(["p99", "p98", "p1", "p2"])
        expected = {"p1"}
        assert mean_reciprocal_rank(candidates, expected, k=10) == pytest.approx(1 / 3)

    def test_mrr_no_relevant_in_top_k(self):
        """No relevant page in top-k → MRR = 0.0."""
        candidates = make_candidates(["p99", "p98", "p97"])
        expected = {"p1"}
        assert mean_reciprocal_rank(candidates, expected, k=10) == 0.0

    def test_mrr_k_less_than_relevant_position(self):
        """First relevant at rank 5 but k=3 → MRR = 0.0."""
        # p1 is at rank 5, outside top-3 window
        candidates = make_candidates(["p99", "p98", "p97", "p96", "p1"])
        expected = {"p1"}
        assert mean_reciprocal_rank(candidates, expected, k=3) == 0.0

    def test_mrr_empty_expected(self):
        """Empty ground truth → MRR = 0.0."""
        candidates = make_candidates(["p1", "p2"])
        assert mean_reciprocal_rank(candidates, expected_pages=set(), k=10) == 0.0

    def test_mrr_multiple_relevant(self):
        """Only first relevant rank matters for MRR; p3 is first relevant at rank 3."""
        # p3 at rank 3 is first relevant (p1, p2, p99 are not in expected)
        candidates = make_candidates(["p1", "p2", "p3", "p4"])
        expected = {"p3"}
        assert mean_reciprocal_rank(candidates, expected, k=10) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# NDCG@k
# ---------------------------------------------------------------------------


class TestNDCGAtK:
    def test_ndcg_perfect(self):
        """All top-3 are relevant → NDCG = 1.0."""
        candidates = make_candidates(["p1", "p2", "p3"])
        expected = {"p1", "p2", "p3"}
        assert ndcg_at_k(candidates, expected, k=3) == 1.0

    def test_ndcg_half_relevant_top_half(self):
        """2 of 4 retrieved at top-2 → NDCG below 1.0."""
        # Top-2: p1 (rel), p2 (rel), ideal top-2: p1, p2 → identical
        candidates = make_candidates(["p1", "p2", "p3", "p4"])
        expected = {"p1", "p2", "p3", "p4"}
        assert ndcg_at_k(candidates, expected, k=2) == 1.0

    def test_ndcg_relevant_at_end(self):
        """Only last retrieved is relevant → NDCG < 1.0."""
        candidates = make_candidates(["p99", "p98", "p1"])
        expected = {"p1"}
        # DCG = 1/log2(4) = 1/2 ≈ 0.5; IDCG = 1/log2(2) = 1.0
        assert ndcg_at_k(candidates, expected, k=3) == pytest.approx(0.5)

    def test_ndcg_empty_expected(self):
        """Empty ground truth → NDCG = 0.0."""
        candidates = make_candidates(["p1", "p2"])
        assert ndcg_at_k(candidates, expected_pages=set(), k=5) == 0.0

    def test_ndcg_no_relevant_in_retrieved(self):
        """No relevant pages retrieved → NDCG = 0.0."""
        candidates = make_candidates(["p99", "p98", "p97"])
        expected = {"p1", "p2"}
        assert ndcg_at_k(candidates, expected, k=5) == 0.0

    def test_ndcg_ideal_order(self):
        """Relevance at the top maximizes NDCG vs relevance at the bottom."""
        candidates = make_candidates(["p1", "p99", "p2", "p98"])
        expected = {"p1", "p2"}
        ndcg_top = ndcg_at_k(candidates, expected, k=4)
        # Swap: relevant at end
        candidates_bottom = make_candidates(["p99", "p98", "p1", "p2"])
        ndcg_bottom = ndcg_at_k(candidates_bottom, expected, k=4)
        assert ndcg_top > ndcg_bottom

    def test_ndcg_k5_full_coverage(self):
        """All 5 retrieved pages relevant → NDCG@5 = 1.0."""
        candidates = make_candidates(["p1", "p2", "p3", "p4", "p5"])
        expected = {"p1", "p2", "p3", "p4", "p5"}
        assert ndcg_at_k(candidates, expected, k=5) == 1.0


# ---------------------------------------------------------------------------
# coverage@k
# ---------------------------------------------------------------------------


class TestCoverageAtK:
    def test_coverage_full(self):
        """Top-3 contain all 2 expected documents → coverage@3 = 1.0."""
        # page_ids encode doc_id as prefix before "#p"
        candidates = make_candidates(["docA#p001", "docB#p001", "docB#p002"])
        expected_docs = {"docA", "docB"}
        assert coverage_at_k(candidates, expected_docs, k=3) == 1.0

    def test_coverage_partial(self):
        """Only 1 of 2 docs in top-3 → coverage@3 = 0.5."""
        candidates = make_candidates(["docA#p001", "docA#p002", "docA#p003"])
        expected_docs = {"docA", "docB"}
        assert coverage_at_k(candidates, expected_docs, k=3) == 0.5

    def test_coverage_none(self):
        """No expected doc in top-k → coverage = 0.0."""
        candidates = make_candidates(["docX#p001", "docX#p002"])
        expected_docs = {"docA", "docB"}
        assert coverage_at_k(candidates, expected_docs, k=3) == 0.0

    def test_coverage_empty_expected(self):
        """Empty ground truth → coverage = 0.0."""
        candidates = make_candidates(["docA#p001", "docB#p002"])
        assert coverage_at_k(candidates, expected_documents=set(), k=3) == 0.0

    def test_coverage_k_larger_than_candidates(self):
        """k=10 but only 3 candidates → works correctly."""
        candidates = make_candidates(["docA#p001"])
        expected_docs = {"docA", "docB"}
        assert coverage_at_k(candidates, expected_docs, k=10) == 0.5


# ---------------------------------------------------------------------------
# compute_retrieval_metrics
# ---------------------------------------------------------------------------


class TestComputeRetrievalMetrics:
    def test_all_metrics_full_recall(self):
        """Perfect retrieval: recall@10=1, MRR=1, NDCG=1, coverage=1."""
        candidates = make_candidates([
            "docA#p001", "docA#p002", "docB#p001", "docB#p002",
        ])
        expected_pages = {"docA#p001", "docA#p002", "docB#p001", "docB#p002"}
        expected_docs = {"docA", "docB"}

        result = compute_retrieval_metrics(candidates, expected_pages, expected_docs)

        assert result.recall_at_10 == 1.0
        assert result.mrr_at_10 == 1.0
        assert result.ndcg_at_10 == 1.0
        assert result.coverage_at_10 == 1.0

    def test_all_metrics_zero(self):
        """Zero recall: no relevant pages retrieved."""
        candidates = make_candidates(["p99", "p98"])
        expected_pages = {"p1", "p2"}
        expected_docs = {"docX"}

        result = compute_retrieval_metrics(candidates, expected_pages, expected_docs)

        assert result.recall_at_10 == 0.0
        assert result.mrr_at_10 == 0.0
        assert result.ndcg_at_10 == 0.0
        assert result.coverage_at_10 == 0.0

    def test_result_dataclass_fields(self):
        """Result has all expected fields."""
        result = compute_retrieval_metrics(
            candidates=make_candidates(["p1"]),
            expected_pages=set(),
            expected_documents=set(),
        )
        assert isinstance(result, RetrievalMetricsResult)
        assert hasattr(result, "recall_at_1")
        assert hasattr(result, "recall_at_5")
        assert hasattr(result, "recall_at_10")
        assert hasattr(result, "recall_at_20")
        assert hasattr(result, "mrr_at_10")
        assert hasattr(result, "ndcg_at_5")
        assert hasattr(result, "ndcg_at_10")
        assert hasattr(result, "ndcg_at_20")
        assert hasattr(result, "coverage_at_10")

    def test_as_dict(self):
        """as_dict returns all metric keys."""
        result = RetrievalMetricsResult()
        d = result.as_dict()
        assert "recall@1" in d
        assert "recall@10" in d
        assert "mrr@10" in d
        assert "ndcg@10" in d
        assert "coverage@10" in d
