"""Retrieval metrics: recall@k, MRR, NDCG@K, and coverage.

Implements the deterministic retrieval metrics defined in PLAN.md §10.3.1.
All metrics operate on synthetic toy data with known ground-truth in unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from urban_rag.common.types import RetrievalCandidate


@dataclass(frozen=True)
class RetrievalMetricsResult:
    """Aggregated retrieval metrics for a single query."""

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    coverage_at_10: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "recall@20": self.recall_at_20,
            "mrr@10": self.mrr_at_10,
            "ndcg@5": self.ndcg_at_5,
            "ndcg@10": self.ndcg_at_10,
            "ndcg@20": self.ndcg_at_20,
            "coverage@10": self.coverage_at_10,
        }


def recall_at_k(
    candidates: list[RetrievalCandidate],
    expected_pages: set[str],
    k: int,
) -> float:
    """Fraction of expected pages retrieved in top-k.

    Args:
        candidates: Ordered list of retrieval candidates (best first).
        expected_pages: Set of page_ids that are relevant.
        k: Cut-off rank.

    Returns:
        Fraction in [0.0, 1.0].
    """
    if not expected_pages:
        return 0.0
    top_k = {c.page_id for c in candidates[:k]}
    return len(top_k & expected_pages) / len(expected_pages)


def mean_reciprocal_rank(
    candidates: list[RetrievalCandidate],
    expected_pages: set[str],
    k: int = 10,
) -> float:
    """Mean reciprocal rank of the first correct page in top-k.

    Args:
        candidates: Ordered list of retrieval candidates (best first).
        expected_pages: Set of page_ids that are relevant.
        k: Cut-off rank.

    Returns:
        MRR in [0.0, 1.0]; 0.0 if no expected page appears in top-k.
    """
    for i, candidate in enumerate(candidates[:k], start=1):
        if candidate.page_id in expected_pages:
            return 1.0 / i
    return 0.0


def _dcg(gains: list[float], k: int) -> float:
    """Discounted cumulative gain at rank k."""
    dcg = 0.0
    for i, g in enumerate(gains[:k], start=1):
        dcg += g / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    candidates: list[RetrievalCandidate],
    expected_pages: set[str],
    k: int,
) -> float:
    """Normalized DCG at rank k.

    Args:
        candidates: Ordered list of retrieval candidates (best first).
        expected_pages: Set of page_ids that are relevant.
        k: Cut-off rank.

    Returns:
        NDCG in [0.0, 1.0]; 0.0 if no expected pages retrieved.
    """
    # Binary relevance gains
    gains = [1.0 if c.page_id in expected_pages else 0.0 for c in candidates[:k]]
    dcg_k = _dcg(gains, k)

    # Ideal DCG: all relevant docs at the top
    num_relevant = min(len(expected_pages), k)
    ideal_gains = [1.0] * num_relevant + [0.0] * (k - num_relevant)
    idcg_k = _dcg(ideal_gains, k)

    if idcg_k == 0.0:
        return 0.0
    return dcg_k / idcg_k


def coverage_at_k(
    candidates: list[RetrievalCandidate],
    expected_documents: set[str],
    k: int,
) -> float:
    """Fraction of expected documents represented in top-k candidates.

    Args:
        candidates: Ordered list of retrieval candidates.
        expected_documents: Set of doc_ids that contain relevant content.
        k: Cut-off rank.

    Returns:
        Fraction in [0.0, 1.0].
    """
    if not expected_documents:
        return 0.0
    # doc_id is derived from page_id (format: "{doc_id}#p{page_num:04d}")
    top_k_docs = {c.page_id.rsplit("#p", 1)[0] for c in candidates[:k]}
    return len(top_k_docs & expected_documents) / len(expected_documents)


def compute_retrieval_metrics(
    candidates: list[RetrievalCandidate],
    expected_pages: set[str],
    expected_documents: set[str],
) -> RetrievalMetricsResult:
    """Compute all deterministic retrieval metrics for a single query.

    Args:
        candidates: Ordered retrieval candidates (best first).
        expected_pages: Ground-truth relevant page_ids.
        expected_documents: Ground-truth relevant doc_ids.

    Returns:
        RetrievalMetricsResult with all metrics populated.
    """
    return RetrievalMetricsResult(
        recall_at_1=recall_at_k(candidates, expected_pages, k=1),
        recall_at_5=recall_at_k(candidates, expected_pages, k=5),
        recall_at_10=recall_at_k(candidates, expected_pages, k=10),
        recall_at_20=recall_at_k(candidates, expected_pages, k=20),
        mrr_at_10=mean_reciprocal_rank(candidates, expected_pages, k=10),
        ndcg_at_5=ndcg_at_k(candidates, expected_pages, k=5),
        ndcg_at_10=ndcg_at_k(candidates, expected_pages, k=10),
        ndcg_at_20=ndcg_at_k(candidates, expected_pages, k=20),
        coverage_at_10=coverage_at_k(candidates, expected_documents, k=10),
    )
