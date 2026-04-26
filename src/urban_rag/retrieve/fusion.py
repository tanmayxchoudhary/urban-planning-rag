"""Reciprocal Rank Fusion (RRF) across visual + text + sparse retrieval channels.

This module implements the fusion stage per PLAN.md Part VII §7.4:
- RRF k=60 constant (standard in the IR literature)
- Combines ranked lists from visual, text, and sparse channels
- Returns top-N candidates with per-channel provenance (channel_scores, channel_ranks)
- Each candidate carries which channels contributed and at what rank/score

RRF Score Formula:
    RRF_score(candidate) = Σ_{channel ∈ channels} 1 / (k + rank_channel(candidate))

where k=60 is the fusion constant and rank is 1-indexed (rank 1 = top result).

Example:
    - Visual rank 1 → contributes 1/(60+1) = 0.0164
    - Text rank 3 → contributes 1/(60+3) = 0.0159
    - Sparse rank 10 → contributes 1/(60+10) = 0.0143
    - Total RRF = 0.0466
"""

from __future__ import annotations

import time
from typing import Literal

import structlog

from urban_rag.common.types import RetrievalCandidate, RetrievalResult

logger = structlog.get_logger(__name__, service="retrieve-fusion")

# ---------------------------------------------------------------------------
# RRF constants
# ---------------------------------------------------------------------------

RRF_K: int = 60  # Standard RRF constant from IR literature


# ---------------------------------------------------------------------------
# Core RRF fusion
# ---------------------------------------------------------------------------


def fuse_candidates(
    visual_candidates: list[RetrievalCandidate] | None = None,
    text_candidates: list[RetrievalCandidate] | None = None,
    sparse_candidates: list[RetrievalCandidate] | None = None,
    top_n: int = 20,
) -> list[RetrievalCandidate]:
    """Fuse ranked retrieval candidates from multiple channels using RRF.

    Args:
        visual_candidates: Ordered list from visual channel (best first).
        text_candidates: Ordered list from text channel (best first).
        sparse_candidates: Ordered list from sparse channel (best first).
        top_n: Maximum number of fused candidates to return.

    Returns:
        List of fused RetrievalCandidate objects with channel_scores and
        channel_ranks populated. Candidates are ordered by RRF score (descending).

    Note:
        - A candidate can appear in multiple channels
        - channel_scores and channel_ranks only include channels where the
          candidate actually appeared
        - The final score is the RRF sum; rerank_score is not set here (that's
          for the VLM rerank stage)
    """
    # Track best rank per channel for each page_id
    # page_id → {channel: (rank, score)}
    page_channel_data: dict[str, dict[str, tuple[int, float]]] = {}

    def _index_channel(
        candidates: list[RetrievalCandidate] | None,
        channel_name: Literal["visual", "text", "sparse"],
    ) -> None:
        if candidates is None:
            return
        for rank, candidate in enumerate(candidates, start=1):
            if candidate.page_id not in page_channel_data:
                page_channel_data[candidate.page_id] = {}
            # Only record the best (lowest) rank per channel
            if channel_name not in page_channel_data[candidate.page_id]:
                page_channel_data[candidate.page_id][channel_name] = (
                    rank,
                    candidate.score,
                )
            else:
                # Already have this channel for this page — keep the better rank
                existing_rank = page_channel_data[candidate.page_id][channel_name][0]
                if rank < existing_rank:
                    page_channel_data[candidate.page_id][channel_name] = (
                        rank,
                        candidate.score,
                    )

    _index_channel(visual_candidates, "visual")
    _index_channel(text_candidates, "text")
    _index_channel(sparse_candidates, "sparse")

    # Compute RRF scores
    scored_pages: list[tuple[str, float, dict[str, tuple[int, float]]]] = []
    for page_id, channel_data in page_channel_data.items():
        rrf_score = 0.0
        for _channel, (rank, _score) in channel_data.items():
            rrf_score += 1.0 / (RRF_K + rank)
        scored_pages.append((page_id, rrf_score, channel_data))

    # Sort by RRF score descending
    scored_pages.sort(key=lambda x: -x[1])

    # Build fused candidates with provenance
    fused: list[RetrievalCandidate] = []
    for page_id, rrf_score, channel_data in scored_pages[:top_n]:
        # Gather channel_scores and channel_ranks
        channel_scores: dict[str, float] = {}
        channel_ranks: dict[str, int] = {}

        for channel, (rank, score) in channel_data.items():
            channel_scores[channel] = score
            channel_ranks[channel] = rank

        # Use the best candidate from any channel as template
        # Find it from one of the input lists
        template = _find_template(
            page_id,
            visual_candidates,
            text_candidates,
            sparse_candidates,
        )

        fused_candidate = RetrievalCandidate(
            page_id=page_id,
            score=rrf_score,
            channel_scores=channel_scores,
            channel_ranks=channel_ranks,
            rerank_score=template.rerank_score if template else None,
            rerank_rationale=template.rerank_rationale if template else None,
            page_image_uri=template.page_image_uri if template else "",
            extracted_text_excerpt=template.extracted_text_excerpt if template else "",
            section_title=template.section_title if template else None,
        )
        fused.append(fused_candidate)

    return fused


def _find_template(
    page_id: str,
    visual_candidates: list[RetrievalCandidate] | None,
    text_candidates: list[RetrievalCandidate] | None,
    sparse_candidates: list[RetrievalCandidate] | None,
) -> RetrievalCandidate | None:
    """Find the best template candidate for a page_id across all channels.

    Prefers the candidate from the channel where the page has the best
    (lowest) rank. This ensures the template reflects the strongest
    retrieval signal for this page.
    """
    candidates_by_channel = [
        (visual_candidates, "visual"),
        (text_candidates, "text"),
        (sparse_candidates, "sparse"),
    ]

    best_candidate: RetrievalCandidate | None = None
    best_rank = float("inf")

    for candidates, _channel in candidates_by_channel:
        if candidates is None:
            continue
        for c in candidates:
            if c.page_id == page_id:
                rank = c.channel_ranks.get(_channel, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_candidate = c

    return best_candidate


# ---------------------------------------------------------------------------
# Fusion with timing and diagnostics
# ---------------------------------------------------------------------------


def fuse_results(
    visual_result: RetrievalResult | None = None,
    text_result: RetrievalResult | None = None,
    sparse_result: RetrievalResult | None = None,
    top_n: int = 20,
) -> RetrievalResult:
    """Fuse multiple RetrievalResult objects and return a fused RetrievalResult.

    This is the main entrypoint for the fusion stage. It takes individual channel
    results and produces a fused result with full provenance metadata.

    Args:
        visual_result: Result from visual retrieval channel.
        text_result: Result from text retrieval channel.
        sparse_result: Result from sparse retrieval channel.
        top_n: Maximum number of candidates in the fused result.

    Returns:
        RetrievalResult with:
        - candidates: Fused candidates with channel_scores and channel_ranks
        - latency_ms: Time spent in fusion
        - expanded_queries: Empty (filled by query expansion in orchestrator)
        - retrieval_strategy: "hybrid"
        - flags: Any degradation flags from channels

    Note:
        If a channel result is None or has no candidates, it is skipped
        in the fusion (graceful degradation).
    """
    start_time = time.perf_counter()

    logger.info(
        "fusion_start",
        visual_candidates=len(visual_result.candidates) if visual_result else 0,
        text_candidates=len(text_result.candidates) if text_result else 0,
        sparse_candidates=len(sparse_result.candidates) if sparse_result else 0,
        top_n=top_n,
    )

    # Extract candidate lists
    visual_candidates = visual_result.candidates if visual_result else None
    text_candidates = text_result.candidates if text_result else None
    sparse_candidates = sparse_result.candidates if sparse_result else None

    # Fuse
    fused_candidates = fuse_candidates(
        visual_candidates=visual_candidates,
        text_candidates=text_candidates,
        sparse_candidates=sparse_candidates,
        top_n=top_n,
    )

    fusion_ms = int((time.perf_counter() - start_time) * 1000)

    # Collect flags from input results
    flags: dict[str, bool] = {}
    for result in [visual_result, text_result, sparse_result]:
        if result and result.flags:
            flags.update(result.flags)

    # Determine query from inputs (visual > text > sparse priority)
    query = ""
    for result in [visual_result, text_result, sparse_result]:
        if result:
            query = result.query
            break

    # Determine retrieval strategy
    active_channels = []
    if visual_result and visual_result.candidates:
        active_channels.append("visual")
    if text_result and text_result.candidates:
        active_channels.append("text")
    if sparse_result and sparse_result.candidates:
        active_channels.append("sparse")

    if len(active_channels) == 3:
        strategy: Literal["visual_primary", "text_primary", "hybrid"] = "hybrid"
    elif len(active_channels) == 1:
        strategy = f"{active_channels[0]}_primary"  # type: ignore
    else:
        strategy = "hybrid"

    result = RetrievalResult(
        query=query,
        expanded_queries=[],
        candidates=fused_candidates,
        latency_ms=fusion_ms,
        flags=flags,
        retrieval_strategy=strategy,
    )

    logger.info(
        "fusion_complete",
        fused_candidates=len(fused_candidates),
        active_channels=active_channels,
        fusion_ms=fusion_ms,
    )

    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test() -> dict:
    """Run a smoke test of the RRF fusion with synthetic data.

    Verifies:
    - RRF k=60 constant is used
    - Candidates from all contributing channels appear in top-N
    - channel_scores and channel_ranks are populated correctly

    Returns:
        Dict with smoke test results.
    """
    from urban_rag.common.types import RetrievalCandidate

    # Synthetic visual candidates
    visual_candidates = [
        RetrievalCandidate(
            page_id="doc1#p001",
            score=12.5,
            channel_scores={"visual": 12.5},
            channel_ranks={"visual": 1},
            page_image_uri="s3://pages/doc1/p001.png",
            extracted_text_excerpt="Visual page 1",
        ),
        RetrievalCandidate(
            page_id="doc1#p002",
            score=11.2,
            channel_scores={"visual": 11.2},
            channel_ranks={"visual": 2},
            page_image_uri="s3://pages/doc1/p002.png",
            extracted_text_excerpt="Visual page 2",
        ),
    ]

    # Synthetic text candidates
    text_candidates = [
        RetrievalCandidate(
            page_id="doc1#p001",
            score=0.82,
            channel_scores={"text": 0.82},
            channel_ranks={"text": 1},
            page_image_uri="s3://pages/doc1/p001.png",
            extracted_text_excerpt="Text page 1",
        ),
        RetrievalCandidate(
            page_id="doc2#p010",
            score=0.75,
            channel_scores={"text": 0.75},
            channel_ranks={"text": 2},
            page_image_uri="s3://pages/doc2/p010.png",
            extracted_text_excerpt="Text page 10",
        ),
    ]

    # Synthetic sparse candidates
    sparse_candidates = [
        RetrievalCandidate(
            page_id="doc1#p002",
            score=4.1,
            channel_scores={"sparse": 4.1},
            channel_ranks={"sparse": 1},
            page_image_uri="s3://pages/doc1/p002.png",
            extracted_text_excerpt="Sparse page 2",
        ),
        RetrievalCandidate(
            page_id="doc3#p005",
            score=3.8,
            channel_scores={"sparse": 3.8},
            channel_ranks={"sparse": 2},
            page_image_uri="s3://pages/doc3/p005.png",
            extracted_text_excerpt="Sparse page 5",
        ),
    ]

    fused = fuse_candidates(
        visual_candidates=visual_candidates,
        text_candidates=text_candidates,
        sparse_candidates=sparse_candidates,
        top_n=10,
    )

    # Verify RRF formula: doc1#p001 has visual rank 1 + text rank 1
    # RRF = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0328
    doc1_p001 = next((c for c in fused if c.page_id == "doc1#p001"), None)
    assert doc1_p001 is not None, "doc1#p001 should be in fused results"

    expected_rrf = 2.0 / (RRF_K + 1)  # appears in 2 channels, each rank 1
    assert abs(doc1_p001.score - expected_rrf) < 0.001, (
        f"RRF score for doc1#p001 should be {expected_rrf:.6f}, got {doc1_p001.score:.6f}"
    )

    # Verify channel_scores and channel_ranks
    assert doc1_p001.channel_scores == {"visual": 12.5, "text": 0.82}, (
        f"channel_scores mismatch: {doc1_p001.channel_scores}"
    )
    assert doc1_p001.channel_ranks == {"visual": 1, "text": 1}, (
        f"channel_ranks mismatch: {doc1_p001.channel_ranks}"
    )

    # Verify doc1#p002 appears (visual rank 2, sparse rank 1)
    doc1_p002 = next((c for c in fused if c.page_id == "doc1#p002"), None)
    assert doc1_p002 is not None, "doc1#p002 should be in fused results"
    expected_rrf_p002 = 1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1)
    assert abs(doc1_p002.score - expected_rrf_p002) < 0.001

    logger.info(
        "smoke_test_passed",
        fused_count=len(fused),
        doc1_p001_score=doc1_p001.score,
        doc1_p002_score=doc1_p002.score,
        rrf_k=RRF_K,
    )

    return {
        "passed": True,
        "fused_count": len(fused),
        "doc1_p001_score": doc1_p001.score,
        "doc1_p002_score": doc1_p002.score,
        "rrf_k": RRF_K,
    }
