"""Visual retrieval channel: pooled ANN → MaxSim server-side rerank.

This module implements the visual retrieval pipeline per PLAN.md Part VII §7.3:
- Query encoding: ColQwen2.5 embeds the text query to patch-level query vectors
- Stage 1: ANN over pooled vector (200 candidates) via Qdrant prefetch
- Stage 2: Server-side MaxSim over multi-vector patches → top 20
- Returns list[RetrievalCandidate] with channel_scores and channel_ranks

The key Qdrant pattern (PLAN §6.4):
    client.query_points(
        collection_name="pages_visual",
        prefetch=[
            models.Prefetch(
                query=query_pooled.tolist(),
                using="pooled",
                limit=200,
            ),
        ],
        query=query_patches.tolist(),
        using="patches",
        limit=20,
        with_payload=True,
    )

The visual channel is the primary retrieval signal for layout-heavy planning
documents. It runs in parallel with text and sparse channels, and its candidates
feed into the RRF fusion stage (PLAN §7.4).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from urban_rag.common.errors import RetrievalError, ServiceUnavailableError
from urban_rag.common.types import RetrievalCandidate, RetrievalResult
from urban_rag.index.qdrant_client import (
    COLLECTION_PAGES_VISUAL,
    get_qdrant_client,
)

if TYPE_CHECKING:
    from qdrant_client import models

logger = structlog.get_logger(__name__, service="retrieve-visual")

# ---------------------------------------------------------------------------
# Query encoding
# ---------------------------------------------------------------------------


def _encode_query(embedder, query: str) -> tuple[list[list[float]], list[float]]:
    """Encode a text query to both pooled and patch-level vectors.

    The ColQwen2.5 model produces multi-vector embeddings: one vector per
    visual patch. We use:
    - patch vectors: for MaxSim scoring (late interaction)
    - pooled (mean) vector: for ANN candidate pre-filtering

    Args:
        query: Text query string.
        embedder: VisualEmbedder instance (ColQwen2.5).

    Returns:
        Tuple of (query_patches, query_pooled):
        - query_patches: list of patch vectors for MaxSim scoring
        - query_pooled: single pooled vector for ANN prefetch
    """
    import numpy as np

    # embed_query returns tensor of shape (1, num_query_tokens, embedding_dim)
    query_tensor = embedder.embed_query(query)  # type: ignore[reportCallIssue]
    # Shape: (1, num_tokens, 128)

    # Convert to list of patch vectors for MaxSim
    # Handle both torch.Tensor (has .tolist()) and mock objects (return plain list)
    first_row = query_tensor[0]
    if hasattr(first_row, "tolist"):
        query_patches: list[list[float]] = first_row.tolist()
    else:
        # Already a list (from mock)
        query_patches = list(first_row)

    # Compute pooled vector as mean of patch vectors
    query_array = np.array(query_patches, dtype=np.float32)
    query_pooled: list[float] = query_array.mean(axis=0).tolist()

    return query_patches, query_pooled  # tuple[patch_vectors, pooled_vector]


# ---------------------------------------------------------------------------
# Qdrant search — pooled ANN → MaxSim rerank
# ---------------------------------------------------------------------------


def _qdrant_visual_search(
    client,
    query_patches: list[list[float]],
    query_pooled: list[float],
    top_k: int = 20,
    prefetch_limit: int = 200,
    filters: dict[str, str] | None = None,
) -> list[RetrievalCandidate]:
    """Execute the two-stage visual retrieval on Qdrant.

    Stage 1: ANN over pooled vector → top prefetch_limit candidates
    Stage 2: Server-side MaxSim rerank → top_k results

    Args:
        client: QdrantClient instance.
        query_patches: Multi-vector query embedding (list of patch vectors).
        query_pooled: Pooled query vector (single vector, list of floats).
        top_k: Number of final candidates to return.
        prefetch_limit: Number of ANN candidates to fetch for MaxSim reranking.
        filters: Optional Qdrant payload filters.

    Returns:
        List of RetrievalCandidate objects ordered by MaxSim score.

    Raises:
        ServiceUnavailableError: If Qdrant is unreachable.
        RetrievalError: If the search query fails unexpectedly.
    """
    from qdrant_client import models

    try:
        # Build the prefetch + MaxSim query
        # The query_patches is a list of vectors (multi-vector query)
        # The query_pooled is the single pooled vector for ANN
        search_results = client.query_points(
            collection_name=COLLECTION_PAGES_VISUAL,
            prefetch=[
                models.Prefetch(
                    query=query_pooled,
                    using="pooled",
                    limit=prefetch_limit,
                ),
            ],
            query=query_patches,
            using="patches",
            limit=top_k,
            with_payload=True,
            query_filter=_build_filter(filters) if filters else None,
        )

        # Convert Qdrant results to RetrievalCandidate objects
        candidates: list[RetrievalCandidate] = []
        for scored_point in search_results.points:
            payload = scored_point.payload or {}

            # Decode section_path from stored JSON string or list
            section_path = payload.get("section_path", [])
            if isinstance(section_path, str):
                import json as _json
                try:
                    section_path = _json.loads(section_path)
                except Exception:
                    section_path = []

            candidate = RetrievalCandidate(
                page_id=scored_point.id,
                score=scored_point.score,
                channel_scores={"visual": scored_point.score},
                channel_ranks={"visual": len(candidates) + 1},
                rerank_score=None,
                rerank_rationale=None,
                page_image_uri=payload.get("image_uri", ""),
                extracted_text_excerpt=payload.get("extracted_text", "")[:500],
                section_title=section_path[-1] if section_path else None,
            )
            candidates.append(candidate)

        return candidates

    except Exception as e:
        logger.error("Qdrant visual search failed", error=str(e))
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise ServiceUnavailableError(
                f"Qdrant visual search failed: {e}"
            ) from e
        raise RetrievalError(f"Visual search failed: {e}") from e


def _build_filter(filters: dict[str, str]) -> models.Filter | None:  # type: ignore[valid-type]
    """Build a Qdrant filter from a dict of payload field conditions.

    Args:
        filters: Dict of field -> value constraints.

    Returns:
        Qdrant Filter object or None if filters is empty.
    """
    from qdrant_client import models

    if not filters:
        return None

    # Build filter using dict-based construction (avoids type annotation issues)
    must_clauses = [
        models.FieldCondition(  # type: ignore[reportCallIssue]
            key=field,
            match=models.MatchValue(value=value),  # type: ignore[reportCallIssue]
        )
        for field, value in filters.items()
    ]

    return models.Filter(must=must_clauses)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main visual retrieval function
# ---------------------------------------------------------------------------


def retrieve_visual(
    query: str,
    top_k: int = 20,
    prefetch_limit: int = 200,
    filters: dict[str, str] | None = None,
    embedder=None,
) -> RetrievalResult:
    """Execute the visual retrieval pipeline.

    Pipeline:
        1. Encode query to patch vectors (ColQwen2.5) and pooled vector
        2. ANN search on pooled vector → top prefetch_limit candidates
        3. Server-side MaxSim rerank → top_k candidates
        4. Return RetrievalResult with candidates and diagnostics

    Args:
        query: Text query string.
        top_k: Number of candidates to return (default 20).
        prefetch_limit: Number of ANN candidates for MaxSim reranking (default 200).
        filters: Optional payload filters (jurisdiction, doc_type, etc.).
        embedder: VisualEmbedder instance. If None, loads from singleton.

    Returns:
        RetrievalResult with candidates sorted by MaxSim score.

    Raises:
        RetrievalError: If retrieval fails for a non-service reason.
        ServiceUnavailableError: If Qdrant or the embedder service is unavailable.
    """
    from urban_rag.embed.colqwen import get_embedder

    start_time = time.perf_counter()
    encode_ms = 0
    retrieve_ms = 0

    logger.info("visual_retrieval_start", query=query[:100], top_k=top_k)  # type: ignore

    # ── Step 1: Encode query ──────────────────────────────────────────────────
    encode_start = time.perf_counter()

    if embedder is None:
        embedder = get_embedder()

    try:
        query_patches, query_pooled = _encode_query(embedder, query)
    except Exception as e:
        logger.error("Query encoding failed", error=str(e))
        raise RetrievalError(f"Query encoding failed: {e}") from e

    encode_ms = int((time.perf_counter() - encode_start) * 1000)

    # ── Step 2: Qdrant ANN + MaxSim ───────────────────────────────────────────
    retrieve_start = time.perf_counter()

    try:
        client = get_qdrant_client()
    except Exception as e:
        raise ServiceUnavailableError(f"Cannot connect to Qdrant: {e}") from e

    try:
        candidates = _qdrant_visual_search(
            client=client,
            query_patches=query_patches,
            query_pooled=query_pooled,
            top_k=top_k,
            prefetch_limit=prefetch_limit,
            filters=filters,
        )
    except (ServiceUnavailableError, RetrievalError):
        raise
    except Exception as e:
        raise RetrievalError(f"Visual retrieval failed: {e}") from e

    retrieve_ms = int((time.perf_counter() - retrieve_start) * 1000)
    total_ms = int((time.perf_counter() - start_time) * 1000)

    # ── Step 3: Build result ─────────────────────────────────────────────────
    result = RetrievalResult(
        query=query,
        expanded_queries=[],
        candidates=candidates,
        latency_ms=total_ms,
        flags={},
        retrieval_strategy="visual_primary",
    )

    logger.info(
        "visual_retrieval_complete",
        query=query[:100],
        candidates=len(candidates),
        encode_ms=encode_ms,
        retrieve_ms=retrieve_ms,
        total_ms=total_ms,
    )

    return result


# ---------------------------------------------------------------------------
# Smoke query — verify the pipeline end-to-end
# ---------------------------------------------------------------------------


def smoke_query(
    query: str = "FAR for residential",
    top_k: int = 10,
) -> dict:
    """Run a smoke query to verify the visual retrieval pipeline.

    This is used by the validation contract (VAL expected behavior):
    "Query 'FAR for residential' returns at least 3 URDPFI pages in top 10"

    Args:
        query: The smoke query string.
        top_k: Number of candidates to return.

    Returns:
        Dict with keys: candidates, latency_ms, urdpfi_count.
    """
    from urban_rag.embed.colqwen import get_embedder

    logger.info("smoke_query_start", query=query)

    embedder = get_embedder(force_cpu=True)  # Use CPU if no GPU

    try:
        result = retrieve_visual(
            query=query,
            top_k=top_k,
            embedder=embedder,
        )
    finally:
        # Release embedder to free memory after smoke test
        from urban_rag.embed.colqwen import release_embedder
        release_embedder()

    # Count URDPFI pages in top results
    urdpfi_count = sum(
        1 for c in result.candidates
        if _is_urdpfi_page(c)
    )

    logger.info(
        "smoke_query_complete",
        candidates=len(result.candidates),
        urdpfi_count=urdpfi_count,
        latency_ms=result.latency_ms,
    )

    return {
        "candidates": result.candidates,
        "latency_ms": result.latency_ms,
        "urudpfi_count": urdpfi_count,
        "query": query,
    }


def _is_urdpfi_page(candidate: RetrievalCandidate) -> bool:
    """Check if a candidate is from a URDPFI document.

    URDPFI pages can be identified by:
    - page_id contains 'urdpfi' (case-insensitive)
    - doc_filename contains 'URDPFI' or 'urdpfi'
    - section_title mentions URDPFI

    Args:
        candidate: RetrievalCandidate to check.

    Returns:
        True if the candidate appears to be from a URDPFI document.
    """
    page_id_lower = candidate.page_id.lower()
    if "urdpfi" in page_id_lower:
        return True

    # Check extracted text excerpt for URDPFI mentions
    excerpt_lower = candidate.extracted_text_excerpt.lower()
    if "urdpfi" in excerpt_lower:
        return True

    # Check section title
    return bool(candidate.section_title and "urdpfi" in candidate.section_title.lower())
