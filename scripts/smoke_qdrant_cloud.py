#!/usr/bin/env python3
"""
Smoke test for Qdrant Cloud provisioning.

This script verifies the end-to-end Qdrant Cloud path:
1. Create collections (pages_visual, pages_text) with correct schema
2. Upsert synthetic multi-vector points (128-dim, ColQwen2.5-compatible)
3. Query and verify results return

Usage:
    # Local Qdrant (default from .env):
    python scripts/smoke_qdrant_cloud.py

    # Explicit cloud URL:
    QDRANT_URL=https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.us-east-1-0.aws.cloud.qdrant.io \
    QDRANT_API_KEY=your_key \
    python scripts/smoke_qdrant_cloud.py

Exit codes:
    0 = success (collections exist, upsert worked, query returns results)
    1 = failure (connection error, schema mismatch, query returned no results)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import structlog
from qdrant_client import models
from qdrant_client.qdrant_client import QdrantClient

from urban_rag.index.qdrant_client import (
    COLLECTION_PAGES_VISUAL,
    COLLECTION_PAGES_TEXT,
    COLQWEN_DIM,
    TEXT_EMBED_DIM,
    create_collections,
    get_qdrant_client,
)
from urban_rag.common.settings import get_settings

logger = structlog.get_logger(__name__, service="smoke-qdrant")

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_synthetic_patch_vectors(n_patches: int = 8, dim: int = COLQWEN_DIM) -> list[list[float]]:
    """Create synthetic patch vectors for smoke testing.

    These are random vectors — not real ColQwen2.5 embeddings.
    They verify the collection schema accepts the right shape.

    Args:
        n_patches: Number of patch vectors per page.
        dim: Embedding dimension (128 for ColQwen2.5).

    Returns:
        List of n_patches vectors, each of length dim.
    """
    rng = np.random.default_rng(seed=42)
    vectors = rng.normal(size=(n_patches, dim)).astype(np.float32)
    # Normalize to unit length (simulates real embeddings)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-8)
    return vectors.tolist()


def make_synthetic_pooled_vector(dim: int = COLQWEN_DIM) -> list[float]:
    """Create a synthetic pooled (mean) vector."""
    rng = np.random.default_rng(seed=42)
    vec = rng.normal(size=dim).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


def make_synthetic_text_vector(dim: int = TEXT_EMBED_DIM) -> list[float]:
    """Create a synthetic text embedding vector."""
    rng = np.random.default_rng(seed=42)
    vec = rng.normal(size=dim).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    cleanup: bool = False,
) -> bool:
    """Run the full smoke test against Qdrant.

    Args:
        qdrant_url: Optional URL override.
        qdrant_api_key: Optional API key override.
        cleanup: If True, delete the test collections after.

    Returns:
        True if all steps pass, False otherwise.
    """
    log = logger.info

    # ── Step 1: Connect ──────────────────────────────────────────────────────
    log("Step 1: Connecting to Qdrant")
    try:
        # Allow override for testing
        if qdrant_url is not None:
            os.environ["QDRANT_URL"] = qdrant_url
        if qdrant_api_key is not None:
            os.environ["QDRANT_API_KEY"] = qdrant_api_key

        # Reset singleton to pick up new env
        import urban_rag.index.qdrant_client as qc_module
        qc_module._client = None

        client = get_qdrant_client()

        # Quick health check
        collections = client.get_collections()
        log("connected", url=get_settings().qdrant_url, existing_collections=len(collections.collections))
    except Exception as e:
        logger.error("Step 1 FAILED: Could not connect to Qdrant", error=str(e))
        return False

    # ── Step 2: Create collections ────────────────────────────────────────────
    log("Step 2: Creating collections")
    try:
        create_collections(client)
        log("collections_created",
            visual=COLLECTION_PAGES_VISUAL,
            text=COLLECTION_PAGES_TEXT)
    except Exception as e:
        logger.error("Step 2 FAILED: Could not create collections", error=str(e))
        return False

    # Verify collections exist
    try:
        assert client.collection_exists(COLLECTION_PAGES_VISUAL), \
            f"{COLLECTION_PAGES_VISUAL} does not exist after create_collections"
        assert client.collection_exists(COLLECTION_PAGES_TEXT), \
            f"{COLLECTION_PAGES_TEXT} does not exist after create_collections"
        log("collections_verified")
    except AssertionError as e:
        logger.error("Step 2 verification FAILED", error=str(e))
        return False

    # ── Step 3: Upsert synthetic points to pages_visual ───────────────────────
    log("Step 3: Upserting synthetic points to pages_visual")
    try:
        synthetic_points = []
        for i in range(5):
            page_id = f"smoke_test_page_{i:04d}"
            patches = make_synthetic_patch_vectors(n_patches=8 + i, dim=COLQWEN_DIM)
            pooled = make_synthetic_pooled_vector(dim=COLQWEN_DIM)

            synthetic_points.append(
                models.PointStruct(
                    id=page_id,
                    vector={
                        "patches": patches,
                        "pooled": pooled,
                    },
                    payload={
                        "doc_hash": "smoke_test_doc",
                        "doc_filename": "smoke_test.pdf",
                        "page_num": i + 1,
                        "page_type": "TEXT" if i % 2 == 0 else "VISUAL",
                        "dpi": 100 if i % 2 == 0 else 250,
                        "section_path": ["Smoke Test", f"Section {i}"],
                        "image_uri": f"smoke://test/page_{i:04d}.png",
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_PAGES_VISUAL,
            points=synthetic_points,
        )
        log("upserted", count=len(synthetic_points))
    except Exception as e:
        logger.error("Step 3 FAILED: Could not upsert points", error=str(e))
        return False

    # ── Step 4: Query pages_visual ──────────────────────────────────────────
    log("Step 4: Querying pages_visual")
    try:
        # Use the pooled vector of page 0 as the query vector
        query_vec = make_synthetic_pooled_vector(dim=COLQWEN_DIM)

        results = client.query_points(
            collection_name=COLLECTION_PAGES_VISUAL,
            query=query_vec,
            using="pooled",
            limit=3,
            with_payload=True,
        )

        log("query_results", returned=len(results.points), top_score=results.points[0].score if results.points else None)

        if len(results.points) == 0:
            logger.error("Step 4 FAILED: Query returned no results")
            return False

        # Verify all returned points have the expected payload structure
        for point in results.points:
            assert point.payload is not None, "Point payload is None"
            assert "doc_hash" in point.payload, "Missing doc_hash in payload"
            assert "page_num" in point.payload, "Missing page_num in payload"
            assert "page_type" in point.payload, "Missing page_type in payload"

        log("query_verified")
    except Exception as e:
        logger.error("Step 4 FAILED: Query failed", error=str(e))
        return False

    # ── Step 5: Upsert synthetic points to pages_text ─────────────────────────
    log("Step 5: Upserting synthetic points to pages_text")
    try:
        text_points = []
        for i in range(5):
            chunk_id = f"smoke_test_chunk_{i:04d}"
            text_vec = make_synthetic_text_vector(dim=TEXT_EMBED_DIM)

            text_points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector={
                        "text": [text_vec],  # multi-vector: list of token vectors
                        "pooled": text_vec,
                    },
                    payload={
                        "doc_id": "smoke_test_doc",
                        "page_id": f"smoke_test_page_{i:04d}",
                        "section_id": f"section_{i}",
                        "chunk_text": f"Smoke test chunk {i} with some planning document text.",
                        "token_count": 10 + i,
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_PAGES_TEXT,
            points=text_points,
        )
        log("text_upserted", count=len(text_points))
    except Exception as e:
        logger.error("Step 5 FAILED: Could not upsert text points", error=str(e))
        return False

    # ── Step 6: Query pages_text ──────────────────────────────────────────────
    log("Step 6: Querying pages_text")
    try:
        query_text_vec = make_synthetic_text_vector(dim=TEXT_EMBED_DIM)

        results_text = client.query_points(
            collection_name=COLLECTION_PAGES_TEXT,
            query=query_text_vec,
            using="pooled",
            limit=3,
            with_payload=True,
        )

        log("text_query_results", returned=len(results_text.points))

        if len(results_text.points) == 0:
            logger.error("Step 6 FAILED: Text query returned no results")
            return False

        log("text_query_verified")
    except Exception as e:
        logger.error("Step 6 FAILED: Text query failed", error=str(e))
        return False

    # ── Step 7: Cleanup (optional) ─────────────────────────────────────────────
    if cleanup:
        log("Step 7: Cleaning up test collections")
        try:
            for name in [COLLECTION_PAGES_VISUAL, COLLECTION_PAGES_TEXT]:
                if client.collection_exists(name):
                    client.delete_collection(name)
                    log("deleted", collection=name)
        except Exception as e:
            logger.warning("Cleanup failed (non-fatal)", error=str(e))

    log("SMOKE TEST PASSED — Qdrant Cloud is correctly provisioned")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point for smoke test."""
    import typer

    app = typer.Typer(
        name="smoke-qdrant",
        help="Smoke test for Qdrant Cloud provisioning",
        no_args_is_help=True,
    )

    @app.command()
    def run(
        url: str | None = None,
        api_key: str | None = None,
        cleanup: bool = False,
    ) -> None:
        """Run smoke test against Qdrant.

        Uses QDRANT_URL and QDRANT_API_KEY from environment by default.
        Override with --url and --api-key flags.

        Use --cleanup to delete test collections after the test.
        """
        success = run_smoke_test(
            qdrant_url=url,
            qdrant_api_key=api_key,
            cleanup=cleanup,
        )

        if not success:
            logger.error("Smoke test FAILED")
            raise typer.Exit(code=1)

        logger.info("Smoke test PASSED")
        raise typer.Exit(code=0)

    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
