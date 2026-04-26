"""Qdrant client for Urban RAG — collection bootstrap and schema management.

This module implements the collection bootstrap per PLAN.md Part VI §6.3:
- pages_visual: multi-vector (patches + pooled) with MAX_SIM comparator
- pages_text: text embedding collection for GTE-ModernColBERT
- Scalar quantization on visual collection for INT8 compression
- Idempotent collection creation (skip if exists)
- Alias-based versioning for zero-downtime re-indexing

The visual embedding dimension is 128 (ColQwen2.5 ColPali projection).
The text embedding dimension is 768 (GTE-ModernColBERT).
"""

from __future__ import annotations

import logging

from qdrant_client import models
from qdrant_client.qdrant_client import QdrantClient

from urban_rag.common.errors import ServiceUnavailableError
from urban_rag.common.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — collection names and vector dimensions
# ---------------------------------------------------------------------------

# Visual collection: ColQwen2.5 multi-vector (128-dim per patch)
COLLECTION_PAGES_VISUAL = "pages_visual"
COLQWEN_DIM = 128

# Text collection: GTE-ModernColBERT multi-vector (768-dim per token)
COLLECTION_PAGES_TEXT = "pages_text"
TEXT_EMBED_DIM = 768

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return the module-level Qdrant client singleton.

    The client is created once on first call and reused for all subsequent calls.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=getattr(settings, "qdrant_api_key", None) or None,
            timeout=10,
        )
        logger.info("Qdrant client initialized", url=settings.qdrant_url)
    return _client


def _reset_client() -> None:
    """Reset the singleton — used for testing only."""
    global _client
    _client = None


# ---------------------------------------------------------------------------
# Bootstrap — idempotent collection creation
# ---------------------------------------------------------------------------


def create_collections(client: QdrantClient | None = None) -> None:
    """Create pages_visual and pages_text collections if they don't exist.

    This function is idempotent: running it multiple times is a no-op once
    collections exist. We use collection_exists() to check before creating,
    which avoids the error Qdrant raises when trying to create an existing
    collection with different parameters.

    Args:
        client: Optional Qdrant client. If None, uses the singleton.

    Raises:
        ServiceUnavailableError: If Qdrant is unreachable.
    """
    if client is None:
        client = get_qdrant_client()

    try:
        _create_visual_collection(client)
        _create_text_collection(client)
    except Exception as e:
        logger.error("Failed to bootstrap Qdrant collections", error=str(e))
        raise ServiceUnavailableError(
            f"Failed to bootstrap Qdrant collections: {e}"
        ) from e


def _create_visual_collection(client: QdrantClient) -> None:
    """Create the pages_visual collection with multi-vector + scalar quantization.

    Schema per PLAN.md §6.3:
    - patches: 128-dim multivector with MAX_SIM comparator
    - pooled: 128-dim single vector for ANN candidate pre-filtering
    - HNSW m=16, ef_construct=128
    - Scalar quantization: INT8, 0.99 quantile, always_ram=True
    """
    name = COLLECTION_PAGES_VISUAL
    if client.collection_exists(name):
        logger.debug("Collection already exists, skipping", collection=name)
        return

    try:
        client.create_collection(
            collection_name=name,
            vectors_config={
                "patches": models.VectorParams(
                    size=COLQWEN_DIM,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                ),
                "pooled": models.VectorParams(
                    size=COLQWEN_DIM,
                    distance=models.Distance.COSINE,
                ),
            },
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=128,
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
        )
        logger.info("Collection created", collection=name)
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.debug("Collection created by another process", collection=name)
            return
        logger.error("Failed to create collection", collection=name, error=str(e))
        raise


def _create_text_collection(client: QdrantClient) -> None:
    """Create the pages_text collection for GTE-ModernColBERT text embeddings.

    Schema per PLAN.md §6.3:
    - text: 768-dim single-vector per text chunk
    - Optional sparse BM25 field added at index time
    """
    name = COLLECTION_PAGES_TEXT
    if client.collection_exists(name):
        logger.debug("Collection already exists, skipping", collection=name)
        return

    try:
        client.create_collection(
            collection_name=name,
            vectors_config={
                "text": models.VectorParams(
                    size=TEXT_EMBED_DIM,
                    distance=models.Distance.COSINE,
                ),
            },
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=128,
            ),
            # No scalar quantization on text collection — different compression profile
        )
        logger.info("Collection created", collection=name)
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.debug("Collection created by another process", collection=name)
            return
        logger.error("Failed to create collection", collection=name, error=str(e))
        raise


def delete_collections(client: QdrantClient | None = None) -> None:
    """Delete both pages_visual and pages_text collections.

    WARNING: This deletes all indexed data. Use only in development or when
    resetting the corpus.

    Args:
        client: Optional Qdrant client. If None, uses the singleton.
    """
    if client is None:
        client = get_qdrant_client()

    for name in (COLLECTION_PAGES_VISUAL, COLLECTION_PAGES_TEXT):
        try:
            if client.collection_exists(name):
                client.delete_collection(name)
                logger.info("Collection deleted", collection=name)
        except Exception as e:
            logger.warning("Failed to delete collection", collection=name, error=str(e))


# ---------------------------------------------------------------------------
# Collection info helpers
# ---------------------------------------------------------------------------


def get_collection_info(
    collection_name: str, client: QdrantClient | None = None
) -> models.CollectionInfo | None:
    """Return info about a collection, or None if it doesn't exist.

    Args:
        collection_name: Name of the collection.
        client: Optional Qdrant client. If None, uses the singleton.

    Returns:
        CollectionInfo if the collection exists, None otherwise.
    """
    if client is None:
        client = get_qdrant_client()

    if not client.collection_exists(collection_name):
        return None

    return client.get_collection(collection_name)


def list_collections(client: QdrantClient | None = None) -> list[str]:
    """Return the list of collection names in Qdrant.

    Args:
        client: Optional Qdrant client. If None, uses the singleton.

    Returns:
        List of collection name strings.
    """
    if client is None:
        client = get_qdrant_client()

    return [c.name for c in client.get_collections().collections]
