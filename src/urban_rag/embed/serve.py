"""FastAPI service for ColQwen embeddings.

This service exposes:
- GET /health - liveness check
- POST /embed - batch image embedding for indexing
- POST /embed_query - text query embedding for retrieval

Used by batch indexing (index-batch-visual) and live queries (retrieve-visual).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from urban_rag.common.errors import EmbeddingError, ServiceUnavailableError
from urban_rag.common.logging import configure_logging
from urban_rag.embed.colqwen import VisualEmbedder, get_embedder, release_embedder

logger = logging.getLogger(__name__)

# Configure logging
configure_logging()
app = FastAPI(
    title="urban-rag-embed",
    description="ColQwen2.5 visual embedding service for Urban RAG",
    version="0.1.0",
)

# CORS for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    """Request to embed a batch of image paths."""

    image_paths: list[str] = Field(
        ...,
        min_length=1,
        description="List of absolute paths to page image PNG files",
    )
    batch_size: int | None = Field(
        default=None,
        description="Optional batch size for processing (default: all at once)",
    )


class EmbedResponse(BaseModel):
    """Response containing batch embeddings."""

    embeddings: list[list[list[float]]] = Field(
        ...,
        description=(
            "List of page embeddings. Each page embedding is a list of "
            "patch vectors (variable length). Each patch vector is a list of floats."
        ),
    )
    model_id: str = Field(..., description="Model used for embedding")
    device: str = Field(..., description="Device used (cuda/cpu)")
    batch_size: int = Field(..., description="Number of images processed")
    latency_ms: int = Field(..., description="Processing time in milliseconds")


class EmbedQueryRequest(BaseModel):
    """Request to embed a text query."""

    query: str = Field(..., min_length=1, description="Text query to embed")
    max_length: int | None = Field(
        default=None,
        description="Optional max query length override",
    )


class EmbedQueryResponse(BaseModel):
    """Response containing query embeddings."""

    query_embedding: list[list[float]] = Field(
        ...,
        description=(
            "Multi-vector query embedding. Shape: (query_tokens, embedding_dim). "
            "Each token has its own vector for MaxSim scoring."
        ),
    )
    model_id: str = Field(..., description="Model used for embedding")
    device: str = Field(..., description="Device used (cuda/cpu)")
    latency_ms: int = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_id: str = Field(..., description="Model identifier")
    device: str = Field(..., description="Device (cuda/cpu)")
    embedding_dim: int = Field(..., description="Embedding dimension")


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

# Global embedder instance (lazy-loaded)
_embedder: VisualEmbedder | None = None


def get_embedder_instance() -> VisualEmbedder:
    """Get or create the global embedder instance.

    Returns:
        VisualEmbedder instance.

    Raises:
        ServiceUnavailableError: If embedder cannot be created.
    """
    global _embedder
    if _embedder is None:
        try:
            _embedder = get_embedder(force_cpu=False)
        except EmbeddingError as e:
            raise ServiceUnavailableError(
                f"Failed to initialize embedder: {e}"
            ) from e
    return _embedder


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up embedder on shutdown."""
    global _embedder
    if _embedder is not None:
        release_embedder()
        _embedder = None
    logger.info("Embed service shutdown complete")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check endpoint.

    Returns basic service health and model info.
    Does not load the model if not already loaded.
    """
    embedder = get_embedder_instance()
    return HealthResponse(
        status="ok",
        model_loaded=embedder.is_loaded,
        model_id=embedder.model_id,
        device=embedder.device,
        embedding_dim=embedder.embedding_dim,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed_images(request: EmbedRequest) -> EmbedResponse:
    """Embed a batch of page images.

    This endpoint is used by the batch indexing pipeline to embed
    page images for storage in Qdrant.

    Args:
        request: Contains list of image paths and optional batch_size.

    Returns:
        EmbedResponse with multi-vector embeddings for each image.

    Raises:
        HTTPException: 400 if paths are invalid, 503 if embedding fails.
    """
    start_time = time.perf_counter()

    # Validate paths
    invalid_paths: list[str] = []
    path_objects: list[Path] = []
    for p in request.image_paths:
        path = Path(p)
        if not path.exists():
            invalid_paths.append(p)
        else:
            path_objects.append(path)

    if invalid_paths:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image paths (not found): {invalid_paths}",
        )

    try:
        embedder = get_embedder_instance()

        # Perform embedding
        embeddings_tensor = embedder.embed_pages(path_objects)

        # Convert to list format
        embeddings_list: list[list[list[float]]] = []
        for page_emb in embeddings_tensor:
            # page_emb shape: (num_patches, embedding_dim)
            page_list: list[list[float]] = page_emb.tolist()
            embeddings_list.append(page_list)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return EmbedResponse(
            embeddings=embeddings_list,
            model_id=embedder.model_id,
            device=embedder.device,
            batch_size=len(path_objects),
            latency_ms=elapsed_ms,
        )

    except EmbeddingError as e:
        logger.exception("Embedding failed", exc_info=e)
        raise HTTPException(
            status_code=503,
            detail=f"Embedding failed: {e}",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error during embedding", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {e}",
        ) from e


@app.post("/embed_query", response_model=EmbedQueryResponse)
async def embed_query(request: EmbedQueryRequest) -> EmbedQueryResponse:
    """Embed a text query for retrieval.

    This endpoint is used by live queries to encode the user question
    for MaxSim scoring against page embeddings in Qdrant.

    Args:
        request: Contains the text query.

    Returns:
        EmbedQueryResponse with multi-vector query embedding.

    Raises:
        HTTPException: 400 if query is empty, 503 if encoding fails.
    """
    start_time = time.perf_counter()

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty",
        )

    try:
        embedder = get_embedder_instance()

        # Perform embedding
        query_tensor = embedder.embed_query(request.query.strip())

        # query_tensor shape: (1, query_tokens, embedding_dim)
        # Squeeze batch dim and convert to list
        query_embedding: list[list[float]] = query_tensor.squeeze(0).tolist()

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return EmbedQueryResponse(
            query_embedding=query_embedding,
            model_id=embedder.model_id,
            device=embedder.device,
            latency_ms=elapsed_ms,
        )

    except EmbeddingError as e:
        logger.exception("Query embedding failed", exc_info=e)
        raise HTTPException(
            status_code=503,
            detail=f"Query embedding failed: {e}",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error during query embedding", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {e}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104 - embed service needs to bind to all interfaces
        port=3102,
        log_config=None,  # We use structlog
    )
