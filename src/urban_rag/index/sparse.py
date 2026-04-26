"""Sparse BM25 indexer using Qdrant native sparse vectors.

This module implements the BM25 sparse index per PLAN.md Part VI §5.8:
- Computes BM25 token weights for each text chunk
- Stores sparse vectors in Qdrant using the native sparse vector field
- Enables keyword-based retrieval (rare terms like clause numbers, annexures)

BM25 (Best Matching 25) is a probabilistic ranking function used for information retrieval.
Qdrant supports sparse vectors natively via an inverted index.

Usage:
    python -m urban_rag.index.sparse [--docs-dir DATA/docs] [--manifest PATH]
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from qdrant_client.qdrant_client import QdrantClient

from urban_rag.common.errors import IndexingError, ServiceUnavailableError, ValidationError
from urban_rag.common.settings import get_settings
from urban_rag.index.qdrant_client import (
    COLLECTION_PAGES_TEXT,
    get_qdrant_client,
)

if TYPE_CHECKING:
    from urban_rag.index.text_index import ChunkIndexEntry

logger = structlog.get_logger(__name__, service="index-sparse")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# BM25 parameters
BM25_K1: float = 1.5  # Term frequency saturation parameter
BM25_B: float = 0.75  # Length normalization parameter
# Average document length is computed at index time

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens.

    Uses simple whitespace + punctuation splitting suitable for legal/
    planning documents. Avoids stemming for better precision on
    clause numbers and technical terms.

    Args:
        text: Input text string.

    Returns:
        List of lowercase tokens.
    """
    # Lowercase and split on non-alphanumeric
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    # Filter empty tokens and very short ones
    return [t for t in tokens if len(t) >= 2]


def _compute_term_frequencies(texts: list[str]) -> list[dict[str, int]]:
    """Compute term frequencies for each document.

    Args:
        texts: List of text strings.

    Returns:
        List of term frequency dicts (one per document).
    """
    tfs: list[dict[str, int]] = []
    for text in texts:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        tfs.append(tf)
    return tfs


# ---------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------


class BM25Scorer:
    """BM25 scorer for sparse vector computation.

    Computes BM25 scores for all terms across a corpus. This is used
    to build sparse vectors for Qdrant indexing and to score queries.
    """

    def __init__(self, texts: list[str]) -> None:
        """Initialize BM25 scorer with corpus texts.

        Args:
            texts: List of corpus text strings.
        """
        self.texts = texts
        self.N = len(texts)

        # Compute term frequencies for all docs
        self._term_freqs = _compute_term_frequencies(texts)

        # Build vocabulary and compute IDF
        self._vocab: set[str] = set()
        self._doc_freqs: dict[str, int] = {}
        self._avgdl = 0.0

        for tf in self._term_freqs:
            self._vocab.update(tf.keys())
            self._avgdl += sum(tf.values())

            for term in tf:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        if self.N > 0:
            self._avgdl /= self.N

        # Compute IDF for all terms
        self._idf: dict[str, float] = {}
        for term in self._vocab:
            df = self._doc_freqs.get(term, 0)
            if df > 0:
                self._idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            else:
                self._idf[term] = 0.0

        logger.info(
            "BM25 scorer initialized",
            vocab_size=len(self._vocab),
            avg_doc_len=self._avgdl,
        )

    def compute_sparse_vector(
        self,
        text: str,
        top_k: int = 100,
    ) -> tuple[list[int], list[float]]:
        """Compute BM25-weighted sparse vector for a document.

        Args:
            text: Document text string.
            top_k: Maximum number of top terms to keep (by BM25 score).

        Returns:
            Tuple of (indices, values) for Qdrant SparseVector.
            indices: Term indices (integer positions in vocabulary)
            values: BM25 weights for each term
        """
        tokens = _tokenize(text)
        tf = Counter(tokens)
        doc_len = max(sum(tf.values()), 1)

        # Compute BM25 score for each term
        scores: dict[str, float] = {}
        for term, freq in tf.items():
            if term not in self._idf:
                continue

            idf = self._idf[term]
            # BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
            numerator = freq * (BM25_K1 + 1)
            denominator = freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self._avgdl)
            score = idf * numerator / denominator
            scores[term] = score

        # Sort by score descending and take top_k
        sorted_terms = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        # Build vocabulary index
        vocab_list = sorted(self._vocab)
        vocab_idx = {term: i for i, term in enumerate(vocab_list)}

        indices = [vocab_idx[term] for term, _ in sorted_terms if term in vocab_idx]
        values = [score for _, score in sorted_terms if _ in vocab_idx]

        return indices, values

    def score_query(self, query: str) -> dict[int, float]:
        """Score a query text against the corpus using BM25.

        Args:
            query: Query text string.

        Returns:
            Dict mapping document index (int) to BM25 score.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return {}

        query_tf = Counter(query_tokens)
        if not query_tf:
            return {}

        scores: dict[int, float] = {}
        for i, tf in enumerate(self._term_freqs):
            score = 0.0
            for term, _ in query_tf.items():
                if term not in self._idf:
                    continue
                idf = self._idf[term]
                # Use 1 as document frequency for query terms in scoring
                doc_term_freq = tf.get(term, 0)
                numerator = doc_term_freq * (BM25_K1 + 1)
                denominator = (
                    doc_term_freq
                    + BM25_K1
                    * (1 - BM25_B + BM25_B * sum(tf.values()) / self._avgdl)
                )
                score += idf * numerator / denominator
            scores[i] = score

        return scores


# ---------------------------------------------------------------------------
# Sparse vector upsert
# ---------------------------------------------------------------------------


def _build_sparse_point(
    chunk_entry: ChunkIndexEntry,
    indices: list[int],
    values: list[float],
) -> dict:
    """Build a Qdrant point dict with sparse vector.

    Args:
        chunk_entry: The chunk metadata entry.
        indices: Sparse vector indices (term positions).
        values: Sparse vector values (BM25 weights).

    Returns:
        A dict ready for Qdrant upsert with sparse vector + payload.
    """
    from qdrant_client import models

    return {
        "id": chunk_entry.chunk_id,
        "vector": {
            "sparse": models.SparseVector(indices=indices, values=values),
        },
        "payload": {
            "doc_id": chunk_entry.doc_id,
            "page_id": chunk_entry.page_id,
            "section_id": chunk_entry.section_id,
            "chunk_text": chunk_entry.text[:500],
            "token_count": chunk_entry.token_count,
        },
    }


def _upsert_sparse_batch(
    client: QdrantClient,
    points: list[dict],
    collection_name: str = COLLECTION_PAGES_TEXT,
) -> None:
    """Upsert a batch of points with sparse vectors to Qdrant.

    Args:
        client: QdrantClient instance.
        points: List of point dicts.
        collection_name: Target collection name.
    """
    from qdrant_client import models

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"],
            )
            for p in points
        ],
    )


# ---------------------------------------------------------------------------
# Collection setup for sparse vectors
# ---------------------------------------------------------------------------


def ensure_sparse_config(client: QdrantClient | None = None) -> None:
    """Ensure the pages_text collection has sparse vector config.

    Adds the sparse vector field to the collection if it doesn't exist.
    This is safe to call on an existing collection — it only adds the field.

    Args:
        client: Optional Qdrant client. Uses singleton if None.
    """
    if client is None:
        client = get_qdrant_client()

    from qdrant_client import models

    collection_name = COLLECTION_PAGES_TEXT

    # Check current config
    try:
        info = client.get_collection(collection_name)
        # Check if sparse vector already configured
        # pyright doesn't know params attribute on CollectionInfo stub
        info_params = getattr(info, "params", None)
        vector_fields = getattr(info_params, "vector_fields", None) if info_params else None
        if vector_fields and "sparse" in vector_fields:
            logger.debug("sparse_vector_already_configured")
            return
    except Exception as e:
        logger.debug("sparse_config_check_failed", error=str(e))

    # Add sparse vector config via update
    try:
        client.update_collection(
            collection_name=collection_name,
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            },
        )
        logger.info("sparse_vector_config_added", collection=collection_name)
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            logger.debug("sparse_vector_config_already_exists")
            return
        logger.error("failed_to_add_sparse_config", error=str(e))
        raise


# ---------------------------------------------------------------------------
# Main sparse indexing job
# ---------------------------------------------------------------------------


def run_sparse_index(
    docs_dir: Path | None = None,
    manifest_path: Path | None = None,
    rebuild: bool = False,
) -> dict:
    """Run BM25 sparse indexing for the corpus.

    Pipeline:
        1. Discover text chunks from docs/<hash>/parsed.json
        2. Compute BM25 scores for all terms across corpus
        3. Upsert sparse vectors to Qdrant pages_text collection

    Args:
        docs_dir: Override for data/docs/ directory.
        manifest_path: Override for manifest.parquet path.
        rebuild: If True, re-compute all BM25 scores.

    Returns:
        Dict with keys: total_chunks, indexed_chunks, elapsed_seconds, errors

    Raises:
        IndexingError: If a fatal error occurs.
    """
    # Import here to avoid circular imports
    from urban_rag.index.text_index import _discover_chunks

    settings = get_settings()

    docs_dir = docs_dir or Path(settings.docs_dir)
    manifest_path = manifest_path or Path(settings.manifest_path)

    logger.info(
        "sparse_index_starting",
        docs_dir=str(docs_dir),
        manifest=str(manifest_path),
    )

    # Step 1: Discover chunks
    try:
        chunks = _discover_chunks(docs_dir, manifest_path)
    except ValidationError:
        raise
    except Exception as e:
        raise IndexingError(f"Chunk discovery failed: {e}") from e

    if not chunks:
        raise IndexingError("No text chunks found for sparse indexing")

    # Step 2: Initialize Qdrant
    try:
        client = get_qdrant_client()
    except Exception as e:
        raise IndexingError(f"Failed to connect to Qdrant: {e}") from e

    # Step 3: Ensure sparse config
    try:
        ensure_sparse_config(client)
    except Exception as e:
        raise IndexingError(f"Failed to configure sparse vectors: {e}") from e

    # Step 4: Compute BM25 scorer
    texts = [c.text for c in chunks]
    try:
        scorer = BM25Scorer(texts)
    except Exception as e:
        raise IndexingError(f"BM25 initialization failed: {e}") from e

    # Step 5: Batch upsert sparse vectors
    total_chunks = len(chunks)
    indexed_chunks = 0
    error_count = 0
    errors: list[str] = []
    batch_size = 64
    start_time = time.perf_counter()

    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_chunks = chunks[batch_start:batch_end]
        batch_texts = texts[batch_start:batch_end]

        batch_desc = (
            f"batch {batch_start // batch_size + 1} "
            f"({batch_start + 1}-{batch_end}/{total_chunks})"
        )

        points = []
        try:
            for i, chunk_entry in enumerate(batch_chunks):
                indices, values = scorer.compute_sparse_vector(batch_texts[i])
                if not indices:
                    # Empty document — use a zero sparse vector
                    indices = [0]
                    values = [0.0]
                point = _build_sparse_point(chunk_entry, indices, values)
                points.append(point)
        except Exception as e:
            error_count += 1
            errors.append(f"Batch {batch_desc} sparse computation failed: {e}")
            logger.error("batch_sparse_failed", batch=batch_desc, error=str(e))
            continue

        try:
            _upsert_sparse_batch(client, points)
        except Exception as e:
            error_count += 1
            errors.append(f"Batch {batch_desc} upsert failed: {e}")
            logger.error("batch_upsert_failed", batch=batch_desc, error=str(e))
            continue

        indexed_chunks += len(points)
        logger.info("batch_complete", batch=batch_desc, points=len(points))

    elapsed = time.perf_counter() - start_time

    result = {
        "total_chunks": total_chunks,
        "indexed_chunks": indexed_chunks,
        "batches": (total_chunks + batch_size - 1) // batch_size,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors,
        "error_count": error_count,
    }

    logger.info(
        "sparse_index_complete",
        indexed_chunks=indexed_chunks,
        total_chunks=total_chunks,
        elapsed_seconds=result["elapsed_seconds"],
        error_count=error_count,
    )

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for sparse indexing."""
    import typer

    app = typer.Typer(
        name="sparse-index",
        help="Build BM25 sparse index over parsed document text",
        no_args_is_help=True,
    )

    @app.command()
    def run(
        docs_dir: str = "data/docs",
        manifest: str = "data/manifest.parquet",
    ) -> None:
        """Run sparse (BM25) index on the corpus."""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        try:
            result = run_sparse_index(
                docs_dir=Path(docs_dir),
                manifest_path=Path(manifest),
            )
        except (ValidationError, IndexingError, ServiceUnavailableError) as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from e

        table = Table(title="Sparse Index Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total chunks", str(result["total_chunks"]))
        table.add_row("Indexed chunks", str(result["indexed_chunks"]))
        table.add_row("Batches", str(result["batches"]))
        table.add_row("Elapsed", f"{result['elapsed_seconds']}s")
        table.add_row("Errors", str(result["error_count"]))

        console.print(table)

        if result["errors"]:
            console.print("\n[bold red]Errors:[/bold red]")
            for err in result["errors"]:
                console.print(f"  [red]-[/red] {err}")
            raise typer.Exit(code=1) from None

        console.print("[green]✓[/green] Sparse index complete")

    app()
