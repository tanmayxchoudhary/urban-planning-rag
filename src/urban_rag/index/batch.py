"""Full corpus batch indexing job.

This module implements the batch indexing pipeline per PLAN.md Part VI §6.5:
- Reads page image metadata from manifest + pages.jsonl
- Calls the embed service (POST /embed) in batches of 64
- Upserts multi-vector points to Qdrant pages_visual collection
- Indexes the existing 738-page corpus in ≤ 30 min on H100-equivalent GPU
- Idempotent: re-running does not duplicate points (uses deterministic point IDs)

Usage:
    python -m urban_rag.index.batch [--docs-dir DATA/docs] [--manifest PATH] [--batch-size 64]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd
import structlog
from qdrant_client.qdrant_client import QdrantClient

from urban_rag.common.errors import IndexingError, ServiceUnavailableError, ValidationError
from urban_rag.common.settings import get_settings
from urban_rag.index.qdrant_client import (
    COLLECTION_PAGES_VISUAL,
    get_qdrant_client,
)

logger = structlog.get_logger(__name__, service="index-batch")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class PageIndexEntry:
    """A single page ready for indexing."""

    def __init__(
        self,
        page_id: str,
        doc_hash: str,
        doc_filename: str,
        page_num: int,
        page_type: str,
        dpi_used: int,
        image_path: Path,
        section_path: list[str],
    ) -> None:
        self.page_id = page_id
        self.doc_hash = doc_hash
        self.doc_filename = doc_filename
        self.page_num = page_num
        self.page_type = page_type
        self.dpi_used = dpi_used
        self.image_path = image_path
        self.section_path = section_path

    @property
    def image_uri(self) -> str:
        """Return the LanceDB-style URI for the page image."""
        return f"lance://{self.image_path}"


# ---------------------------------------------------------------------------
# Page discovery
# ---------------------------------------------------------------------------


def _discover_pages(
    docs_dir: Path,
    manifest_path: Path,
    page_images_dir: Path,
) -> list[PageIndexEntry]:
    """Discover all pages ready for indexing from manifest + pages.jsonl files.

    Args:
        docs_dir: Path to data/docs/ containing per-hash directories.
        manifest_path: Path to manifest.parquet.
        page_images_dir: Path to data/page_images/ with legacy flat PNG files.

    Returns:
        List of PageIndexEntry objects for all pages in the corpus.

    Raises:
        ValidationError: If manifest doesn't exist or is empty.
    """
    if not manifest_path.exists():
        raise ValidationError(f"Manifest not found: {manifest_path}")

    df = pd.read_parquet(manifest_path)
    if df.empty:
        raise ValidationError("Manifest is empty — no documents to index")

    entries: list[PageIndexEntry] = []

    for _, row in df.iterrows():
        doc_hash: str = str(row["doc_hash"])  # type: ignore[reportAssignmentType]
        filename: str = str(row["filename"])  # type: ignore[reportAssignmentType]

        # Each doc has a pages.jsonl in docs/<hash>/pages.jsonl
        pages_jsonl_path = docs_dir / doc_hash / "pages.jsonl"
        if not pages_jsonl_path.exists():
            logger.warning(
                "no_pages_jsonl_skipping_document",
                doc_hash=doc_hash[:12],
                filename=filename,
            )
            continue

        # Read page records from pages.jsonl
        with pages_jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                page_record = json.loads(line)

                page_id = page_record["page_id"]
                page_num = page_record["page_num"]
                page_type = page_record.get("page_type", "TEXT")
                dpi_used = page_record.get("dpi_used", 100)
                section_path = page_record.get("section_title", None)

                # Image path: data/page_images/{sanitized_filename}__page_{pagenum:04d}.png
                # The filename is sanitized during ingest; use the original as stored
                # We derive the image path from filename + page number
                sanitized_filename = _sanitize_for_path(filename)
                image_name = f"{sanitized_filename}__page_{page_num:04d}.png"
                image_path = page_images_dir / image_name

                if not image_path.exists():
                    logger.warning(
                        "page_image_not_found_skipping",
                        page_id=page_id,
                        image_path=str(image_path),
                    )
                    continue

                entries.append(
                    PageIndexEntry(
                        page_id=page_id,
                        doc_hash=doc_hash,
                        doc_filename=filename,
                        page_num=page_num,
                        page_type=page_type,
                        dpi_used=dpi_used,
                        image_path=image_path,
                        section_path=section_path or [],
                    )
                )

    logger.info("discovered_pages", total_pages=len(entries))
    return entries


def _sanitize_for_path(filename: str) -> str:
    """Sanitize a filename for use in image path lookup.

    Converts spaces and special characters to underscores but preserves
    the base structure used when page images were written.
    """
    # Replace common special chars with underscore
    result = filename.replace(" ", "_").replace("'", "_").replace('"', "_")
    # Remove any path separators
    result = result.replace("/", "_").replace("\\", "_")
    # Remove multiple underscores
    while "__" in result:
        result = result.replace("__", "_")
    return result


# ---------------------------------------------------------------------------
# Embedding via HTTP
# ---------------------------------------------------------------------------


def _embed_pages_via_http(
    image_paths: list[Path],
    embed_service_url: str,
    timeout_s: float = 300.0,
) -> list[list[list[float]]]:
    """Call the embed service to get multi-vector embeddings for a batch of images.

    Args:
        image_paths: List of paths to page image PNG files.
        embed_service_url: Base URL of the embed service (e.g., http://localhost:3102).
        timeout_s: Request timeout in seconds.

    Returns:
        List of page embeddings. Each page embedding is a list of patch vectors.

    Raises:
        ServiceUnavailableError: If the embed service is unreachable or returns an error.
    """
    payload = {
        "image_paths": [str(p) for p in image_paths],
    }

    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                f"{embed_service_url}/embed",
                json=payload,
            )
    except httpx.TimeoutException as e:
        raise ServiceUnavailableError(
            f"Embed service timed out after {timeout_s}s: {e}"
        ) from e
    except httpx.TransportError as e:
        raise ServiceUnavailableError(
            f"Cannot connect to embed service at {embed_service_url}: {e}"
        ) from e

    if response.status_code != 200:
        raise ServiceUnavailableError(
            f"Embed service returned {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    return data["embeddings"]


def _embed_query_via_http(
    query: str,
    embed_service_url: str,
    timeout_s: float = 60.0,
) -> list[list[float]]:
    """Call the embed service to get a multi-vector query embedding.

    Args:
        query: Text query string.
        embed_service_url: Base URL of the embed service.
        timeout_s: Request timeout.

    Returns:
        Multi-vector query embedding (list of token vectors).
    """
    payload = {"query": query}

    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                f"{embed_service_url}/embed_query",
                json=payload,
            )
    except (httpx.TransportError, httpx.TimeoutException) as e:
        raise ServiceUnavailableError(
            f"Embed service unreachable during query embedding: {e}"
        ) from e

    if response.status_code != 200:
        raise ServiceUnavailableError(
            f"Embed query failed: {response.status_code}"
        )

    return response.json()["query_embedding"]


# ---------------------------------------------------------------------------
# Qdrant upsert
# ---------------------------------------------------------------------------


def _build_point(
    page_entry: PageIndexEntry,
    patch_embeddings: list[list[float]],
) -> dict:
    """Build a Qdrant point dict for upsert.

    Uses the deterministic page_id as the point ID to ensure idempotency.

    Args:
        page_entry: The page metadata entry.
        patch_embeddings: Multi-vector embeddings from the embed service.

    Returns:
        A dict ready for Qdrant upsert with vector + payload.
    """
    # Compute pooled vector as mean of patch vectors (per PLAN §6.1)
    import numpy as np

    patch_array = np.array(patch_embeddings, dtype=np.float32)
    pooled = patch_array.mean(axis=0).tolist()

    return {
        "id": page_entry.page_id,
        "vector": {
            "patches": patch_embeddings,
            "pooled": pooled,
        },
        "payload": {
            "doc_hash": page_entry.doc_hash,
            "doc_filename": page_entry.doc_filename,
            "page_num": page_entry.page_num,
            "page_type": page_entry.page_type,
            "dpi": page_entry.dpi_used,
            "section_path": page_entry.section_path,
            "image_uri": page_entry.image_uri,
        },
    }


def _upsert_batch(
    client: QdrantClient,
    points: list[dict],
    collection_name: str,
) -> None:
    """Upsert a batch of points to Qdrant.

    Args:
        client: QdrantClient instance.
        points: List of point dicts from _build_point.
        collection_name: Target collection name.
    """
    # Import Qdrant models here to avoid import cycle
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
# Main batch indexing job
# ---------------------------------------------------------------------------


def run_batch_index(
    docs_dir: Path | None = None,
    manifest_path: Path | None = None,
    page_images_dir: Path | None = None,
    embed_service_url: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    collection_name: str = COLLECTION_PAGES_VISUAL,
    rebuild: bool = False,
) -> dict:
    """Run the full corpus batch indexing job.

    Pipeline:
        1. Discover all pages from manifest + pages.jsonl
        2. Batch pages (batch_size at a time)
        3. For each batch: call embed service → build points → upsert to Qdrant
        4. Report timing and counts

    Args:
        docs_dir: Override for data/docs/ directory.
        manifest_path: Override for manifest.parquet path.
        page_images_dir: Override for data/page_images/ directory.
        embed_service_url: Override for embed service URL (default: http://localhost:3102).
        batch_size: Number of pages per embed+upsert batch (default: 64).
        collection_name: Qdrant collection name (default: pages_visual).
        rebuild: If True, delete existing collection and re-index from scratch.

    Returns:
        Dict with keys: total_pages, indexed_pages, batches, elapsed_seconds, errors

    Raises:
        IndexingError: If a fatal error occurs during indexing.
    """
    settings = get_settings()

    docs_dir = docs_dir or Path(settings.docs_dir)
    manifest_path = manifest_path or Path(settings.manifest_path)
    page_images_dir = page_images_dir or (
        Path(settings.docs_dir).parent / "page_images"
    )
    embed_service_url = embed_service_url or "http://localhost:3102"

    logger.info(
        "batch_index_starting",
        docs_dir=str(docs_dir),
        manifest=str(manifest_path),
        page_images_dir=str(page_images_dir),
        embed_service=embed_service_url,
        batch_size=batch_size,
        collection=collection_name,
    )

    # ── Step 1: Discover all pages ──────────────────────────────────────────
    try:
        pages = _discover_pages(docs_dir, manifest_path, page_images_dir)
    except ValidationError:
        raise
    except Exception as e:
        raise IndexingError(f"Page discovery failed: {e}") from e

    if not pages:
        raise IndexingError("No pages found to index — check manifest and data/ directory")

    # ── Step 2: Initialize Qdrant client ──────────────────────────────────
    try:
        client = get_qdrant_client()
    except Exception as e:
        raise IndexingError(f"Failed to connect to Qdrant: {e}") from e

    # ── Step 3: Bootstrap collection ─────────────────────────────────────────
    from urban_rag.index.qdrant_client import create_collections

    try:
        create_collections(client)
    except ServiceUnavailableError:
        raise
    except Exception as e:
        raise IndexingError(f"Collection bootstrap failed: {e}") from e

    # ── Step 4: Optionally clear collection (rebuild mode) ─────────────────
    if rebuild:
        try:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
                logger.info("collection_deleted_rebuild", collection=collection_name)
                # Recreate after deletion
                create_collections(client)
        except Exception as e:
            raise IndexingError(f"Rebuild failed: {e}") from e

    # ── Step 5: Batch processing ─────────────────────────────────────────────
    total_pages = len(pages)
    indexed_pages = 0
    error_count = 0
    errors: list[str] = []
    batches = 0
    start_time = time.perf_counter()

    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        batch_pages = pages[batch_start:batch_end]
        batches += 1

        batch_desc = f"batch {batches} ({batch_start + 1}-{batch_end}/{total_pages})"
        logger.info("processing_batch", batch=batch_desc)

        # Collect image paths for this batch
        image_paths = [p.image_path for p in batch_pages]

        # Embed batch via HTTP
        try:
            patch_embeddings = _embed_pages_via_http(
                image_paths,
                embed_service_url,
                timeout_s=300.0,
            )
        except ServiceUnavailableError:
            raise
        except Exception as e:
            error_count += 1
            errors.append(f"Batch {batches} embed failed: {e}")
            logger.error("batch_embed_failed", batch=batch_desc, error=str(e))
            continue

        # Build points
        points = []
        for i, page_entry in enumerate(batch_pages):
            embeddings = patch_embeddings[i]
            points.append(_build_point(page_entry, embeddings))

        # Upsert to Qdrant
        try:
            _upsert_batch(client, points, collection_name)
        except Exception as e:
            error_count += 1
            errors.append(f"Batch {batches} upsert failed: {e}")
            logger.error("batch_upsert_failed", batch=batch_desc, error=str(e))
            continue

        indexed_pages += len(points)
        logger.info(
            "batch_complete",
            batch=batch_desc,
            points=len(points),
            indexed_total=indexed_pages,
        )

    elapsed = time.perf_counter() - start_time

    # ── Step 6: Report ───────────────────────────────────────────────────────
    result = {
        "total_pages": total_pages,
        "indexed_pages": indexed_pages,
        "batches": batches,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors,
        "error_count": error_count,
    }

    logger.info(
        "batch_index_complete",
        indexed_pages=indexed_pages,
        total_pages=total_pages,
        batches=batches,
        elapsed_seconds=result["elapsed_seconds"],
        error_count=error_count,
        throughput_pages_per_sec=round(indexed_pages / elapsed, 2) if elapsed > 0 else 0,
    )

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for batch indexing."""
    import typer

    app = typer.Typer(
        name="batch-index",
        help="Batch index the corpus: embed page images and upsert to Qdrant",
        no_args_is_help=True,
    )

    @app.command()
    def run(
        docs_dir: str = "data/docs",
        manifest: str = "data/manifest.parquet",
        page_images: str = "data/page_images",
        embed_service: str = "http://localhost:3102",
        batch_size: int = DEFAULT_BATCH_SIZE,
        rebuild: bool = False,
    ) -> None:
        """Run batch index on the corpus.

        Reads all pages from manifest + pages.jsonl, embeds via the embed service,
        and upserts to Qdrant pages_visual collection.
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()

        try:
            result = run_batch_index(
                docs_dir=Path(docs_dir),
                manifest_path=Path(manifest),
                page_images_dir=Path(page_images),
                embed_service_url=embed_service,
                batch_size=batch_size,
                rebuild=rebuild,
            )
        except (ValidationError, IndexingError, ServiceUnavailableError) as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from e

        # Print summary table
        table = Table(title="Batch Index Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total pages", str(result["total_pages"]))
        table.add_row("Indexed pages", str(result["indexed_pages"]))
        table.add_row("Batches", str(result["batches"]))
        table.add_row("Elapsed", f"{result['elapsed_seconds']}s")
        if result["error_count"] > 0:
            table.add_row("[red]Errors[/red]", str(result["error_count"]))
        else:
            table.add_row("Errors", "0")

        if result["elapsed_seconds"] > 0:
            rate = result["indexed_pages"] / result["elapsed_seconds"]
            table.add_row("Throughput", f"{rate:.1f} pages/sec")

        console.print(table)

        if result["errors"]:
            console.print("\n[bold red]Errors:[/bold red]")
            for err in result["errors"]:
                console.print(f"  [red]-[/red] {err}")
            raise typer.Exit(code=1) from None

        console.print("[green]✓[/green] Batch index complete")

    @app.command()
    def check() -> None:
        """Check Qdrant collection status."""
        from rich.console import Console

        console = Console()
        client = get_qdrant_client()
        info = client.get_collection(collection_name=COLLECTION_PAGES_VISUAL)

        console.print(f"[cyan]Collection:[/cyan] {COLLECTION_PAGES_VISUAL}")
        console.print(f"[cyan]Points:[/cyan] {info.points_count}")
        console.print(f"[cyan]Status:[/cyan] {info.status}")

    app()
