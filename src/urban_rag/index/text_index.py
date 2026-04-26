"""Text indexer for GTE-ModernColBERT embeddings on Qdrant pages_text collection.

This module implements the text indexing pipeline per PLAN.md Part VI §6.2:
- Reads chunk text from docs/<hash>/parsed.json and pages.jsonl
- Embeds chunks via GTE-ModernColBERT (CPU-friendly, batched)
- Upserts multi-vector points to Qdrant pages_text collection
- Separate collection from visual, enabling parallel text retrieval

Usage:
    python -m urban_rag.index.text_index [--docs-dir DATA/docs] [--manifest PATH]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import structlog
from qdrant_client.qdrant_client import QdrantClient

from urban_rag.common.errors import IndexingError, ServiceUnavailableError, ValidationError
from urban_rag.common.settings import get_settings
from urban_rag.embed.text_encoder import TextEmbedder, get_text_embedder
from urban_rag.index.qdrant_client import (
    COLLECTION_PAGES_TEXT,
    get_qdrant_client,
)

logger = structlog.get_logger(__name__, service="index-text")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ChunkIndexEntry:
    """A single text chunk ready for indexing."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        page_id: str,
        section_id: str | None,
        text: str,
        token_count: int,
        chunk_index_in_section: int = 0,
    ) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.page_id = page_id
        self.section_id = section_id
        self.text = text
        self.token_count = token_count
        self.chunk_index_in_section = chunk_index_in_section


# ---------------------------------------------------------------------------
# Chunk discovery from parsed.json + pages.jsonl
# ---------------------------------------------------------------------------


def _discover_chunks(
    docs_dir: Path,
    manifest_path: Path,
) -> list[ChunkIndexEntry]:
    """Discover all text chunks from docs/<hash>/parsed.json and pages.jsonl.

    Chunks are extracted from the Docling-parsed document content.
    Each page has its own section; chunks are created per section.

    Args:
        docs_dir: Path to data/docs/ containing per-hash directories.
        manifest_path: Path to manifest.parquet.

    Returns:
        List of ChunkIndexEntry objects for all chunks.

    Raises:
        ValidationError: If manifest doesn't exist or is empty.
    """
    import pandas as pd

    if not manifest_path.exists():
        raise ValidationError(f"Manifest not found: {manifest_path}")

    df = pd.read_parquet(manifest_path)
    if df.empty:
        raise ValidationError("Manifest is empty — no documents to index")

    entries: list[ChunkIndexEntry] = []

    for _, row in df.iterrows():
        doc_hash: str = str(row["doc_hash"])
        filename: str = str(row["filename"])

        # Read parsed.json for full text content
        parsed_path = docs_dir / doc_hash / "parsed.json"
        pages_jsonl_path = docs_dir / doc_hash / "pages.jsonl"

        if not parsed_path.exists():
            logger.warning(
                "no_parsed_json_skipping",
                doc_hash=doc_hash[:12],
                filename=filename,
            )
            continue

        # Load page metadata
        page_metadata: dict[int, dict] = {}
        if pages_jsonl_path.exists():
            with pages_jsonl_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    page_rec = json.loads(line)
                    page_num = page_rec.get("page_num", 0)
                    page_metadata[page_num] = page_rec

        # Parse document structure
        try:
            parsed_data = json.loads(parsed_path.read_text())
        except json.JSONDecodeError as e:
            logger.warning(
                "parsed_json_corrupt_skipping",
                doc_hash=doc_hash[:12],
                filename=filename,
                error=str(e),
            )
            continue

        # Extract text from Docling document
        # Docling format: {"elements": [...]} or {"pages": [...]} or similar
        all_texts = _extract_docling_text(parsed_data)

        # Create chunks per page
        for page_num, texts_for_page in all_texts.items():
            page_rec = page_metadata.get(page_num, {})
            page_id = page_rec.get("page_id", f"{doc_hash}#p{page_num:04d}")
            section_title = page_rec.get("section_title")

            # Section ID derived from doc_hash + section_title
            section_id = None
            if section_title:
                section_id = f"{doc_hash}#s{section_title[:30].replace(' ', '_')}"

            # Chunk the text (target 256 tokens, 32-token overlap)
            page_chunks = _chunk_text(
                texts_for_page,
                doc_id=doc_hash,
                page_id=page_id,
                section_id=section_id,
                chunk_index_offset=len(entries),
            )
            entries.extend(page_chunks)

    logger.info("discovered_chunks", total_chunks=len(entries))
    return entries


def _extract_docling_text(parsed_data: dict) -> dict[int, list[str]]:
    """Extract text content from Docling parsed.json structure.

    DoclingDocument format varies. We handle common patterns:
    - elements: list of element dicts with "text" and "page" fields
    - pages: list of page dicts with "texts" or "lines"
    - flat "text" field for simple extractions

    Returns:
        Dict mapping page_num -> list of text strings for that page.
    """
    page_texts: dict[int, list[str]] = {}

    # Pattern 1: elements list
    if "elements" in parsed_data:
        elements = parsed_data["elements"]
        if isinstance(elements, list):
            for el in elements:
                if not isinstance(el, dict):
                    continue
                text = el.get("text", "")
                if not text or not isinstance(text, str):
                    continue
                # Determine page number
                page_num = el.get("page", el.get("page_num", 1))
                if isinstance(page_num, str):
                    page_num = int(page_num) if page_num.isdigit() else 1
                page_texts.setdefault(int(page_num), []).append(text)

    # Pattern 2: pages list
    elif "pages" in parsed_data:
        pages = parsed_data["pages"]
        if isinstance(pages, list):
            for page_idx, page_el in enumerate(pages, start=1):
                texts_for_page: list[str] = []
                # Check for texts or lines fields
                if isinstance(page_el, dict):
                    texts_field = page_el.get("texts", page_el.get("lines", []))
                    if isinstance(texts_field, list):
                        for txt in texts_field:
                            if isinstance(txt, dict):
                                t = txt.get("text", "")
                            elif isinstance(txt, str):
                                t = txt
                            else:
                                t = ""
                            if t:
                                texts_for_page.append(t)
                page_texts[page_idx] = texts_for_page

    # Pattern 3: flat text field
    elif "text" in parsed_data:
        text_content = parsed_data["text"]
        if isinstance(text_content, str):
            # Split by pages if possible
            lines = text_content.split("\n")
            current_page = 1
            for line in lines:
                if line.strip():
                    page_texts.setdefault(current_page, []).append(line)
                # Simple heuristic: every 50 lines is a new page
                if len(page_texts.get(current_page, [])) >= 50:
                    current_page += 1

    return page_texts


def _chunk_text(
    texts: list[str],
    doc_id: str,
    page_id: str,
    section_id: str | None,
    chunk_index_offset: int = 0,
    target_tokens: int = 256,
    overlap_tokens: int = 32,
) -> list[ChunkIndexEntry]:
    """Split page texts into overlapping chunks of target token size.

    Args:
        texts: List of text strings (paragraphs, lines) from one page.
        doc_id: Document identifier.
        page_id: Page identifier.
        section_id: Optional section identifier.
        chunk_index_offset: Offset for chunk_id numbering.
        target_tokens: Target chunk size in tokens.
        overlap_tokens: Overlap between chunks.

    Returns:
        List of ChunkIndexEntry objects.
    """
    # Combine all texts for this page
    full_text = "\n".join(texts)

    if not full_text.strip():
        return []

    # Simple tokenization: ~4 chars per token average
    # More accurate would be to use the tokenizer
    estimated_tokens = len(full_text) // 4

    if estimated_tokens <= target_tokens:
        # Single chunk
        chunk_id = f"{doc_id}#c{chunk_index_offset:06d}"
        return [
            ChunkIndexEntry(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page_id=page_id,
                section_id=section_id,
                text=full_text.strip(),
                token_count=estimated_tokens,
                chunk_index_in_section=0,
            )
        ]

    # Multiple chunks with overlap
    entries: list[ChunkIndexEntry] = []
    chunk_idx = 0
    start = 0
    text_len = len(full_text)

    while start < text_len:
        end = start + (target_tokens * 4)
        if end >= text_len:
            end = text_len
        else:
            # Try to break at sentence or paragraph boundary
            chunk_text = full_text[start:end]
            for sep in ["\n\n", "\n", ". "]:
                last_sep = chunk_text.rfind(sep)
                if last_sep > int(target_tokens * 2):
                    end = start + last_sep + len(sep)
                    break

        chunk_str = full_text[start:end].strip()
        if not chunk_str:
            start += target_tokens * 4
            continue

        est_tokens = len(chunk_str) // 4
        chunk_id = f"{doc_id}#c{chunk_index_offset + chunk_idx:06d}"
        entries.append(
            ChunkIndexEntry(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page_id=page_id,
                section_id=section_id,
                text=chunk_str,
                token_count=est_tokens,
                chunk_index_in_section=chunk_idx,
            )
        )

        chunk_idx += 1
        start = end - (overlap_tokens * 4)

    return entries


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _embed_chunks(
    embedder: TextEmbedder,
    chunks: list[ChunkIndexEntry],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[list[float]]]:
    """Embed text chunks using GTE-ModernColBERT.

    Args:
        embedder: TextEmbedder instance.
        chunks: List of ChunkIndexEntry objects.
        batch_size: Batch size for encoding.

    Returns:
        List of multi-vector embeddings (list of token vectors).
    """
    texts = [c.text for c in chunks]
    all_embeddings: list[list[list[float]]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embedder.encode_text_batch(batch, batch_size=len(batch))
        all_embeddings.extend(embeddings)

    return all_embeddings


# ---------------------------------------------------------------------------
# Qdrant upsert
# ---------------------------------------------------------------------------


def _build_point(
    chunk_entry: ChunkIndexEntry,
    token_embeddings: list[list[float]],
) -> dict:
    """Build a Qdrant point dict for the text chunk.

    Args:
        chunk_entry: The chunk metadata entry.
        token_embeddings: Multi-vector embeddings (list of token vectors).

    Returns:
        A dict ready for Qdrant upsert with vector + payload.
    """
    import numpy as np

    # Compute pooled vector as mean of token vectors
    token_array = np.array(token_embeddings, dtype=np.float32)
    pooled = token_array.mean(axis=0).tolist()

    return {
        "id": chunk_entry.chunk_id,
        "vector": {
            "text": token_embeddings,  # multivector with MAX_SIM comparator
            "pooled": pooled,
        },
        "payload": {
            "doc_id": chunk_entry.doc_id,
            "page_id": chunk_entry.page_id,
            "section_id": chunk_entry.section_id,
            "chunk_text": chunk_entry.text[:500],  # Store preview only
            "token_count": chunk_entry.token_count,
            "chunk_index_in_section": chunk_entry.chunk_index_in_section,
        },
    }


def _upsert_batch(
    client: QdrantClient,
    points: list[dict],
    collection_name: str = COLLECTION_PAGES_TEXT,
) -> None:
    """Upsert a batch of points to Qdrant.

    Args:
        client: QdrantClient instance.
        points: List of point dicts from _build_point.
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
# Main text indexing job
# ---------------------------------------------------------------------------


def run_text_index(
    docs_dir: Path | None = None,
    manifest_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rebuild: bool = False,
) -> dict:
    """Run the full text indexing job for the corpus.

    Pipeline:
        1. Discover all text chunks from docs/<hash>/parsed.json
        2. Embed chunks via GTE-ModernColBERT (batch processed)
        3. Upsert to Qdrant pages_text collection

    Args:
        docs_dir: Override for data/docs/ directory.
        manifest_path: Override for manifest.parquet path.
        batch_size: Number of chunks per embed batch (default: 64).
        rebuild: If True, re-index all chunks from scratch.

    Returns:
        Dict with keys: total_chunks, indexed_chunks, batches, elapsed_seconds, errors

    Raises:
        IndexingError: If a fatal error occurs during indexing.
    """
    settings = get_settings()

    docs_dir = docs_dir or Path(settings.docs_dir)
    manifest_path = manifest_path or Path(settings.manifest_path)

    logger.info(
        "text_index_starting",
        docs_dir=str(docs_dir),
        manifest=str(manifest_path),
        batch_size=batch_size,
    )

    # Step 1: Discover chunks
    try:
        chunks = _discover_chunks(docs_dir, manifest_path)
    except ValidationError:
        raise
    except Exception as e:
        raise IndexingError(f"Chunk discovery failed: {e}") from e

    if not chunks:
        raise IndexingError("No text chunks found — check docs/ directory and parsed.json files")

    # Step 2: Initialize embedder
    try:
        embedder = get_text_embedder()
    except Exception as e:
        raise IndexingError(f"Failed to initialize text embedder: {e}") from e

    # Step 3: Initialize Qdrant
    try:
        client = get_qdrant_client()
    except Exception as e:
        raise IndexingError(f"Failed to connect to Qdrant: {e}") from e

    # Step 4: Ensure collection exists
    from urban_rag.index.qdrant_client import create_collections
    try:
        create_collections(client)
    except ServiceUnavailableError:
        raise
    except Exception as e:
        raise IndexingError(f"Collection bootstrap failed: {e}") from e

    # Step 5: Batch processing
    total_chunks = len(chunks)
    indexed_chunks = 0
    error_count = 0
    errors: list[str] = []
    batches = 0
    start_time = time.perf_counter()

    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_chunks = chunks[batch_start:batch_end]
        batches += 1

        batch_desc = f"batch {batches} ({batch_start + 1}-{batch_end}/{total_chunks})"
        logger.info("processing_batch", batch=batch_desc)

        # Embed batch
        try:
            token_embeddings = _embed_chunks(embedder, batch_chunks, batch_size)
        except Exception as e:
            error_count += 1
            errors.append(f"Batch {batches} embed failed: {e}")
            logger.error("batch_embed_failed", batch=batch_desc, error=str(e))
            continue

        # Build points
        points = []
        for i, chunk_entry in enumerate(batch_chunks):
            points.append(_build_point(chunk_entry, token_embeddings[i]))

        # Upsert to Qdrant
        try:
            _upsert_batch(client, points)
        except Exception as e:
            error_count += 1
            errors.append(f"Batch {batches} upsert failed: {e}")
            logger.error("batch_upsert_failed", batch=batch_desc, error=str(e))
            continue

        indexed_chunks += len(points)
        logger.info(
            "batch_complete",
            batch=batch_desc,
            points=len(points),
            indexed_total=indexed_chunks,
        )

    elapsed = time.perf_counter() - start_time

    # Step 6: Report
    result = {
        "total_chunks": total_chunks,
        "indexed_chunks": indexed_chunks,
        "batches": batches,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors,
        "error_count": error_count,
    }

    logger.info(
        "text_index_complete",
        indexed_chunks=indexed_chunks,
        total_chunks=total_chunks,
        batches=batches,
        elapsed_seconds=result["elapsed_seconds"],
        error_count=error_count,
    )

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for text indexing."""
    import typer

    app = typer.Typer(
        name="text-index",
        help="Index text chunks from parsed documents to Qdrant pages_text collection",
        no_args_is_help=True,
    )

    @app.command()
    def run(
        docs_dir: str = "data/docs",
        manifest: str = "data/manifest.parquet",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Run text index on the corpus."""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        try:
            result = run_text_index(
                docs_dir=Path(docs_dir),
                manifest_path=Path(manifest),
                batch_size=batch_size,
            )
        except (ValidationError, IndexingError, ServiceUnavailableError) as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from e

        # Print summary table
        table = Table(title="Text Index Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total chunks", str(result["total_chunks"]))
        table.add_row("Indexed chunks", str(result["indexed_chunks"]))
        table.add_row("Batches", str(result["batches"]))
        table.add_row("Elapsed", f"{result['elapsed_seconds']}s")
        if result["error_count"] > 0:
            table.add_row("[red]Errors[/red]", str(result["error_count"]))
        else:
            table.add_row("Errors", "0")

        console.print(table)

        if result["errors"]:
            console.print("\n[bold red]Errors:[/bold red]")
            for err in result["errors"]:
                console.print(f"  [red]-[/red] {err}")
            raise typer.Exit(code=1) from None

        console.print("[green]✓[/green] Text index complete")

    @app.command()
    def check() -> None:
        """Check pages_text collection status."""
        from rich.console import Console

        console = Console()
        client = get_qdrant_client()
        info = client.get_collection(collection_name=COLLECTION_PAGES_TEXT)

        console.print(f"[cyan]Collection:[/cyan] {COLLECTION_PAGES_TEXT}")
        console.print(f"[cyan]Points:[/cyan] {info.points_count}")
        console.print(f"[cyan]Status:[/cyan] {info.status}")

    app()
