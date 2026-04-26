"""Combined text + sparse indexer for pages_text collection.

This module handles both GTE-ModernColBERT text embedding and BM25 sparse
vector computation in a single pass, upserting points with both vector
fields to the Qdrant pages_text collection.

Usage:
    python -m urban_rag.index.text_index run  # text + sparse together
    python -m urban_rag.index.sparse run      # sparse only (for re-indexing)
"""

from __future__ import annotations

import json
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
from urban_rag.embed.text_encoder import TextEmbedder, get_text_embedder
from urban_rag.index.qdrant_client import (
    COLLECTION_PAGES_TEXT,
    get_qdrant_client,
)

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__, service="index-text+sparse")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 64

# BM25 parameters (per PLAN §6.7)
BM25_K1: float = 1.5
BM25_B: float = 0.75


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
# Chunk discovery
# ---------------------------------------------------------------------------


def _discover_chunks(
    docs_dir: Path,
    manifest_path: Path,
) -> list[ChunkIndexEntry]:
    """Discover all text chunks from docs/<hash>/parsed.json and pages.jsonl."""
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

        parsed_path = docs_dir / doc_hash / "parsed.json"
        pages_jsonl_path = docs_dir / doc_hash / "pages.jsonl"

        if not parsed_path.exists():
            logger.warning(
                "no_parsed_json_skipping",
                doc_hash=doc_hash[:12],
                filename=filename,
            )
            continue

        page_metadata: dict[int, dict] = {}
        if pages_jsonl_path.exists():
            with pages_jsonl_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    page_rec = json.loads(line)
                    page_num = page_rec.get("page_num", 0)
                    page_metadata[page_num] = page_rec

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

        all_texts = _extract_docling_text(parsed_data)

        for page_num, texts_for_page in all_texts.items():
            page_rec = page_metadata.get(page_num, {})
            page_id = page_rec.get(
                "page_id", f"{doc_hash}#p{page_num:04d}"
            )
            section_title = page_rec.get("section_title")

            section_id = None
            if section_title:
                section_id = (
                    f"{doc_hash}#s{section_title[:30].replace(' ', '_')}"
                )

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
    """Extract text content from Docling parsed.json structure."""
    page_texts: dict[int, list[str]] = {}

    if "elements" in parsed_data:
        elements = parsed_data["elements"]
        if isinstance(elements, list):
            for el in elements:
                if not isinstance(el, dict):
                    continue
                text = el.get("text", "")
                if not text or not isinstance(text, str):
                    continue
                page_num = el.get("page", el.get("page_num", 1))
                if isinstance(page_num, str):
                    page_num = (
                        int(page_num) if page_num.isdigit() else 1
                    )
                page_texts.setdefault(int(page_num), []).append(text)

    elif "pages" in parsed_data:
        pages = parsed_data["pages"]
        if isinstance(pages, list):
            for page_idx, page_el in enumerate(pages, start=1):
                texts_for_page: list[str] = []
                if isinstance(page_el, dict):
                    texts_field = page_el.get(
                        "texts", page_el.get("lines", [])
                    )
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

    elif "text" in parsed_data:
        text_content = parsed_data["text"]
        if isinstance(text_content, str):
            lines = text_content.split("\n")
            current_page = 1
            for line in lines:
                if line.strip():
                    page_texts.setdefault(current_page, []).append(line)
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
    """Split page texts into overlapping chunks."""
    full_text = "\n".join(texts)

    if not full_text.strip():
        return []

    estimated_tokens = len(full_text) // 4

    if estimated_tokens <= target_tokens:
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

    entries: list[ChunkIndexEntry] = []
    chunk_idx = 0
    start = 0
    text_len = len(full_text)

    while start < text_len:
        end = start + (target_tokens * 4)
        if end >= text_len:
            end = text_len
        else:
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
# BM25 scorer
# ---------------------------------------------------------------------------


class BM25Scorer:
    """BM25 scorer for sparse vector computation."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.N = len(texts)
        self._term_freqs = _compute_term_frequencies(texts)

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

        self._idf: dict[str, float] = {}
        for term in self._vocab:
            df = self._doc_freqs.get(term, 0)
            if df > 0:
                self._idf[term] = math.log(
                    (self.N - df + 0.5) / (df + 0.5) + 1
                )
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
        """Compute BM25-weighted sparse vector for a document."""
        tokens = _tokenize(text)
        tf = Counter(tokens)
        doc_len = max(sum(tf.values()), 1)

        scores: dict[str, float] = {}
        for term, freq in tf.items():
            if term not in self._idf:
                continue
            idf = self._idf[term]
            numerator = freq * (BM25_K1 + 1)
            denominator = (
                freq
                + BM25_K1
                * (1 - BM25_B + BM25_B * doc_len / self._avgdl)
            )
            score = idf * numerator / denominator
            scores[term] = score

        sorted_terms = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        vocab_list = sorted(self._vocab)
        vocab_idx = {term: i for i, term in enumerate(vocab_list)}

        indices = [
            vocab_idx[term] for term, _ in sorted_terms if term in vocab_idx
        ]
        values = [score for _, score in sorted_terms]

        return indices, values


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


def _compute_term_frequencies(
    texts: list[str],
) -> list[dict[str, int]]:
    """Compute term frequencies for each document."""
    tfs: list[dict[str, int]] = []
    for text in texts:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        tfs.append(tf)
    return tfs


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _embed_chunks(
    embedder: TextEmbedder,
    chunks: list[ChunkIndexEntry],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[list[float]]]:
    """Embed text chunks using GTE-ModernColBERT."""
    texts = [c.text for c in chunks]
    all_embeddings: list[list[list[float]]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embedder.encode_text_batch(batch, batch_size=len(batch))
        all_embeddings.extend(embeddings)

    return all_embeddings


# ---------------------------------------------------------------------------
# Qdrant point building
# ---------------------------------------------------------------------------


def _build_point(
    chunk_entry: ChunkIndexEntry,
    token_embeddings: list[list[float]],
    sparse_indices: list[int],
    sparse_values: list[float],
) -> dict:
    """Build a Qdrant point with text multivector + sparse BM25 vector."""
    import numpy as np
    from qdrant_client import models

    token_array = np.array(token_embeddings, dtype=np.float32)
    pooled = token_array.mean(axis=0).tolist()

    return {
        "id": chunk_entry.chunk_id,
        "vector": {
            "text": token_embeddings,  # multivector with MAX_SIM
            "pooled": pooled,
            "sparse": models.SparseVector(
                indices=sparse_indices, values=sparse_values
            ),
        },
        "payload": {
            "doc_id": chunk_entry.doc_id,
            "page_id": chunk_entry.page_id,
            "section_id": chunk_entry.section_id,
            "chunk_text": chunk_entry.text[:500],
            "token_count": chunk_entry.token_count,
            "chunk_index_in_section": chunk_entry.chunk_index_in_section,
        },
    }


def _upsert_batch(
    client: QdrantClient,
    points: list[dict],
) -> None:
    """Upsert a batch of points to Qdrant."""
    from qdrant_client import models

    client.upsert(
        collection_name=COLLECTION_PAGES_TEXT,
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
# Main indexing job
# ---------------------------------------------------------------------------


def run_text_and_sparse_index(
    docs_dir: Path | None = None,
    manifest_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rebuild: bool = False,
) -> dict:
    """Run combined text + sparse indexing for the corpus.

    Pipeline:
        1. Discover all text chunks from docs/<hash>/parsed.json
        2. Compute BM25 scorer across corpus
        3. Embed chunks via GTE-ModernColBERT (batch processed)
        4. Compute BM25 sparse vectors per chunk
        5. Upsert combined points (text + sparse) to pages_text collection

    Args:
        docs_dir: Override for data/docs/ directory.
        manifest_path: Override for manifest.parquet path.
        batch_size: Number of chunks per embed batch.
        rebuild: If True, re-index all chunks from scratch.

    Returns:
        Dict with keys: total_chunks, indexed_chunks, batches,
                        elapsed_seconds, errors

    Raises:
        IndexingError: If a fatal error occurs.
    """
    settings = get_settings()

    docs_dir = docs_dir or Path(settings.docs_dir)
    manifest_path = manifest_path or Path(settings.manifest_path)

    logger.info(
        "text_sparse_index_starting",
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
        raise IndexingError(
            "No text chunks found — check docs/ directory and parsed.json files"
        )

    # Step 2: Initialize text embedder
    try:
        embedder = get_text_embedder()
    except Exception as e:
        raise IndexingError(f"Failed to initialize text embedder: {e}") from e

    # Step 3: Initialize BM25 scorer
    texts = [c.text for c in chunks]
    try:
        bm25_scorer = BM25Scorer(texts)
    except Exception as e:
        raise IndexingError(f"BM25 initialization failed: {e}") from e

    # Step 4: Initialize Qdrant
    try:
        client = get_qdrant_client()
    except Exception as e:
        raise IndexingError(f"Failed to connect to Qdrant: {e}") from e

    # Step 5: Bootstrap collection
    from urban_rag.index.qdrant_client import create_collections

    try:
        create_collections(client)
    except ServiceUnavailableError:
        raise
    except Exception as e:
        raise IndexingError(f"Collection bootstrap failed: {e}") from e

    # Step 6: Batch processing
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

        batch_desc = (
            f"batch {batches} ({batch_start + 1}-{batch_end}/{total_chunks})"
        )
        logger.info("processing_batch", batch=batch_desc)

        # Embed text
        try:
            token_embeddings = _embed_chunks(
                embedder, batch_chunks, batch_size
            )
        except Exception as e:
            error_count += 1
            errors.append(f"Batch {batches} embed failed: {e}")
            logger.error("batch_embed_failed", batch=batch_desc, error=str(e))
            continue

        # Build points with sparse vectors
        points = []
        batch_texts = texts[batch_start:batch_end]
        for i, chunk_entry in enumerate(batch_chunks):
            # Compute sparse vector for this chunk
            sparse_indices, sparse_values = bm25_scorer.compute_sparse_vector(
                batch_texts[i]
            )
            if not sparse_indices:
                sparse_indices = [0]
                sparse_values = [0.0]

            point = _build_point(
                chunk_entry,
                token_embeddings[i],
                sparse_indices,
                sparse_values,
            )
            points.append(point)

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

    result = {
        "total_chunks": total_chunks,
        "indexed_chunks": indexed_chunks,
        "batches": batches,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors,
        "error_count": error_count,
    }

    logger.info(
        "text_sparse_index_complete",
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
    """CLI entry point for combined text + sparse indexing."""
    import typer

    app = typer.Typer(
        name="text-sparse-index",
        help="Index text chunks with GTE-ModernColBERT + BM25 sparse vectors",
        no_args_is_help=True,
    )

    @app.command()
    def run(
        docs_dir: str = "data/docs",
        manifest: str = "data/manifest.parquet",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Run combined text + sparse index on the corpus."""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        try:
            result = run_text_and_sparse_index(
                docs_dir=Path(docs_dir),
                manifest_path=Path(manifest),
                batch_size=batch_size,
            )
        except (ValidationError, IndexingError, ServiceUnavailableError) as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from e

        table = Table(title="Text + Sparse Index Results")
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

        console.print("[green]✓[/green] Text + sparse index complete")

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
