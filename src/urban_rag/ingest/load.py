"""Stage 1 — Validate & hash: PDF integrity, SHA256 content key, manifest write.

Idempotent: re-ingesting the same file bytes produces no duplicate manifest entry.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from urban_rag.common.errors import DocumentNotFoundError, ValidationError
from urban_rag.common.logging import get_logger
from urban_rag.common.settings import get_settings
from urban_rag.common.types import DocumentRecord

log = get_logger(__name__, service="ingest")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCS_DIR: Path | None = None  # Lazily resolved from settings


def _get_docs_dir() -> Path:
    """Return the docs directory, resolving from settings on first call."""
    global DOCS_DIR
    if DOCS_DIR is None:
        DOCS_DIR = Path(get_settings().docs_dir)
    return DOCS_DIR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MIN_PDF_SIZE_BYTES = 1024  # 1 KB — PDFs smaller than this are rejected


def validate_and_hash(pdf_path: Path) -> DocumentRecord:
    """Validate a PDF file, compute its SHA256 content hash, and store it.

    This is the first stage of the ingest pipeline (PLAN.md §5.2).

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A DocumentRecord for the ingested document.

    Raises:
        ValidationError: If the file is not a PDF, is smaller than 1 KB,
            or cannot be read.

    Idempotency:
        If the file has already been ingested (same SHA256), this function
        returns the existing DocumentRecord without creating duplicate files
        or manifest entries.
    """
    # ── 1. Path must exist ──────────────────────────────────────────────────
    if not pdf_path.exists():
        raise ValidationError(f"File not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValidationError(f"Not a file: {pdf_path}")

    # ── 2. Read bytes once ───────────────────────────────────────────────────
    try:
        file_bytes = pdf_path.read_bytes()
    except OSError as e:
        raise ValidationError(f"Cannot read file {pdf_path}: {e}") from e

    # ── 3. Verify PDF magic bytes FIRST ─────────────────────────────────────
    if not file_bytes.startswith(b"%PDF"):
        raise ValidationError(
            f"Not a PDF (missing %PDF header): {pdf_path}"
        )

    # ── 4. Reject files smaller than 1 KB ──────────────────────────────────
    if len(file_bytes) < MIN_PDF_SIZE_BYTES:
        raise ValidationError(
            f"PDF too small ({len(file_bytes)} bytes): {pdf_path}. "
            f"Minimum accepted size is {MIN_PDF_SIZE_BYTES} bytes."
        )

    # ── 5. SHA256 content hash ──────────────────────────────────────────────
    doc_hash = hashlib.sha256(file_bytes).hexdigest()
    log.info("computed_hash", doc_hash=doc_hash, size_bytes=len(file_bytes))

    # ── 6. Content-addressed destination ───────────────────────────────────
    docs_dir = _get_docs_dir()
    dest_dir = docs_dir / doc_hash
    dest_pdf = dest_dir / "source.pdf"

    # ── 7. Idempotency: already ingested → return existing record ──────────
    if dest_pdf.exists():
        log.info("already_ingested_noop", doc_hash=doc_hash, path=str(pdf_path))
        return _load_record(doc_hash)

    # ── 8. Write content-addressed file ─────────────────────────────────────
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_pdf.write_bytes(file_bytes)
    log.info("wrote_source_pdf", doc_hash=doc_hash, path=str(dest_pdf))

    # ── 9. Page count via PyMuPDF ───────────────────────────────────────────
    page_count = _count_pdf_pages(pdf_path)

    # ── 10. Build & persist DocumentRecord ──────────────────────────────────
    record = DocumentRecord(
        doc_hash=doc_hash,
        filename=pdf_path.name,
        page_count=page_count,
        size_bytes=len(file_bytes),
        ingested_at=datetime.now(tz=UTC),
        version="1",
    )

    _upsert_manifest_record(record)
    log.info("manifest_updated", doc_hash=doc_hash, filename=record.filename)

    return record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_pdf_pages(pdf_path: Path) -> int:
    """Count pages in a PDF using PyMuPDF.

    Returns 0 if PyMuPDF is unavailable (deferred to later stages).
    """
    try:
        import fitz  # type: ignore
    except ImportError:
        log.warning("pymupdf_not_available_page_count_deferred", path=str(pdf_path))
        return 0

    try:
        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except OSError as e:
        log.warning("page_count_failed", path=str(pdf_path), error=str(e))
        return 0


def _load_record(doc_hash: str) -> DocumentRecord:
    """Load a DocumentRecord from the manifest by hash."""
    settings = get_settings()
    manifest_path = Path(settings.manifest_path)

    if not manifest_path.exists():
        raise DocumentNotFoundError(f"No manifest entry for {doc_hash[:12]}...")

    df = pd.read_parquet(manifest_path)
    matches = df[df["doc_hash"] == doc_hash]
    if matches.empty:
        raise DocumentNotFoundError(f"No manifest entry for {doc_hash[:12]}...")

    return DocumentRecord.model_validate(matches.iloc[0].to_dict())


def _upsert_manifest_record(record: DocumentRecord) -> None:
    """Add or replace a DocumentRecord in the manifest parquet file."""
    settings = get_settings()
    manifest_path = Path(settings.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        df = pd.read_parquet(manifest_path)
        # Remove existing entry for this hash if present (upsert)
        df = df[df["doc_hash"] != record.doc_hash]
    else:
        df = pd.DataFrame()

    new_df = pd.DataFrame([record.model_dump()])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_parquet(manifest_path, index=False)
