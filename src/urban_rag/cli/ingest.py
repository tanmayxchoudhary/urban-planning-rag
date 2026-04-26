"""Ingest CLI commands."""

from __future__ import annotations

from pathlib import Path

from urban_rag.common.errors import IngestError, ValidationError
from urban_rag.common.logging import get_logger
from urban_rag.common.types import DocumentRecord
from urban_rag.ingest.chunk import chunk_document
from urban_rag.ingest.classify import classify_pages
from urban_rag.ingest.load import validate_and_hash
from urban_rag.ingest.parse import parse_document
from urban_rag.ingest.sections import detect_sections

log = get_logger(__name__, service="cli")


def rebuild_document(pdf_path: Path, doc_hash: str) -> None:
    """Remove cached stage outputs for a document so the next ingest rebuilds them.

    Called when ``--rebuild`` is passed to the CLI. Does NOT remove the source.pdf
    (the hash is stable regardless), but does remove:
    - parsed.json
    - pages.jsonl
    - chunks.jsonl
    - sections.jsonl

    Args:
        pdf_path: Path to the PDF file.
        doc_hash: SHA256 content hash of the document.
    """
    from urban_rag.common.settings import get_settings

    docs_dir = Path(get_settings().docs_dir)
    doc_dir = docs_dir / doc_hash

    if not doc_dir.exists():
        log.warning("rebuild_doc_dir_not_found", doc_hash=doc_hash[:12], path=str(doc_dir))
        return

    for artifact in ("parsed.json", "pages.jsonl", "chunks.jsonl", "sections.jsonl"):
        artifact_path = doc_dir / artifact
        if artifact_path.exists():
            artifact_path.unlink()
            log.info("rebuild_removed_artifact", doc_hash=doc_hash[:12], artifact=artifact)

    log.info("rebuild_complete", doc_hash=doc_hash[:12])


def ingest_file(
    path: Path,
    rebuild: bool = False,
    skip_eval: bool = False,
) -> DocumentRecord:
    """Ingest a single PDF file into the corpus, running the full ingest pipeline.

    Pipeline stages (in order):
        1. validate_and_hash  — validates PDF, computes SHA256, stores source.pdf
        2. parse_document      — Docling primary + Marker fallback → parsed.json
        3. classify_pages      — adaptive DPI classification → pages.jsonl
        4. chunk_document      — hierarchical text chunking → chunks.jsonl
        5. detect_sections     — TOC + regex + LLM fallback → sections.jsonl
        6. [eval check]        — skipped when skip_eval=True

    Args:
        path: Path to the PDF file to ingest.
        rebuild: If True, remove cached outputs before re-running each stage.
        skip_eval: If True, skip the eval-set validation stage.

    Returns:
        DocumentRecord for the ingested document.

    Raises:
        ValidationError: If the file does not exist, is not a valid PDF,
            or fails validation.
        IngestError: If any pipeline stage fails.
    """
    log.info("ingest_file_started", path=str(path), rebuild=rebuild, skip_eval=skip_eval)

    # ── Stage 1: Validate & hash ─────────────────────────────────────────
    # validate_and_hash handles:
    # - Path existence check (raises ValidationError if not found)
    # - PDF magic bytes verification
    # - Minimum file size check (≥1 KB)
    # - SHA256 content hash computation
    # - Content-addressed storage (source.pdf)
    record = validate_and_hash(path)
    doc_hash = record.doc_hash

    log.info("stage_completed", doc_hash=doc_hash[:12], stage="validate_and_hash")

    # ── Stage 6: Rebuild — remove cached outputs before re-running ───────
    if rebuild:
        rebuild_document(path, doc_hash)

    # ── Stage 2: Parse (Docling primary, Marker fallback) ────────────────
    source_pdf = Path(get_docs_dir()) / doc_hash / "source.pdf"
    if not source_pdf.exists():
        source_pdf = path  # Fall back to original path if not yet copied

    try:
        parse_document(source_pdf, doc_hash)
        log.info("stage_completed", doc_hash=doc_hash[:12], stage="parse_document")
    except Exception as exc:
        log.error("stage_failed", doc_hash=doc_hash[:12], stage="parse_document", error=str(exc))
        raise IngestError(f"Parse stage failed for {path.name}: {exc}") from exc

    # ── Stage 3: Classify pages (adaptive DPI) ────────────────────────────
    try:
        classify_pages(doc_hash)
        log.info("stage_completed", doc_hash=doc_hash[:12], stage="classify_pages")
    except Exception as exc:
        log.error(
            "stage_failed", doc_hash=doc_hash[:12], stage="classify_pages", error=str(exc)
        )
        raise IngestError(f"Classify stage failed for {path.name}: {exc}") from exc

    # ── Stage 4: Chunk document (hierarchical text chunking) ─────────────
    try:
        chunk_document(doc_hash)
        log.info("stage_completed", doc_hash=doc_hash[:12], stage="chunk_document")
    except Exception as exc:
        log.error(
            "stage_failed", doc_hash=doc_hash[:12], stage="chunk_document", error=str(exc)
        )
        raise IngestError(f"Chunk stage failed for {path.name}: {exc}") from exc

    # ── Stage 5: Detect sections (TOC + regex + LLM fallback) ─────────────
    try:
        detect_sections(doc_hash)
        log.info("stage_completed", doc_hash=doc_hash[:12], stage="detect_sections")
    except Exception as exc:
        log.error(
            "stage_failed", doc_hash=doc_hash[:12], stage="detect_sections", error=str(exc)
        )
        raise IngestError(f"Section detection failed for {path.name}: {exc}") from exc

    # ── Stage 6: Eval-set validation (skipped when skip_eval=True) ─────────
    if not skip_eval:
        log.info("eval_check_skipped_due_to_skip_eval", doc_hash=doc_hash[:12])
        # TODO: implement eval-set validation when eval modules are available
        # _run_eval_validation(doc_hash)

    log.info(
        "ingest_file_completed",
        doc_hash=doc_hash[:12],
        filename=record.filename,
        page_count=record.page_count,
    )

    return record


def get_docs_dir() -> Path:
    """Return the docs directory from settings."""
    from urban_rag.common.settings import get_settings

    return Path(get_settings().docs_dir)


def ingest_directory(
    directory: Path,
    rebuild: bool = False,
    skip_eval: bool = False,
) -> tuple[int, int]:
    """Ingest all PDF files in a directory.

    Args:
        directory: Path to the directory containing PDF files.
        rebuild: If True, force re-render and re-embed for each document.
        skip_eval: If True, skip eval-set validation for each document.

    Returns:
        A (exit_code, count) tuple where:
        - exit_code: 0 if all PDFs ingested successfully, 1 if any failed
        - count: number of PDFs successfully ingested
    """
    if not directory.exists() or not directory.is_dir():
        raise ValidationError(f"Not a directory: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        log.info("no_pdfs_found_in_directory", path=str(directory))
        return (0, 0)

    log.info("batch_ingest_started", path=str(directory), pdf_count=len(pdf_files))

    success_count = 0
    for pdf_path in pdf_files:
        try:
            ingest_file(pdf_path, rebuild=rebuild, skip_eval=skip_eval)
            success_count += 1
        except Exception as exc:
            log.warning(
                "batch_ingest_file_failed",
                path=str(pdf_path),
                error=str(exc),
            )

    exit_code = 0 if success_count == len(pdf_files) else 1
    log.info(
        "batch_ingest_completed",
        total=len(pdf_files),
        success=success_count,
        failed=len(pdf_files) - success_count,
    )

    return (exit_code, success_count)
