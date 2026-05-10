"""Stage 5 — Render PDF pages to PNG at adaptive DPI.

Each page of a parsed+classified document is rendered to a PNG file at the
appropriate DPI:
  - TEXT pages  → 100 DPI  (smaller file, faster processing)
  - VISUAL pages → 250 DPI (higher fidelity for diagrams/tables/figures)

Rendering is driven by the per-page DPI values produced by classify.py and
stored in pages.jsonl. Output goes to data/page_images/{sanitized_filename}__page_{pagenum:04d}.png.

The updated pages.jsonl is written back with the image_uri field populated
for each page, enabling the batch indexer to discover rendered PNGs.

PART V §5.4 governs the render stage.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # type: ignore[import]

from urban_rag.common.errors import RenderError
from urban_rag.common.logging import get_logger
from urban_rag.common.settings import get_settings
from urban_rag.common.types import PageRecord

log = get_logger(__name__, service="ingest")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DPI_TEXT = 100
DPI_VISUAL = 250

DOCS_DIR: Path | None = None  # Lazily resolved from settings
PAGE_IMAGES_DIR: Path | None = None  # Lazily resolved from settings


def _get_docs_dir() -> Path:
    """Return docs directory from settings."""
    global DOCS_DIR
    if DOCS_DIR is None:
        DOCS_DIR = Path(get_settings().docs_dir)
    return DOCS_DIR


def _get_page_images_dir() -> Path:
    """Return page images directory from settings."""
    global PAGE_IMAGES_DIR
    if PAGE_IMAGES_DIR is None:
        PAGE_IMAGES_DIR = Path(get_settings().page_images_dir)
    return PAGE_IMAGES_DIR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_document(doc_hash: str) -> list[PageRecord]:
    """Render all pages of a document to PNG at adaptive DPI.

    Reads docs/<hash>/pages.jsonl (produced by classify.py) to get per-page
    DPI and page number, opens docs/<hash>/source.pdf, and renders each page
    as a PNG in data/page_images/.

    Args:
        doc_hash: SHA256 content hash of the document.

    Returns:
        A list of PageRecord objects with image_uri populated.

    Raises:
        FileNotFoundError: If source.pdf or pages.jsonl is missing.

    Idempotency:
        Re-rendering is safe — existing PNGs are overwritten and the
        image_uri field in pages.jsonl is updated to point to the correct path.
    """
    docs_dir = _get_docs_dir()
    dest_dir = docs_dir / doc_hash
    source_pdf_path = dest_dir / "source.pdf"
    pages_jsonl_path = dest_dir / "pages.jsonl"

    if not source_pdf_path.exists():
        raise FileNotFoundError(
            f"source.pdf not found for {doc_hash[:12]}... "
            "(run validate_and_hash before render_document)"
        )

    if not pages_jsonl_path.exists():
        raise FileNotFoundError(
            f"pages.jsonl not found for {doc_hash[:12]}... "
            "(run classify_pages before render_document)"
        )

    # Load page records
    page_records = _load_pages_jsonl(pages_jsonl_path)

    log.info("render_start", doc_hash=doc_hash, page_count=len(page_records))

    # Open PDF once
    try:
        pdf_doc = fitz.open(str(source_pdf_path))
    except Exception as e:
        raise RenderError(f"Failed to open PDF {source_pdf_path}: {e}") from e

    try:
        updated_records: list[PageRecord] = []

        for record in page_records:
            page_num = record.page_num

            # PyMuPDF is 0-indexed
            try:
                page = pdf_doc[page_num - 1]
            except IndexError:
                log.warning(
                    "render_page_out_of_range",
                    doc_hash=doc_hash,
                    page_num=page_num,
                    max_pages=pdf_doc.page_count,
                )
                raise

            dpi = record.dpi_used
            base_name = _sanitize_basename(record.doc_id)

            # Render to PNG at specified DPI
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

            # Build output path
            page_images_dir = _get_page_images_dir()
            page_images_dir.mkdir(parents=True, exist_ok=True)

            png_filename = f"{base_name}__page_{page_num:04d}.png"
            png_path = page_images_dir / png_filename

            # Save PNG
            pix.save(str(png_path))
            log.debug(
                "rendered_page",
                doc_hash=doc_hash,
                page_num=page_num,
                dpi=dpi,
                output=str(png_path),
                size_bytes=png_path.stat().st_size,
            )

            # Update record with image_uri (relative to project root)
            record.image_uri = str(png_path)
            updated_records.append(record)

    finally:
        pdf_doc.close()

    # Write updated pages.jsonl with image_uri populated
    with pages_jsonl_path.open("w", encoding="utf-8") as f:
        for record in updated_records:
            f.write(record.model_dump_json() + "\n")

    log.info(
        "render_complete",
        doc_hash=doc_hash,
        page_count=len(updated_records),
        page_images_dir=str(_get_page_images_dir()),
    )

    return updated_records


# ---------------------------------------------------------------------------
# Low-level helpers (also exported for direct use)
# ---------------------------------------------------------------------------


def _render_page(
    pdf_path: Path,
    page_num: int,
    dpi: int,
    output_dir: Path,
    base_name: str,
) -> Path:
    """Render a single PDF page to PNG at the given DPI.

    Args:
        pdf_path: Path to the PDF file.
        page_num: 1-based page number.
        dpi: Target DPI for rendering (100 or 250).
        output_dir: Directory to write the PNG into.
        base_name: Base name for the output file (no extension).

    Returns:
        Path to the rendered PNG file.

    Raises:
        IndexError: If page_num is beyond the PDF's page count.
    """
    pdf_doc = fitz.open(str(pdf_path))
    try:
        page = pdf_doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    finally:
        pdf_doc.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_filename = f"{base_name}__page_{page_num:04d}.png"
    png_path = output_dir / png_filename
    pix.save(str(png_path))
    return png_path


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename into a safe base name.

    Removes extension, replaces spaces/special chars with underscores,
    collapses multiple underscores, and strips leading/trailing underscores.
    """
    # Remove extension
    name = filename
    if "." in name:
        name = name.rsplit(".", 1)[0]

    # Replace any non-alphanumeric char (except underscore and hyphen) with underscore
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    # Replace hyphens with underscores too
    name = name.replace("-", "_")

    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)

    # Strip leading/trailing underscores
    name = name.strip("_")

    if not name:
        name = "source"

    return name


def _sanitize_basename(doc_hash: str) -> str:
    """Create a safe base name from a doc_hash (which is already a hex hash).

    The doc_hash is a SHA256 hex string — it contains only [a-f0-9].
    We use it directly as the base name to avoid any filename collision risk.
    """
    return doc_hash


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_pages_jsonl(path: Path) -> list[PageRecord]:
    """Load PageRecord list from a pages.jsonl file."""
    records: list[PageRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(PageRecord.model_validate_json(line))
    return records
