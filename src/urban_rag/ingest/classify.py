"""Stage 4 — Adaptive DPI page classification.

Classifies each page of a parsed document as TEXT or VISUAL based on
visual content density, then records the classification in pages.jsonl.

  - VISUAL : has images  OR  num_drawings > 40   → 250 DPI
  - TEXT   : all other pages                   → 100 DPI

The page records are written as JSONL to docs/<hash>/pages.jsonl.

PART V §5.3 governs the classification rules.
"""

from __future__ import annotations

import json
from pathlib import Path

from urban_rag.common.logging import get_logger
from urban_rag.common.types import PageRecord

log = get_logger(__name__, service="ingest")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DPI_VISUAL = 250
DPI_TEXT = 100
DRAWING_THRESHOLD = 40

DOCS_DIR: Path | None = None  # Lazily resolved


def _get_docs_dir() -> Path:
    """Return docs directory from settings."""
    global DOCS_DIR
    if DOCS_DIR is None:
        from urban_rag.common.settings import get_settings

        DOCS_DIR = Path(get_settings().docs_dir)
    return DOCS_DIR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_pages(doc_hash: str) -> list[PageRecord]:
    """Classify pages of an already-parsed document and write pages.jsonl.

    Reads docs/<hash>/parsed.json (produced by parse_document), determines
    page_type (TEXT or VISUAL) for each page, and writes a JSONL file with
    one PageRecord per line.

    Classification rules (PART V §5.3):
      - VISUAL : has_images == True  OR  num_drawings > 40
      - TEXT   : all other pages

    Args:
        doc_hash: SHA256 content hash of the document.

    Returns:
        A list of PageRecord objects, one per page.

    Idempotency:
        If docs/<hash>/pages.jsonl already exists, it is returned directly
        without re-classifying.
    """
    docs_dir = _get_docs_dir()
    dest_dir = docs_dir / doc_hash
    parsed_json_path = dest_dir / "parsed.json"
    pages_jsonl_path = dest_dir / "pages.jsonl"

    # ── Idempotency: return cached result if already classified ─────────────
    if pages_jsonl_path.exists():
        log.info("classify_cache_hit", doc_hash=doc_hash)
        return _load_pages_jsonl(pages_jsonl_path)

    # ── Load parsed output ─────────────────────────────────────────────────
    if not parsed_json_path.exists():
        raise FileNotFoundError(
            f"parsed.json not found for {doc_hash[:12]}... "
            "(run parse_document before classify_pages)"
        )

    with parsed_json_path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    log.info("classify_start", doc_hash=doc_hash)

    # ── Classify each page ─────────────────────────────────────────────────
    page_records = _build_page_records(doc_hash, parsed)

    # ── Persist pages.jsonl ────────────────────────────────────────────────
    dest_dir.mkdir(parents=True, exist_ok=True)
    with pages_jsonl_path.open("w", encoding="utf-8") as f:
        for record in page_records:
            f.write(record.model_dump_json() + "\n")

    log.info(
        "classify_complete",
        doc_hash=doc_hash,
        page_count=len(page_records),
        visual_count=sum(1 for r in page_records if r.page_type == "VISUAL"),
        text_count=sum(1 for r in page_records if r.page_type == "TEXT"),
    )

    return page_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_page_records(doc_hash: str, parsed: dict) -> list[PageRecord]:
    """Build a list of PageRecord from a parsed document dict.

    Examines each page for visual content indicators and classifies:
      - VISUAL: has_images == True  OR  num_drawings > 40
      - TEXT:   all other pages
    """
    page_records: list[PageRecord] = []

    # parsed["pages"] is a list of page dicts from Docling/Marker
    raw_pages: list[dict] = parsed.get("pages", [])
    if not isinstance(raw_pages, list):
        raw_pages = []

    for raw_page in raw_pages:
        page_num_raw = raw_page.get("page_num", raw_page.get("page", 1))
        page_num: int = int(page_num_raw) if isinstance(page_num_raw, (int, float)) else 1
        if page_num < 1:
            page_num = 1

        # Detect visual content
        has_images = _page_has_images(raw_page)
        num_drawings = _page_num_drawings(raw_page)

        if has_images or num_drawings > DRAWING_THRESHOLD:
            page_type: str = "VISUAL"
            dpi_used: int = DPI_VISUAL
        else:
            page_type = "TEXT"
            dpi_used = DPI_TEXT

        page_id = f"{doc_hash}#p{page_num:04d}"

        record = PageRecord(
            page_id=page_id,
            doc_id=doc_hash,
            page_num=page_num,
            page_type=page_type,  # type: ignore[arg-type]
            dpi_used=dpi_used,
            image_uri="",  # Set by render stage
            extracted_text="",
            layout=[],
            section_id=None,
            section_title=None,
        )
        page_records.append(record)

    return page_records


def _page_has_images(page: dict) -> bool:
    """Return True if the page has embedded images.

    In Docling export, images are indicated by elements with type "image" or
    "figure", or by a top-level "images" key.
    """
    # Check for explicit images list
    images = page.get("images", [])
    if isinstance(images, list) and len(images) > 0:
        return True

    # Check elements for image/figure blocks
    elements = page.get("elements", [])
    if isinstance(elements, list):
        for el in elements:
            if not isinstance(el, dict):
                continue
            elem_type = str(el.get("type", "")).lower()
            if elem_type in ("image", "figure"):
                return True

    return False


def _page_num_drawings(page: dict) -> int:
    """Count drawing elements on the page.

    Drawings include paths, lines, rectangles, and other vector graphics
    elements in the Docling page structure.
    """
    count = 0
    elements = page.get("elements", [])
    if not isinstance(elements, list):
        return 0

    for el in elements:
        if not isinstance(el, dict):
            continue
        elem_type = str(el.get("type", "")).lower()
        # Docling uses "line", "rect", "path", "polygon", "drawing" for vector art
        if elem_type in ("line", "rect", "path", "polygon", "drawing"):
            count += 1
        # Also count by checking 'drawings' or 'paths' keys
        drawings = el.get("drawings", [])
        paths = el.get("paths", [])
        if isinstance(drawings, list):
            count += len(drawings)
        if isinstance(paths, list):
            count += len(paths)

    return count


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
