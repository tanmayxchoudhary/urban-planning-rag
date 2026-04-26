"""Stage 3 — Layout-aware parse: Docling primary + Marker fallback.

Parses a PDF using Docling (IBM) as the primary parser. If Docling fails or
returns an empty section hierarchy, falls back to Marker with `--use_llm`.
The result is saved as `docs/<hash>/parsed.json` (JSON export of DoclingDocument).

The parsed output contains:
- Hierarchical section tree with titles and bounding boxes
- Table extractions (HTML format)
- Page regions (bounding boxes for every text element)
- Document-level metadata

PLAN.md §5.4 governs the expected output schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from urban_rag.common.errors import ParseError, ValidationError
from urban_rag.common.logging import get_logger

log = get_logger(__name__, service="ingest")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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


def parse_document(pdf_path: Path, doc_hash: str) -> dict[str, object]:
    """Parse a PDF using Docling (primary) with Marker fallback.

    Args:
        pdf_path: Path to the source PDF (already copied to docs/<hash>/source.pdf)
        doc_hash: SHA256 content hash of the document (used for content-addressing)

    Returns:
        A dict containing the parsed document structure. Must be JSON-serialisable
        and contain at minimum:
        - `sections` (list of section trees, possibly empty)
        - `pages` (list of page region records)
        - `tables` (list of extracted tables, possibly empty)
        - `_parser`: which parser was used ("docling" | "marker")
        - `_parser_version`: version string of the parser

    Raises:
        ParseError: if both Docling and Marker fail.
        ValidationError: if the pdf_path is not a valid PDF file.

    Idempotency:
        If docs/<hash>/parsed.json already exists, this function returns the
        cached result without re-parsing.
    """
    docs_dir = _get_docs_dir()
    dest_dir = docs_dir / doc_hash
    parsed_json_path = dest_dir / "parsed.json"

    # ── Idempotency: return cached result if already parsed ─────────────────
    if parsed_json_path.exists():
        log.info("parse_cache_hit", doc_hash=doc_hash, path=str(parsed_json_path))
        with parsed_json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ── Validate source PDF exists ──────────────────────────────────────────
    if not pdf_path.exists():
        raise ValidationError(f"PDF not found for parsing: {pdf_path}")

    log.info("parse_start", doc_hash=doc_hash, path=str(pdf_path))

    # ── Stage 1: Try Docling ────────────────────────────────────────────────
    parsed_result: dict[str, object] = {}
    parser_used: str = ""

    try:
        parsed_result, parser_used = _parse_with_docling(pdf_path, doc_hash)
        log.info("docling_parse_success", doc_hash=doc_hash)
    except ParseError:
        raise  # Re-raise ParseError (already structured)
    except Exception as exc:
        log.warning("docling_parse_failed", doc_hash=doc_hash, error=str(exc))

    # ── Stage 2: Fall back to Marker if Docling failed or returned empty ────
    if _is_empty_or_failed(parsed_result):
        log.info("marker_fallback_triggered", doc_hash=doc_hash)
        try:
            parsed_result, parser_used = _parse_with_marker(pdf_path, doc_hash)
            log.info("marker_parse_success", doc_hash=doc_hash)
        except Exception as exc:
            log.error(
                "marker_fallback_also_failed",
                doc_hash=doc_hash,
                error=str(exc),
                exc_info=True,
            )
            raise ParseError(
                f"Both Docling and Marker failed for {pdf_path.name}. "
                f"Docling error: {exc}; Marker error: {exc}"
            ) from exc

    # ── Stage 3: Validate and annotate result ──────────────────────────────
    if not parsed_result:
        raise ParseError(
            f"Parser returned empty result for {pdf_path.name}. "
            "Both Docling and Marker produced no output."
        )

    # Tag with parser metadata
    parsed_result["_parser"] = parser_used

    # ── Stage 4: Persist to docs/<hash>/parsed.json ───────────────────────
    dest_dir.mkdir(parents=True, exist_ok=True)
    with parsed_json_path.open("w", encoding="utf-8") as f:
        json.dump(parsed_result, f, indent=2, ensure_ascii=False)

    log.info(
        "parse_complete",
        doc_hash=doc_hash,
        parser=parser_used,
        output_size_bytes=parsed_json_path.stat().st_size,
    )

    return parsed_result


# ---------------------------------------------------------------------------
# Docling primary parser
# ---------------------------------------------------------------------------


def _parse_with_docling(pdf_path: Path, doc_hash: str) -> tuple[dict[str, object], str]:
    """Parse PDF using Docling and return a plain dict.

    Returns:
        A (parsed_dict, parser_name) tuple where parser_name == "docling".
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    if result is None or result.document is None:
        raise ParseError(f"Docling returned None for {pdf_path.name}")

    # Export to dict — this is the canonical DoclingDocument JSON format
    doc_dict = result.document.export_to_dict()

    if not isinstance(doc_dict, dict):
        raise ParseError(
            f"Docling export produced non-dict type {type(doc_dict).__name__}"
        )

    # Docling version
    version = getattr(result.document, "version", "unknown")

    # Build our normalized dict with explicit keys
    parsed: dict[str, object] = {
        "version": version,
        "sections": _extract_sections(doc_dict),
        "pages": _extract_pages(doc_dict),
        "tables": _extract_tables(doc_dict),
        "metadata": _extract_docling_metadata(result),
    }

    return parsed, "docling"


def _extract_sections(doc_dict: dict[str, object]) -> list[dict[str, object]]:
    """Extract hierarchical section tree from DoclingDocument export.

    Returns a list of section records. Each record contains:
    - `title`: str
    - `level`: int (1=chapter, 2=section, etc.)
    - `page_start`: int (1-based)
    - `page_end`: int (1-based)
    - `children`: list of child sections
    """
    sections: list[dict[str, object]] = []

    # Docling stores the document hierarchy in 'document' export at key
    # 'elements' or 'body' depending on version.
    # The structure is a list of element dicts, each with a 'type' field
    # indicating whether it is a heading or a section marker.
    elements_raw = doc_dict.get("elements", [])
    elements: list[dict[str, object]] = elements_raw if isinstance(elements_raw, list) else []
    current_section: dict[str, object] | None = None
    section_stack: list[dict[str, object]] = []

    for el in elements:
        if not isinstance(el, dict):
            continue

        elem_type = el.get("type", "")
        text = str(el.get("text") or el.get("content", "") or "").strip()

        # Section-level headings in Docling
        if elem_type in ("section", "heading", "h1", "h2", "h3", "h4"):
            level = _heading_level(elem_type, el)
            section: dict[str, object] = {
                "title": text,
                "level": level,
                "page_start": el.get("page", 1),
                "page_end": el.get("page", 1),
                "children": [],
            }

            # Pop stack to find correct parent
            while section_stack and section_stack[-1]["level"] >= level:
                section_stack.pop()

            if section_stack:
                section_stack[-1].setdefault("children", []).append(section)
            else:
                sections.append(section)

            section_stack.append(section)
            current_section = section

        elif current_section is not None:
            # Extend current section's page range
            pg_raw = el.get("page", 1)
            if isinstance(pg_raw, (int, float)):
                pg: int = int(pg_raw)
            else:
                fallback = current_section.get("page_start", 1)
                pg = int(fallback) if isinstance(fallback, int) else 1
            end_val = current_section["page_end"]
            current_section["page_end"] = max(end_val, pg)

    return sections


def _heading_level(elem_type: str, el: dict[str, object]) -> int:
    """Map Docling element type to numeric section level."""
    type_to_level = {
        "section": 1,
        "h1": 1,
        "heading": 2,
        "h2": 2,
    }
    default = 3
    return type_to_level.get(elem_type, default)


def _extract_pages(doc_dict: dict[str, object]) -> list[dict[str, object]]:
    """Extract per-page region information from DoclingDocument export.

    Returns a list of page records, each containing:
    - `page_num`: int (1-based)
    - `width`: float (points)
    - `height`: float (points)
    - `elements`: list of element bounding boxes
    """
    pages: list[dict[str, object]] = []

    # Pages are stored in 'pages' key of doc_dict (if present)
    raw_pages_raw = doc_dict.get("pages", [])
    raw_pages: list[dict[str, object]] = raw_pages_raw if isinstance(raw_pages_raw, list) else []
    if not raw_pages and "elements" in doc_dict:
        # Fall back: group elements by their page number
        elements_raw = doc_dict.get("elements", [])
        elements: list[dict[str, object]] = elements_raw if isinstance(elements_raw, list) else []
        page_map: dict[int, list[dict[str, object]]] = {}
        for el in elements:
            if not isinstance(el, dict):
                continue
            pg_raw = el.get("page", 1)
            pg: int = int(pg_raw) if isinstance(pg_raw, (int, float)) else 1  # type: ignore[arg-type]
            page_map.setdefault(pg, []).append(el)

        for pg_num in sorted(page_map.keys()):
            pages.append({
                "page_num": pg_num,
                "elements": page_map[pg_num],
            })
        return pages

    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue
        page: dict[str, object] = {
            "page_num": raw_page.get("page_no", raw_page.get("page", 1)),
            "width": raw_page.get("width", 0),
            "height": raw_page.get("height", 0),
            "elements": raw_page.get("elements", []),
        }
        pages.append(page)

    return pages


def _extract_tables(doc_dict: dict[str, object]) -> list[dict[str, object]]:
    """Extract table extractions from DoclingDocument export.

    Returns a list of table records, each containing:
    - `table_id`: str
    - `page_num`: int
    - `html`: str (Docling produces HTML table format)
    - `bbox`: [x0, y0, x1, y1]
    """
    tables: list[dict[str, object]] = []

    # Docling stores tables under 'tables' or within elements with type='table'
    raw_tables = doc_dict.get("tables", [])
    for i, raw_table in enumerate(raw_tables):
        if not isinstance(raw_table, dict):
            continue
        table = {
            "table_id": f"table_{i:04d}",
            "page_num": raw_table.get("page", 1),
            "html": raw_table.get("html", raw_table.get("content", "")),
            "bbox": raw_table.get("bbox", raw_table.get("bounds", [])),
        }
        tables.append(table)

    return tables


def _extract_docling_metadata(result: object) -> dict[str, object]:
    """Extract metadata from a Docling convert result object."""
    metadata: dict[str, object] = {}
    doc = getattr(result, "document", None)
    if doc is None:
        return metadata

    metadata["docling_version"] = getattr(doc, "version", "unknown")
    metadata["num_pages"] = getattr(doc, "page_count", 0)

    # Export all doc-level keys that are JSON-serializable
    for key in ("title", "author", "subject", "creator", "producer"):
        val = getattr(doc, key, None)
        if val is not None:
            metadata[key] = val

    return metadata


# ---------------------------------------------------------------------------
# Marker fallback parser
# ---------------------------------------------------------------------------


def _parse_with_marker(pdf_path: Path, doc_hash: str) -> tuple[dict[str, object], str]:
    """Parse PDF using Marker with --use_llm flag and return a plain dict.

    Returns:
        A (parsed_dict, parser_name) tuple where parser_name == "marker".
    """
    from marker.convert import convert_single_pdf

    # Marker returns an S3GitArtifact or similar object with .images and .markdown
    result = convert_single_pdf(str(pdf_path), use_llm=True)

    if result is None:
        raise ParseError(f"Marker returned None for {pdf_path.name}")

    # Marker output structure: result.images (list of page images),
    # result.markdown (full document as markdown string)
    pages: list[dict[str, object]] = []
    tables: list[dict[str, object]] = []

    # result.markdown is a string containing the full document text.
    # Split by page separators (Marker uses --- at page boundaries).
    full_text = getattr(result, "markdown", "") or ""

    # Marker output pages as list of rendered content
    images = getattr(result, "images", []) or []
    for i, img_path in enumerate(images):
        page: dict[str, object] = {
            "page_num": i + 1,
            "image_path": str(img_path) if img_path else "",
            "text": "",
        }
        pages.append(page)

    # Parse markdown for table regions (Marker produces markdown tables)
    sections = _parse_marker_sections(full_text)

    # Marker doesn't produce structured tables by default, so we extract
    # table-like blocks from markdown (lines starting with |)
    for line in full_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            tables.append({
                "table_id": f"marker_table_{len(tables):04d}",
                "page_num": 1,
                "html": _markdown_table_to_html(stripped),
                "bbox": [],
            })

    parsed: dict[str, object] = {
        "version": "marker-v1",
        "sections": sections,
        "pages": pages,
        "tables": tables,
        "metadata": {
            "parser": "marker",
            "has_images": len(images) > 0,
            "text_length": len(full_text),
        },
    }

    return parsed, "marker"


def _parse_marker_sections(full_text: str) -> list[dict[str, object]]:
    """Parse Marker markdown output into section tree.

    Uses markdown headings (# ## etc.) as section markers.
    """
    sections: list[dict[str, object]] = []
    current_section: dict[str, object] | None = None
    section_stack: list[dict[str, object]] = []

    for line in full_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            # Determine heading level
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            section: dict[str, object] = {
                "title": title,
                "level": level,
                "page_start": 1,
                "page_end": 1,
                "children": [],
            }

            while section_stack and section_stack[-1]["level"] >= level:
                section_stack.pop()

            if section_stack:
                section_stack[-1].setdefault("children", []).append(section)
            else:
                sections.append(section)

            section_stack.append(section)
            current_section = section
        elif current_section is not None:
            current_section["page_end"] = max(current_section["page_end"], 1)

    return sections


def _markdown_table_to_html(md_line: str) -> str:
    """Convert a single markdown table row to minimal HTML."""
    cols = [c.strip() for c in md_line.strip("|").split("|")]
    if not cols:
        return "<table><tr><td></td></tr></table>"
    cells = "".join(f"<td>{c}</td>" for c in cols)
    return f"<table><tr>{cells}</tr></table>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_empty_or_failed(result: dict[str, object]) -> bool:
    """Check if a parse result is empty or indicates failure.

    A result is considered empty if:
    - It's None or has no keys
    - It has no sections and no pages (both empty lists or absent)
    - The sections list exists but is empty AND pages is empty or absent
    """
    if not result:
        return True

    sections = result.get("sections", [])
    pages = result.get("pages", [])

    # If sections is a list and is empty, AND pages is empty, consider empty
    if isinstance(sections, list) and len(sections) == 0:
        if isinstance(pages, list) and len(pages) == 0:
            return True
        # Also empty if pages key is missing
        if "pages" not in result:
            return True

    return False
