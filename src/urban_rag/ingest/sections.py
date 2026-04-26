"""Stage 6 — Section-title detection.

Detects section titles for each page using a three-tier strategy:

  1. TOC parsing   : walk the section hierarchy already built by parse.py from
                    Docling/Marker structured output
  2. Regex fallback: page-number-aware regex patterns for chapter/section headings
                    (handles documents that lack a machine-readable TOC)
  3. LLM fallback : send the page text to Gemini with a structured prompt asking
                    for the section title — only used when both above return None

Output: list of (page_number, section_title, section_id) triples per page,
         written to docs/<hash>/sections.jsonl

PLAN.md §5.5 governs section-title detection; it is the primary source of
section metadata for PageRecord.section_title before embedding.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

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
# Public data types
# ---------------------------------------------------------------------------


class SectionTriple(NamedTuple):
    """A single page section triple."""

    page_number: int
    section_title: str
    section_id: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_sections(doc_hash: str) -> list[SectionTriple]:
    """Detect section titles for every page of an already-parsed document.

    Detection strategy (three-tier):
      1. TOC parsing   — walk the Docling/Marker section hierarchy from
                         parsed.json to find the best matching section for
                         each page (section whose page_start ≤ page ≤ page_end)
      2. Regex fallback — if TOC returns no title for a page, apply
                         chapter-heading regex patterns on the page text
                         (handles unstructured PDFs)
      3. LLM fallback   — if regex also fails, call Gemini with a structured
                         prompt asking for the section title from page content

    Args:
        doc_hash: SHA256 content hash of the document.

    Returns:
        A list of SectionTriple, one per page (in ascending page_number order).
        Pages with no detected title receive ("Page N", "").

    Idempotency:
        If docs/<hash>/sections.jsonl already exists, it is returned directly
        without re-detecting.
    """
    docs_dir = _get_docs_dir()
    dest_dir = docs_dir / doc_hash
    parsed_json_path = dest_dir / "parsed.json"
    sections_jsonl_path = dest_dir / "sections.jsonl"

    # ── Idempotency ───────────────────────────────────────────────────────
    if sections_jsonl_path.exists():
        log.info("sections_cache_hit", doc_hash=doc_hash)
        return _load_sections_jsonl(sections_jsonl_path)

    # ── Load parsed output ─────────────────────────────────────────────────
    if not parsed_json_path.exists():
        raise FileNotFoundError(
            f"parsed.json not found for {doc_hash[:12]}... "
            "(run parse_document before detect_sections)"
        )

    with parsed_json_path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    log.info("sections_detect_start", doc_hash=doc_hash)

    sections_data: list[dict] = parsed.get("sections", [])
    if not isinstance(sections_data, list):
        sections_data = []

    raw_pages: list[dict] = parsed.get("pages", [])
    if not isinstance(raw_pages, list):
        raw_pages = []

    # Collect page numbers present in the document
    page_numbers: list[int] = sorted(
        {int(p.get("page_num", p.get("page", 1))) for p in raw_pages}
    )

    if not page_numbers:
        # Fall back to a single blank page
        page_numbers = [1]

    # ── Tier 1: TOC — walk section hierarchy ────────────────────────────────
    flat_sections = _flatten_sections(sections_data)
    toc_map = _build_toc_map(flat_sections)  # page_number → (section_id, title)

    # ── Tier 2: Regex patterns for pages not covered by TOC ────────────────
    page_text_map = _extract_page_text_map(raw_pages)
    regex_triples = _detect_via_regex(page_numbers, page_text_map)

    # Merge: prefer TOC result, fall back to regex result
    result_triples: list[SectionTriple] = []
    needs_llm: list[int] = []

    for pg in page_numbers:
        if pg in toc_map:
            sec_id, title = toc_map[pg]
            result_triples.append(SectionTriple(pg, title, sec_id))
        elif pg in regex_triples:
            result_triples.append(regex_triples[pg])
        else:
            needs_llm.append(pg)
            result_triples.append(
                SectionTriple(pg, _default_title(pg), _default_section_id(pg))
            )

    # ── Tier 3: LLM fallback for still-unresolved pages ──────────────────
    if needs_llm:
        log.info("sections_llm_fallback", doc_hash=doc_hash, page_count=len(needs_llm))
        llm_results = _detect_via_llm(doc_hash, needs_llm, page_text_map)
        for i, pg in enumerate(needs_llm):
            llm_title = llm_results[i]
            if llm_title and llm_title != "[NONE]":
                # Replace the placeholder with the LLM result
                for j, t in enumerate(result_triples):
                    if t.page_number == pg:
                        result_triples[j] = SectionTriple(
                            pg, llm_title, _llm_section_id(llm_title)
                        )
                        break

    # ── Persist sections.jsonl ────────────────────────────────────────────
    dest_dir.mkdir(parents=True, exist_ok=True)
    with sections_jsonl_path.open("w", encoding="utf-8") as f:
        for triple in result_triples:
            f.write(
                json.dumps(
                    {
                        "page_number": triple.page_number,
                        "section_title": triple.section_title,
                        "section_id": triple.section_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    log.info(
        "sections_detect_complete",
        doc_hash=doc_hash,
        page_count=len(result_triples),
    )

    return result_triples


# ---------------------------------------------------------------------------
# Tier 1 — TOC hierarchy walking
# ---------------------------------------------------------------------------


def _flatten_sections(
    sections: list[dict], prefix: list[str] | None = None
) -> list[dict]:
    """Flatten a section tree into a flat list with title_path annotated."""
    flat: list[dict] = []
    if prefix is None:
        prefix = []

    for sec in sections:
        if not isinstance(sec, dict):
            continue
        title = str(sec.get("title", "") or "").strip()
        current_path = [*prefix, title] if title else prefix

        s = dict(sec)
        s.setdefault("title", title)  # ensure key exists even if missing from source
        if "section_id" not in s:
            s["section_id"] = f"s{len(flat):03d}"
        s["title_path"] = current_path

        flat.append(s)

        children: list[dict] = sec.get("children", [])
        if isinstance(children, list) and children:
            child_flat = _flatten_sections(children, current_path)
            flat.extend(child_flat)

    return flat


def _build_toc_map(
    flat_sections: list[dict],
) -> dict[int, tuple[str, str]]:
    """Map each page number to its deepest-matching (section_id, title).

    Uses the page_start/page_end range of each section.  When a page falls
    within multiple sections, the one with the highest level (deepest) wins.
    """
    page_to_best: dict[int, tuple[int, str, str]] = {}

    for sec in flat_sections:
        level = sec.get("level", 99)
        sec_id = sec.get("section_id", "")
        title = str(sec.get("title", "")).strip()
        if not title:
            title = f"Section {sec_id}"

        start = int(sec.get("page_start", 1))
        end = int(sec.get("page_end", start))

        for pg in range(start, end + 1):
            existing = page_to_best.get(pg)
            existing_level = existing[0] if existing else -1
            # Store (level, section_id, title) — use level as sort key
            if existing is None or level > existing_level:
                page_to_best[pg] = (level, sec_id, title)

    # Strip the level from the stored tuple
    return {pg: (sec_id, title) for pg, (_, sec_id, title) in page_to_best.items()}


# ---------------------------------------------------------------------------
# Tier 2 — Regex fallback
# ---------------------------------------------------------------------------

# Regex patterns for common Indian planning document chapter / section headings.
# These are applied to the raw page text when the TOC hierarchy yields nothing.
#
# Patterns cover:
#   - "Chapter N" / "Chapter N: Title" / "Chapter N - Title"
#   - "Part N", "Section N", "Regulation N", "Rule N"
#   - "Annexure / Appendix / Schedule N"
#   - URDPFI-specific patterns like "5.3", "Part V §5" etc.
#   - Numbered patterns like "4.2.3" at the start of a line
#   - ALL CAPS headings (common in legal documents)
#
# Each pattern MUST capture the title in group 3 (or group 2 for simple number patterns).
# Group 1 is always the primary identifier (chapter/part number or section number).
# DO NOT use group 2 for the title extraction if it contains only a numeric section identifier.

_CHAPTER_PATTERNS = [
    # "Chapter 5: Zoning Regulations" — title in group 3
    re.compile(
        r"^chapter\s+(\d+)\s*[:\-:]\s*(.+)$",
        re.IGNORECASE,
    ),
    # "Chapter 5 Zoning Regulations" (bare number, space, title)
    re.compile(
        r"^chapter\s+(\d+)\s{2,}([^\n]+)$",
        re.IGNORECASE,
    ),
    # "Part V §5.3: Building Height" — title in group 3
    re.compile(
        r"^part\s+([IVXLCDM\d]+)\s+§\s*(\d+(?:\.\d+)?)\s*[:\-]?\s*(.+)$",
        re.IGNORECASE,
    ),
    # "Part V  Fire Safety Requirements" — bare part without section number, title in group 2
    re.compile(
        r"^part\s+([IVXLCDM\d]+)\s{2,}([^\n]+)$",
        re.IGNORECASE,
    ),
    # "Part 5  Fire Safety Requirements" — bare part with arabic numeral
    re.compile(
        r"^part\s+(\d+)\s{2,}([^\n]+)$",
        re.IGNORECASE,
    ),
    # "4.2.3  Floor Space Index" — number with double-space, title in group 2
    re.compile(
        r"^(\d+(?:\.\d+)+)\s{2,}([^\n]+)$",
    ),
    # "4.2.3 Floor Space Index" — single-space, title in group 2
    re.compile(
        r"^(\d+(?:\.\d)+)\s+([^\n]+)$",
    ),
    # ALL CAPS heading (common in legal prefaces / table of contents)
    re.compile(
        r"^([A-Z][A-Z\s]{3,})$",
    ),
]

# Patterns that strongly indicate a TOC/Index page and should be skipped
_TOC_SKIP_PATTERNS = [
    re.compile(r"^(?:table\s+of\s+contents|contents|index|table\s+of\s+contents)$", re.IGNORECASE),
    re.compile(r"^(?:page\s+\d+|pp?\.\s*\d+)", re.IGNORECASE),
]


def _detect_via_regex(
    page_numbers: list[int],
    page_text_map: dict[int, str],
) -> dict[int, SectionTriple]:
    """Apply heading regex to each page text, returning resolved triples.

    Returns a dict mapping page_number → SectionTriple for pages where
    a heading pattern was confidently matched.
    """
    results: dict[int, SectionTriple] = {}

    for pg in page_numbers:
        text = page_text_map.get(pg, "")
        if not text:
            continue

        title = _match_heading_pattern(text)
        if title:
            results[pg] = SectionTriple(
                pg, title.strip(), _regex_section_id(title)
            )

    return results


def _match_heading_pattern(text: str) -> str | None:
    """Apply heading regex patterns to page text.

    Scans the first 10 lines of the text for a heading pattern.
    Skips TOC/index pages and page-header/footer noise.
    """
    lines = text.split("\n")[:10]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip TOC/Index pages
        if any(p.match(line) for p in _TOC_SKIP_PATTERNS):
            return None

        for pattern in _CHAPTER_PATTERNS:
            m = pattern.match(line)
            if m:
                # Extract title from regex match.
                # Strategy: prefer group 2 if it's non-numeric (contains letters or spaces).
                # For patterns with section-number patterns (like "4.2.3" or "5.3"),
                # group 2 is the title. For patterns with compound identifiers
                # (like "Part V §5.3"), group 3 is the title.
                g2 = m.group(2) if m.lastindex is not None and m.lastindex >= 2 else None
                g3 = m.group(3) if m.lastindex is not None and m.lastindex >= 3 else None

                # Prefer group 3 if it exists and is non-trivial; otherwise group 2
                if g3 and g3.strip() and not _is_purely_numeric(g3):
                    title = g3.strip()
                elif g2 and g2.strip() and not _is_purely_numeric(g2):
                    title = g2.strip()
                else:
                    title = line.strip()

                if title and len(title) > 2:
                    return title
                # ALL-CAPS headings match above but fall through here — return line
                return line

    return None


def _is_purely_numeric(s: str) -> bool:
    """Return True if s contains only digits, dots, and whitespace."""
    stripped = s.strip()
    return bool(stripped) and all(c in "0123456789. " for c in stripped)


def _regex_section_id(title: str) -> str:
    """Generate a stable section_id from a title for regex-detected sections."""
    # Use first 3 words, alphanumeric only
    words = [w for w in title.split() if w.isalnum()][:3]
    short = "_".join(words).lower()
    if not short:
        short = "regex"
    # Hash the full title to avoid collisions
    import hashlib
    h = hashlib.sha256(title.encode()).hexdigest()[:8]
    return f"regex_{short[:16]}_{h}"


# ---------------------------------------------------------------------------
# Tier 3 — LLM fallback
# ---------------------------------------------------------------------------

_LLM_SECTION_PROMPT = """You are a section-title detector for Indian urban planning documents.

From the page text below, extract the section or chapter title this page belongs to.

Rules:
- Return ONLY the section title text (no quotes, no explanation)
- Use the exact title as printed in the document (preserve numbering like "5.3" or "Part V")
- If no section title is identifiable, return exactly: [NONE]
- Capitalise normally; do not convert to ALL CAPS unless that's how it appears in the source

Page text (first 500 chars):
{page_text}

Section title:"""


def _detect_via_llm(
    doc_hash: str,
    page_numbers: list[int],
    page_text_map: dict[int, str],
) -> list[str]:
    """Call Gemini to detect section titles for pages that TOC and regex missed.

    Returns a list of titles (in same order as page_numbers) — "[NONE]" if
    the model could not determine a title.
    """
    try:
        from urban_rag.common.settings import get_settings

        api_key = get_settings().gemini_api_key
    except Exception:
        log.warning("sections_llm_unavailable_no_api_key", doc_hash=doc_hash)
        return ["[NONE]"] * len(page_numbers)

    if not api_key or api_key == "test-api-key-for-unit-tests":
        log.warning("sections_llm_skipped_test_env", doc_hash=doc_hash)
        return ["[NONE]"] * len(page_numbers)

    titles: list[str] = []

    for pg in page_numbers:
        page_text = page_text_map.get(pg, "")[:500]

        try:
            title = _call_gemini_section_title(api_key, page_text)
            titles.append(title)
        except Exception as exc:
            log.warning(
                "sections_llm_call_failed",
                doc_hash=doc_hash,
                page=pg,
                error=str(exc),
            )
            titles.append("[NONE]")

    return titles


def _call_gemini_section_title(api_key: str, page_text: str) -> str:
    """Make a single Gemini API call to extract the section title from page text."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    payload = {
        "contents": [{
            "parts": [{
                "text": _LLM_SECTION_PROMPT.format(page_text=page_text[:500])
            }]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 64,
        },
    }

    query = f"?key={api_key}"
    req = urllib.request.Request(  # noqa: S310  # HTTPS only, API key in request body
        url + query,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))

    candidates = data.get("candidates", [])
    if not candidates:
        return "[NONE]"

    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "[NONE]")
    return text.strip()


def _llm_section_id(title: str) -> str:
    """Generate a stable section_id for LLM-detected sections."""
    h = hashlib.sha256(title.encode()).hexdigest()[:8]
    return f"llm_{h}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_page_text_map(raw_pages: list[dict]) -> dict[int, str]:
    """Build a page_number → raw text string map from parsed.json pages."""
    page_map: dict[int, str] = {}

    for page in raw_pages:
        if not isinstance(page, dict):
            continue
        page_num_raw = page.get("page_num", page.get("page", 1))
        page_num: int = int(page_num_raw) if isinstance(page_num_raw, (int, float)) else 1
        if page_num < 1:
            page_num = 1

        elements: list[dict] = page.get("elements", [])
        if not isinstance(elements, list):
            elements = []

        lines: list[str] = []
        for el in elements:
            if not isinstance(el, dict):
                continue
            text = str(el.get("text", "") or el.get("content", "")).strip()
            if text:
                lines.append(text)

        # Fall back to a top-level "text" key
        if not lines:
            text = str(page.get("text", "")).strip()
            if text:
                lines = [text]

        page_map[page_num] = "\n".join(lines)

    return page_map


def _default_title(page_num: int) -> str:
    return f"Page {page_num}"


def _default_section_id(page_num: int) -> str:
    return f"unknown_p{page_num:04d}"


def _load_sections_jsonl(path: Path) -> list[SectionTriple]:
    """Load SectionTriple list from a sections.jsonl file."""
    triples: list[SectionTriple] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            triples.append(
                SectionTriple(
                    page_number=int(obj["page_number"]),
                    section_title=str(obj["section_title"]),
                    section_id=str(obj["section_id"]),
                )
            )
    return triples
