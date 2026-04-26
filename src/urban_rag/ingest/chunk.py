"""Stage 5 — Hierarchical text chunking.

Hierarchical chunking at four levels: document → section → page → patch.

  - Document: all sections and pages
  - Section : a heading-driven region within the document hierarchy
  - Page    : a single rendered page, bounded by its page_num
  - Patch   : a 256-token text chunk (with 32-token overlap), never crossing
              section boundaries

Text is extracted from Docling/Marker `parsed.json` elements (paragraphs,
headings, table captions, etc.). Chunks are never truncated mid-sentence when
possible; the overlap window is aligned to sentence boundaries where possible.

Output: docs/<hash>/chunks.jsonl — one JSON line per ChunkRecord.

PART V §5.6 governs chunk sizing (256 tokens target, 32-token overlap).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from urban_rag.common.logging import get_logger

log = get_logger(__name__, service="ingest")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target chunk size in tokens (PART V §5.6)
TARGET_CHUNK_TOKENS = 256
# Overlap between adjacent chunks in tokens (PART V §5.6)
CHUNK_OVERLAP_TOKENS = 32

DOCS_DIR: Path | None = None  # Lazily resolved


def _get_docs_dir() -> Path:
    """Return docs directory from settings."""
    global DOCS_DIR
    if DOCS_DIR is None:
        from urban_rag.common.settings import get_settings

        DOCS_DIR = Path(get_settings().docs_dir)
    return DOCS_DIR


# ---------------------------------------------------------------------------
# Token counting (approximate)
# ---------------------------------------------------------------------------

# Average chars-per-token ratio for English prose (tiktoken CL100k base)
# 4 chars/token is the canonical OpenAI approximation
_CHARS_PER_TOKEN = 4.0


def token_count_str(text: str) -> int:
    """Approximate token count using the chars-per-token heuristic.

    This avoids adding tiktoken as a hard dependency while providing a
    reasonable estimate for chunk sizing.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text.strip()) / _CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _extract_page_text(page: dict) -> str:
    """Concatenate all text-bearing elements from a page into a single string.

    Elements are processed in document order.  Paragraphs, headings, captions,
    and list items are included.  Empty or non-text elements are skipped.
    """
    lines: list[str] = []
    elements: list[dict] = page.get("elements", [])
    if not isinstance(elements, list):
        elements = []

    for el in elements:
        if not isinstance(el, dict):
            continue
        elem_type = str(el.get("type", "")).lower()
        text = str(el.get("text", "") or el.get("content", "")).strip()
        if not text:
            continue
        if elem_type in (
            "paragraph",
            "heading",
            "h1",
            "h2",
            "h3",
            "h4",
            "caption",
            "list",
            "footnote",
            "table",
        ):
            lines.append(text)

    # Also accept "fields" as a page-level element list (Marker format)
    if not lines:
        fields: list[dict] = page.get("fields", [])
        if isinstance(fields, list):
            for el in fields:
                if not isinstance(el, dict):
                    continue
                text = str(el.get("text", "") or el.get("content", "")).strip()
                if text:
                    lines.append(text)

    return "\n".join(lines)


def _assign_section_to_page(
    page_num: int, sections: list[dict]
) -> tuple[str | None, str | None, list[str]]:
    """Return the section and section title for a given page number.

    Returns:
        (section_id, section_title, title_path) for the deepest section
        whose page_start <= page_num <= page_end.  None if no section matches.
    """
    if not sections:
        return None, None, []

    best: dict | None = None
    best_level = -1

    def matches(sec: dict, pg: int) -> bool:
        start = sec.get("page_start", 1)
        end = sec.get("page_end", start)
        return start <= pg <= end

    for sec in _flatten_sections(sections):
        level = sec.get("level", 99)
        if matches(sec, page_num) and level > best_level:
            best = sec
            best_level = level

    if best is None:
        return None, None, []

    section_id = best.get("section_id")
    title = best.get("title", "")
    title_path = best.get("title_path", [])

    return section_id, title, title_path


def _flatten_sections(
    sections: list[dict], prefix: list[str] | None = None
) -> list[dict]:
    """Flatten a section tree into a flat list, annotating each with a title_path.

    Each section dict in the result also gets a synthetic `section_id` of the form
    ``{doc_hash}#s{idx:03d}`` and its parent's path prepended to title_path.
    """
    flat: list[dict] = []
    if prefix is None:
        prefix = []

    for sec in sections:
        if not isinstance(sec, dict):
            continue
        title = str(sec.get("title", "")).strip()
        current_path = [*prefix, title] if title else prefix

        # Assign section_id if not present
        s = dict(sec)
        if "section_id" not in s:
            s["section_id"] = f"s{len(flat):03d}"
        s["title_path"] = current_path

        flat.append(s)

        # Recurse into children
        children: list[dict] = sec.get("children", [])
        if isinstance(children, list) and children:
            child_flat = _flatten_sections(children, current_path)
            flat.extend(child_flat)

    return flat


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------


def chunk_page_text(
    text: str,
    page_id: str,
    section_id: str | None,
    chunk_index_offset: int,
    max_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """Split page-level text into overlapping 256-token chunks.

    Chunks are produced by sliding a window of ``max_tokens`` tokens forward,
    overlapping by ``overlap_tokens`` tokens on each step.  Windows are
    aligned to sentence boundaries when possible (heuristic: stop at periods,
    question marks, or exclamation marks followed by whitespace).

    Args:
        text              : concatenated page text
        page_id           : f"{doc_hash}#p{page_num:04d}"
        section_id        : section record ID or None
        chunk_index_offset: starting index for this page's chunks in the
                           global section chunk sequence
        max_tokens        : target window size in tokens (default 256)
        overlap_tokens    : overlap in tokens (default 32)

    Returns:
        A list of chunk dicts (NOT ChunkRecord — plain dict for JSON serialisation
        and to avoid import cycles with the types module which the types module
        itself may need).  Each dict has the fields from ChunkRecord.
    """
    if not text or not text.strip():
        return []

    # ── Token-count the full text ─────────────────────────────────────────
    total_tokens = token_count_str(text)
    if total_tokens <= max_tokens:
        # Short enough for a single chunk
        return [
            _make_chunk_dict(
                page_id=page_id,
                section_id=section_id,
                chunk_index=chunk_index_offset,
                chunk_text=text.strip(),
            )
        ]

    # ── Sliding window with overlap ────────────────────────────────────────
    chunks: list[dict] = []
    char_width = math.ceil(max_tokens * _CHARS_PER_TOKEN)
    overlap_char = math.ceil(overlap_tokens * _CHARS_PER_TOKEN)

    # Window advance = max_tokens - overlap_tokens in token space
    advance_char = char_width - overlap_char
    if advance_char <= 0:
        advance_char = max(1, char_width // 2)

    start = 0
    total_len = len(text)
    chunk_idx = chunk_index_offset

    while start < total_len:
        end = min(start + char_width, total_len)

        # Try to break at sentence boundary near the window end
        if end < total_len:
            # Look for sentence-ending punctuation in the last 20% of the window
            search_start = start + int(0.80 * (end - start))
            sentence_end = _find_sentence_boundary(text, search_start, end)
            if sentence_end > start:
                end = sentence_end + 1  # include the punctuation char

        window = text[start:end].strip()
        if not window:
            start += advance_char
            continue

        chunk_dict = _make_chunk_dict(
            page_id=page_id,
            section_id=section_id,
            chunk_index=chunk_idx,
            chunk_text=window,
        )
        chunks.append(chunk_dict)
        chunk_idx += 1

        # Advance: but include the overlap region in the next window
        # The overlap is the last overlap_char characters
        advance_to = end - overlap_char
        if advance_to <= start:
            advance_to = start + max(1, advance_char)
        start = advance_to

        # Safety: prevent infinite loop for very short texts
        if start >= total_len:
            break

    return chunks


def _make_chunk_dict(
    page_id: str,
    section_id: str | None,
    chunk_index: int,
    chunk_text: str,
) -> dict:
    """Build a chunk dict matching ChunkRecord fields."""
    # Derive doc_id from page_id (first segment before #p)
    doc_id = page_id.split("#")[0]
    token_count = token_count_str(chunk_text)
    chunk_id = f"{doc_id}#c{chunk_index:05d}"

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "page_id": page_id,
        "section_id": section_id,
        "text": chunk_text,
        "token_count": token_count,
        "embedding_model": "Alibaba-NLP/gte-modernbert-colbert",
        "chunk_index_in_section": chunk_index,
    }


_SENTENCE_RE = re.compile(r"[.!?]\s+")


def _find_sentence_boundary(text: str, start: int, end: int) -> int:
    """Find the last sentence boundary in ``text[start:end]``.

    Returns the index of the last sentence-ending punctuation, or ``start-1``
    if none is found (in which case no sentence-aligned break is made).
    """
    window = text[start:end]
    # Scan backwards from end for sentence-ending punctuation
    search = window[::-1]
    for i, ch in enumerate(search):
        if ch in ".!?":
            # Return absolute position of this char in original text
            return end - i - 1

    return start - 1  # no boundary found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(doc_hash: str) -> list[dict]:
    """Chunk an already-parsed document into text chunks for the text channel.

    Extraction order:
      1. Walk the section tree (docling hierarchy)
      2. For each section, collect all pages in [page_start, page_end]
      3. Extract text from each page's elements
      4. Split into 256-token chunks with 32-token overlap, never crossing
         section boundaries
      5. Assign section_id and page_id parent links

    Output is written to docs/<hash>/chunks.jsonl for idempotency and re-ingest.

    Args:
        doc_hash: SHA256 content hash of the document.

    Returns:
        A list of chunk dicts (one per chunk).  The dicts contain all
        ChunkRecord fields plus _parent_doc_id, _parent_page_id, _parent_section_id
        for bi-directional navigation.

    Raises:
        FileNotFoundError: if docs/<hash>/parsed.json does not exist
                          (run parse_document before chunk_document).
    """
    docs_dir = _get_docs_dir()
    dest_dir = docs_dir / doc_hash
    parsed_json_path = dest_dir / "parsed.json"
    chunks_jsonl_path = dest_dir / "chunks.jsonl"

    # ── Idempotency: return cached result if already chunked ────────────────
    if chunks_jsonl_path.exists():
        log.info("chunk_cache_hit", doc_hash=doc_hash)
        return _load_chunks_jsonl(chunks_jsonl_path)

    # ── Load parsed output ─────────────────────────────────────────────────
    if not parsed_json_path.exists():
        raise FileNotFoundError(
            f"parsed.json not found for {doc_hash[:12]}... "
            "(run parse_document before chunk_document)"
        )

    with parsed_json_path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    log.info("chunk_start", doc_hash=doc_hash)

    sections: list[dict] = parsed.get("sections", [])
    if not isinstance(sections, list):
        sections = []

    raw_pages: list[dict] = parsed.get("pages", [])
    if not isinstance(raw_pages, list):
        raw_pages = []

    # ── Build flat section list with title paths ───────────────────────────
    flat_sections = _flatten_sections(sections)

    # ── Map page_num → section for quick lookup ───────────────────────────
    # Pre-assign a section to each page
    page_to_section: dict[int, tuple[str | None, str | None, list[str]]] = {}
    for page in raw_pages:
        page_num_raw = page.get("page_num", page.get("page", 1))
        page_num: int = int(page_num_raw) if isinstance(page_num_raw, (int, float)) else 1
        if page_num < 1:
            page_num = 1
        section_id, section_title, title_path = _assign_section_to_page(page_num, sections)
        page_to_section[page_num] = (section_id, section_title, title_path)

    # ── Walk sections and pages, collecting chunks ────────────────────────
    all_chunks: list[dict] = []
    chunk_global_index = 0

    for sec in flat_sections:
        sec_id = sec.get("section_id")
        sec_page_start = int(sec.get("page_start", 1))
        sec_page_end = int(sec.get("page_end", sec_page_start))

        # Collect pages for this section
        section_pages = [
            p for p in raw_pages
            if sec_page_start
            <= int(p.get("page_num", p.get("page", 1)))
            <= sec_page_end
        ]

        for page in section_pages:
            page_num_raw = page.get("page_num", page.get("page", 1))
            page_num: int = int(page_num_raw) if isinstance(page_num_raw, (int, float)) else 1
            if page_num < 1:
                page_num = 1

            page_id = f"{doc_hash}#p{page_num:04d}"

            # Extract text from page elements
            page_text = _extract_page_text(page)
            if not page_text.strip():
                continue

            page_chunks = chunk_page_text(
                text=page_text,
                page_id=page_id,
                section_id=sec_id,
                chunk_index_offset=chunk_global_index,
            )

            # Annotate each chunk with parent links (bi-directional)
            for chunk in page_chunks:
                chunk["_parent_doc_id"] = doc_hash
                chunk["_parent_page_id"] = page_id
                chunk["_parent_section_id"] = sec_id

            all_chunks.extend(page_chunks)
            chunk_global_index += len(page_chunks)

    # ── Also include pages with no section (orphan pages) ─────────────────
    # These are pages that fell outside any defined section range
    section_page_nums: set[int] = set()
    for sec in flat_sections:
        start = int(sec.get("page_start", 1))
        end = int(sec.get("page_end", start))
        for pg in range(start, end + 1):
            section_page_nums.add(pg)

    orphaned_pages = [
        p for p in raw_pages
        if int(p.get("page_num", p.get("page", 1))) not in section_page_nums
    ]

    for page in orphaned_pages:
        page_num_raw = page.get("page_num", page.get("page", 1))
        page_num: int = int(page_num_raw) if isinstance(page_num_raw, (int, float)) else 1
        if page_num < 1:
            page_num = 1

        page_id = f"{doc_hash}#p{page_num:04d}"
        page_text = _extract_page_text(page)
        if not page_text.strip():
            continue

        page_chunks = chunk_page_text(
            text=page_text,
            page_id=page_id,
            section_id=None,  # No section
            chunk_index_offset=chunk_global_index,
        )

        for chunk in page_chunks:
            chunk["_parent_doc_id"] = doc_hash
            chunk["_parent_page_id"] = page_id
            chunk["_parent_section_id"] = None

        all_chunks.extend(page_chunks)
        chunk_global_index += len(page_chunks)

    # ── Persist chunks.jsonl ───────────────────────────────────────────────
    dest_dir.mkdir(parents=True, exist_ok=True)
    with chunks_jsonl_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    log.info(
        "chunk_complete",
        doc_hash=doc_hash,
        chunk_count=len(all_chunks),
    )

    return all_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_chunks_jsonl(path: Path) -> list[dict]:
    """Load chunk dicts from a chunks.jsonl file."""
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks
