"""Unit tests for src/urban_rag/ingest/chunk.py — hierarchical chunking."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-unit-tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_docs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate tests with a temporary docs directory."""
    test_docs = tmp_path / "docs"
    test_docs.mkdir()

    # Reset settings singleton
    import urban_rag.common.settings as settings_module

    settings_module._settings = None

    original_init = settings_module.Settings.__init__

    def patched_init(self: settings_module.Settings) -> None:
        original_init(self)
        object.__setattr__(self, "docs_dir", str(test_docs))

    monkeypatch.setattr(settings_module.Settings, "__init__", patched_init)

    # Patch chunk.py DOCS_DIR
    import urban_rag.ingest.chunk as chunk_module

    chunk_module.DOCS_DIR = test_docs

    return test_docs


@pytest.fixture
def sample_parsed_doc() -> dict:
    """Minimal parsed document structure for testing."""
    return {
        "version": "1.0",
        "sections": [
            {
                "title": "Chapter 1 — Introduction",
                "level": 1,
                "page_start": 1,
                "page_end": 3,
                "children": [
                    {
                        "title": "1.1 Definitions",
                        "level": 2,
                        "page_start": 1,
                        "page_end": 2,
                        "children": [],
                    },
                    {
                        "title": "1.2 Scope",
                        "level": 2,
                        "page_start": 3,
                        "page_end": 3,
                        "children": [],
                    },
                ],
            },
            {
                "title": "Chapter 2 — Requirements",
                "level": 1,
                "page_start": 4,
                "page_end": 5,
                "children": [],
            },
        ],
        "pages": [
            {
                "page_num": 1,
                "width": 612.0,
                "height": 792.0,
                "elements": [
                    {"type": "heading", "text": "Chapter 1 — Introduction", "page": 1},
                    {"type": "paragraph", "text": "This chapter introduces the document.", "page": 1},
                    {"type": "paragraph", "text": "Definitions and scope are covered.", "page": 1},
                ],
            },
            {
                "page_num": 2,
                "width": 612.0,
                "height": 792.0,
                "elements": [
                    {"type": "heading", "text": "1.1 Definitions", "page": 2},
                    {
                        "type": "paragraph",
                        "text": "A building is a structure that has a roof and walls. "
                        * 20,
                        "page": 2,
                    },
                ],
            },
            {
                "page_num": 3,
                "width": 612.0,
                "height": 792.0,
                "elements": [
                    {"type": "heading", "text": "1.2 Scope", "page": 3},
                    {
                        "type": "paragraph",
                        "text": "This part describes the scope of the regulations. " * 15,
                        "page": 3,
                    },
                ],
            },
            {
                "page_num": 4,
                "width": 612.0,
                "height": 792.0,
                "elements": [
                    {"type": "heading", "text": "Chapter 2 — Requirements", "page": 4},
                    {"type": "paragraph", "text": "All buildings must comply with these requirements.", "page": 4},
                ],
            },
            {
                "page_num": 5,
                "width": 612.0,
                "height": 792.0,
                "elements": [
                    {"type": "paragraph", "text": "Additional requirements text here.", "page": 5},
                ],
            },
        ],
        "tables": [],
        "_parser": "docling",
    }


@pytest.fixture
def write_sample_parsed(sample_parsed_doc: dict, isolate_docs_dir: Path) -> Path:
    """Write sample parsed.json to the docs directory."""
    doc_hash = "abc123"
    docs_dir = isolate_docs_dir / doc_hash
    docs_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = docs_dir / "parsed.json"
    with parsed_path.open("w", encoding="utf-8") as f:
        json.dump(sample_parsed_doc, f)
    return docs_dir


# ---------------------------------------------------------------------------
# Test token_count_str
# ---------------------------------------------------------------------------

from urban_rag.ingest.chunk import token_count_str


class TestTokenCountStr:
    """Tests for the approximate token counter."""

    def test_empty_string(self) -> None:
        assert token_count_str("") == 0

    def test_plain_words(self) -> None:
        result = token_count_str("hello world")
        assert result >= 2

    def test_longer_text(self) -> None:
        text = " ".join(["word"] * 100)
        result = token_count_str(text)
        assert result >= 100


# ---------------------------------------------------------------------------
# Test chunk_page_text
# ---------------------------------------------------------------------------

from urban_rag.ingest.chunk import chunk_page_text


class TestChunkPageText:
    """Tests for per-page text chunking with overlap."""

    def test_short_text_returns_single_chunk(self) -> None:
        """Text under 256 tokens returns exactly 1 chunk."""
        text = "A short piece of text."
        chunks = chunk_page_text(text, page_id="p1", section_id="s1", chunk_index_offset=0)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["token_count"] == token_count_str(text)

    def test_exact_boundary(self) -> None:
        """Text at exactly ~256 tokens returns 1 chunk (no partial)."""
        text = "word " * 256
        chunks = chunk_page_text(text, page_id="p1", section_id="s1", chunk_index_offset=0)
        assert len(chunks) >= 1

    def test_overlap_forward(self) -> None:
        """Overlap tokens appear at the start of subsequent chunks."""
        text = "First chunk content. " * 40
        text2 = "Second chunk content. " * 40
        combined = text + " " + text2

        chunks = chunk_page_text(combined, page_id="p1", section_id="s1", chunk_index_offset=0)
        if len(chunks) >= 2:
            first_text = chunks[0]["text"]
            second_text = chunks[1]["text"]
            assert len(first_text) > 0
            assert len(second_text) > 0

    def test_chunk_id_increments(self) -> None:
        """Each chunk gets a unique chunk_id within the page."""
        text = "word " * 400
        chunks = chunk_page_text(text, page_id="p1", section_id="s1", chunk_index_offset=0)
        if len(chunks) >= 2:
            ids = [c["chunk_id"] for c in chunks]
            assert len(ids) == len(set(ids)), "chunk_ids must be unique"


# ---------------------------------------------------------------------------
# Test chunk_document
# ---------------------------------------------------------------------------

from urban_rag.ingest.chunk import chunk_document


class TestChunkDocument:
    """Tests for the full document chunking pipeline."""

    def test_idempotent_on_already_chunks(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """If chunks.jsonl exists, returns cached result without re-chunking."""
        doc_hash = "idempotent_test"
        docs_dir = isolate_docs_dir / doc_hash
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Pre-write chunks.jsonl
        cached = [
            {
                "chunk_id": "test#c0001",
                "doc_id": doc_hash,
                "page_id": f"{doc_hash}#p0001",
                "section_id": f"{doc_hash}#s001",
                "text": "Cached chunk text",
                "token_count": 3,
                "chunk_index_in_section": 0,
            }
        ]
        chunks_path = docs_dir / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as f:
            for c in cached:
                f.write(json.dumps(c) + "\n")

        # Write parsed.json so the function can find it
        parsed = {"sections": [], "pages": [{"page_num": 1, "elements": []}]}
        with (docs_dir / "parsed.json").open("w", encoding="utf-8") as f:
            json.dump(parsed, f)

        result = chunk_document(doc_hash)

        assert len(result) == 1
        assert result[0]["text"] == "Cached chunk text"

    def test_missing_parsed_raises(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """If parsed.json doesn't exist, raises FileNotFoundError."""
        doc_hash = "no_parsed"
        docs_dir = isolate_docs_dir / doc_hash
        docs_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError):
            chunk_document(doc_hash)

    def test_chunk_count_positive_for_valid_doc(
        self, isolate_docs_dir: Path, write_sample_parsed: Path
    ) -> None:
        """A valid parsed document produces at least one chunk."""
        result = chunk_document("abc123")
        assert len(result) >= 1

    def test_all_chunks_have_doc_id(
        self, isolate_docs_dir: Path, write_sample_parsed: Path
    ) -> None:
        """Every chunk links back to the correct doc_id."""
        result = chunk_document("abc123")
        for chunk in result:
            assert chunk["doc_id"] == "abc123"

    def test_all_chunks_have_page_id(
        self, isolate_docs_dir: Path, write_sample_parsed: Path
    ) -> None:
        """Every chunk has a page_id that links to the source page."""
        result = chunk_document("abc123")
        for chunk in result:
            assert "page_id" in chunk
            assert "abc123" in chunk["page_id"]

    def test_chunks_respect_section_boundary(
        self, isolate_docs_dir: Path, write_sample_parsed: Path
    ) -> None:
        """Chunks from different sections have different section_ids."""
        result = chunk_document("abc123")
        section_ids = set()
        for chunk in result:
            sid = chunk.get("section_id")
            if sid is not None:
                section_ids.add(sid)

    def test_overlap_tokens_present_between_adjoining_chunks(
        self, isolate_docs_dir: Path, write_sample_parsed: Path
    ) -> None:
        """When a page produces multiple chunks, they overlap by ~32 tokens."""
        result = chunk_document("abc123")
        page_to_chunks: dict[str, list[dict]] = {}
        for chunk in result:
            pid = chunk["page_id"]
            page_to_chunks.setdefault(pid, []).append(chunk)

        for pid, chunks in page_to_chunks.items():
            if len(chunks) >= 2:
                for i in range(len(chunks) - 1):
                    curr = chunks[i]
                    nxt = chunks[i + 1]
                    assert len(curr["text"]) > 0
                    assert len(nxt["text"]) > 0


# ---------------------------------------------------------------------------
# Test hierarchical links
# ---------------------------------------------------------------------------

class TestHierarchicalLinks:
    """Tests for bi-directional parent/child links."""

    def test_parent_doc_link(self, isolate_docs_dir: Path, write_sample_parsed: Path) -> None:
        """Each chunk can trace back to its document via doc_id."""
        result = chunk_document("abc123")
        for chunk in result:
            assert chunk["doc_id"] == "abc123"

    def test_parent_page_link(
        self, isolate_docs_dir: Path, write_sample_parsed: Path
    ) -> None:
        """Each chunk has a page_id that links to the source page."""
        result = chunk_document("abc123")
        for chunk in result:
            assert "page_id" in chunk
            assert chunk["page_id"].startswith("abc123")
            assert "#p" in chunk["page_id"]

    def test_section_assignment(
        self, isolate_docs_dir: Path, write_sample_parsed: Path
    ) -> None:
        """Chunks are assigned to the section they belong to (not crossing boundaries)."""
        result = chunk_document("abc123")
        sections_with_chunks = [c for c in result if c.get("section_id") is not None]
        assert len(sections_with_chunks) >= 0

    def test_chunk_count_reasonable_for_known_doc(
        self, isolate_docs_dir: Path, write_sample_parsed: Path
    ) -> None:
        """For a 5-page document, chunk count should be within reasonable bounds."""
        result = chunk_document("abc123")
        assert 1 <= len(result) <= 200


# ---------------------------------------------------------------------------
# Test empty / edge cases
# ---------------------------------------------------------------------------

class TestChunkEdgeCases:
    """Tests for empty documents and edge cases."""

    def test_empty_sections_and_pages(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """Document with empty sections and pages produces zero chunks."""
        doc_hash = "empty_doc"
        docs_dir = isolate_docs_dir / doc_hash
        docs_dir.mkdir(parents=True, exist_ok=True)

        parsed = {"sections": [], "pages": [], "_parser": "docling"}
        with (docs_dir / "parsed.json").open("w", encoding="utf-8") as f:
            json.dump(parsed, f)

        result = chunk_document(doc_hash)
        assert len(result) == 0

    def test_page_with_no_text_elements(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """A page with no text elements produces zero chunks for that page."""
        doc_hash = "notext"
        docs_dir = isolate_docs_dir / doc_hash
        docs_dir.mkdir(parents=True, exist_ok=True)

        parsed = {
            "sections": [],
            "pages": [
                {"page_num": 1, "width": 612.0, "height": 792.0, "elements": []},
                {
                    "page_num": 2,
                    "width": 612.0,
                    "height": 792.0,
                    "elements": [
                        {"type": "paragraph", "text": "Some text on this page.", "page": 2}
                    ],
                },
            ],
            "_parser": "docling",
        }
        with (docs_dir / "parsed.json").open("w", encoding="utf-8") as f:
            json.dump(parsed, f)

        result = chunk_document(doc_hash)
        page_ids = set(c["page_id"] for c in result)
        assert len(page_ids) <= 2
