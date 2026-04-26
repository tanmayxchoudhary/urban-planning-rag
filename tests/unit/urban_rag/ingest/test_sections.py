"""Unit tests for src/urban_rag/ingest/sections.py — section-title detection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-unit-tests")

from urban_rag.ingest.sections import (
    SectionTriple,
    _build_toc_map,
    _default_section_id,
    _default_title,
    _detect_via_llm,
    _detect_via_regex,
    _extract_page_text_map,
    _flatten_sections,
    _llm_section_id,
    _match_heading_pattern,
    _regex_section_id,
    detect_sections,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_docs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate tests with a temporary docs directory."""
    test_docs = tmp_path / "docs"
    test_docs.mkdir()

    import urban_rag.common.settings as settings_module

    settings_module._settings = None

    original_init = settings_module.Settings.__init__

    def patched_init(self: settings_module.Settings) -> None:
        original_init(self)
        object.__setattr__(self, "docs_dir", str(test_docs))

    monkeypatch.setattr(settings_module.Settings, "__init__", patched_init)

    import urban_rag.ingest.sections as sections_module

    sections_module.DOCS_DIR = test_docs

    return test_docs


# ---------------------------------------------------------------------------
# Test _flatten_sections
# ---------------------------------------------------------------------------

class TestFlattenSections:
    def test_empty_sections_returns_empty(self) -> None:
        result = _flatten_sections([])
        assert result == []

    def test_single_section_flattened(self) -> None:
        sections = [
            {"title": "Chapter 1", "level": 1, "page_start": 1, "page_end": 10, "children": []}
        ]
        result = _flatten_sections(sections)
        assert len(result) == 1
        assert result[0]["title"] == "Chapter 1"
        assert result[0]["title_path"] == ["Chapter 1"]

    def test_nested_children_flattened(self) -> None:
        sections = [
            {
                "title": "Chapter 1",
                "level": 1,
                "page_start": 1,
                "page_end": 20,
                "children": [
                    {
                        "title": "Section 1.1",
                        "level": 2,
                        "page_start": 1,
                        "page_end": 10,
                        "children": [],
                    },
                ],
            }
        ]
        result = _flatten_sections(sections)
        assert len(result) == 2
        assert result[0]["title"] == "Chapter 1"
        assert result[0]["title_path"] == ["Chapter 1"]
        assert result[1]["title"] == "Section 1.1"
        assert result[1]["title_path"] == ["Chapter 1", "Section 1.1"]

    def test_missing_title_uses_empty_string(self) -> None:
        sections = [{"level": 1, "page_start": 1, "page_end": 1, "children": []}]
        result = _flatten_sections(sections)
        assert result[0]["title"] == ""
        assert result[0]["title_path"] == []

    def test_section_id_preassigned_if_present(self) -> None:
        sections = [
            {"title": "Intro", "level": 1, "page_start": 1, "page_end": 1, "section_id": "custom_id", "children": []}
        ]
        result = _flatten_sections(sections)
        assert result[0]["section_id"] == "custom_id"


# ---------------------------------------------------------------------------
# Test _build_toc_map
# ---------------------------------------------------------------------------

class TestBuildTocMap:
    def test_single_page_in_range(self) -> None:
        flat = [
            {"section_id": "s001", "title": "Chapter 1", "level": 1, "page_start": 1, "page_end": 5},
        ]
        result = _build_toc_map(flat)
        assert result[1] == ("s001", "Chapter 1")
        assert result[5] == ("s001", "Chapter 1")

    def test_deeper_section_wins(self) -> None:
        flat = [
            {"section_id": "s000", "title": "Part I", "level": 1, "page_start": 1, "page_end": 20},
            {"section_id": "s001", "title": "Chapter 3", "level": 2, "page_start": 5, "page_end": 15},
        ]
        result = _build_toc_map(flat)
        # Page 10 is within both; level 2 > level 1, so s001 should win
        assert result[10] == ("s001", "Chapter 3")
        # Page 1 is only in Part I
        assert result[1] == ("s000", "Part I")

    def test_pages_outside_all_sections(self) -> None:
        flat = [
            {"section_id": "s000", "title": "Intro", "level": 1, "page_start": 5, "page_end": 10},
        ]
        result = _build_toc_map(flat)
        # Pages 1-4 are not covered
        assert 1 not in result
        assert 5 in result
        assert 20 not in result  # beyond all ranges


# ---------------------------------------------------------------------------
# Test _extract_page_text_map
# ---------------------------------------------------------------------------

class TestExtractPageTextMap:
    def test_extracts_text_from_elements(self) -> None:
        pages = [
            {
                "page_num": 1,
                "elements": [
                    {"type": "paragraph", "text": "First paragraph."},
                    {"type": "heading", "text": "Chapter 1"},
                ],
            },
        ]
        result = _extract_page_text_map(pages)
        assert "Chapter 1" in result[1]
        assert "First paragraph" in result[1]

    def test_handles_missing_elements(self) -> None:
        pages = [{"page_num": 1}]
        result = _extract_page_text_map(pages)
        assert result[1] == ""

    def test_page_num_as_float(self) -> None:
        pages = [{"page_num": 3.0, "elements": [{"type": "paragraph", "text": "Text"}]}]
        result = _extract_page_text_map(pages)
        assert 3 in result


# ---------------------------------------------------------------------------
# Test _match_heading_pattern
# ---------------------------------------------------------------------------

class TestMatchHeadingPattern:
    def test_chapter_heading(self) -> None:
        text = "Chapter 5: Zoning Regulations"
        result = _match_heading_pattern(text)
        assert result is not None
        assert "Zoning" in result

    def test_chapter_with_dash(self) -> None:
        text = "Chapter 3 - Development Control Rules"
        result = _match_heading_pattern(text)
        assert result is not None
        assert "Development" in result

    def test_part_heading(self) -> None:
        text = "Part V  Fire Safety Requirements"
        result = _match_heading_pattern(text)
        assert result is not None
        assert "Fire Safety" in result

    def test_numbered_section(self) -> None:
        text = "4.2.3  Floor Space Index"
        result = _match_heading_pattern(text)
        assert result is not None
        assert "Floor" in result or "Index" in result

    def test_urdffi_style(self) -> None:
        text = "Part V §5.3  Building Height"
        result = _match_heading_pattern(text)
        assert result is not None

    def test_all_caps_heading(self) -> None:
        text = "CHAPTER 1  PRELIMINARY"
        result = _match_heading_pattern(text)
        assert result is not None

    def test_toc_page_returns_none(self) -> None:
        text = "Table of Contents"
        result = _match_heading_pattern(text)
        assert result is None

    def test_plain_text_returns_none(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        result = _match_heading_pattern(text)
        assert result is None


# ---------------------------------------------------------------------------
# Test _detect_via_regex
# ---------------------------------------------------------------------------

class TestDetectViaRegex:
    def test_resolves_heading_pages(self) -> None:
        page_text_map = {
            1: "Chapter 1: Introduction\nSome text.",
            2: "This is regular page content.",
            3: "Chapter 2: Zoning\nMore text.",
        }
        result = _detect_via_regex([1, 2, 3], page_text_map)
        assert 1 in result
        assert 3 in result
        assert 2 not in result

    def test_empty_page_map(self) -> None:
        result = _detect_via_regex([1, 2], {})
        assert result == {}


# ---------------------------------------------------------------------------
# Test detect_sections — idempotency
# ---------------------------------------------------------------------------

class TestDetectSectionsIdempotency:
    def test_cache_hit_returns_cached(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """If sections.jsonl exists, it is returned without re-detecting."""
        doc_hash = "abc123"
        cached = [
            {"page_number": 1, "section_title": "Chapter 1", "section_id": "s001"},
            {"page_number": 2, "section_title": "Chapter 1", "section_id": "s001"},
        ]
        cached_path = isolate_docs_dir / doc_hash / "sections.jsonl"
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with cached_path.open("w") as f:
            for item in cached:
                f.write(json.dumps(item) + "\n")

        # Create a dummy parsed.json so the function finds it
        parsed_path = isolate_docs_dir / doc_hash / "parsed.json"
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.write_text(json.dumps({"sections": [], "pages": []}))

        result = detect_sections(doc_hash)
        assert len(result) == 2
        assert result[0].section_title == "Chapter 1"


# ---------------------------------------------------------------------------
# Test detect_sections — TOC tier
# ---------------------------------------------------------------------------

class TestDetectSectionsTOC:
    def test_toc_resolves_all_pages(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """TOC hierarchy resolves every page that has a TOC entry."""
        doc_hash = "toc_test"
        parsed = {
            "sections": [
                {
                    "title": "Chapter 1",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 5,
                    "children": [
                        {
                            "title": "Section 1.1",
                            "level": 2,
                            "page_start": 1,
                            "page_end": 3,
                            "children": [],
                        },
                    ],
                },
                {
                    "title": "Chapter 2",
                    "level": 1,
                    "page_start": 6,
                    "page_end": 10,
                    "children": [],
                },
            ],
            "pages": [
                {"page_num": 1}, {"page_num": 2}, {"page_num": 3},
                {"page_num": 6}, {"page_num": 7},
            ],
        }
        parsed_path = isolate_docs_dir / doc_hash / "parsed.json"
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.write_text(json.dumps(parsed))

        result = detect_sections(doc_hash)
        page_map = {t.page_number: t.section_title for t in result}

        assert page_map[1] == "Section 1.1"
        assert page_map[3] == "Section 1.1"
        assert page_map[6] == "Chapter 2"
        # Page 7 falls within Chapter 2's range (6-10), so TOC resolves it
        assert page_map[7] == "Chapter 2"


# ---------------------------------------------------------------------------
# Test detect_sections — regex fallback
# ---------------------------------------------------------------------------

class TestDetectSectionsRegexFallback:
    def test_regex_fills_missing_pages(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """Regex resolves pages that TOC didn't cover."""
        doc_hash = "regex_test"
        parsed = {
            "sections": [
                {
                    "title": "Chapter 1",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 3,
                    "children": [],
                },
            ],
            "pages": [
                {"page_num": 1},
                {"page_num": 2},
                {"page_num": 3},
                {"page_num": 4},
                {"page_num": 5},
            ],
        }
        # Add heading text to page 4
        parsed["pages"].append({
            "page_num": 4,
            "elements": [{"type": "paragraph", "text": "Chapter 2: Parking Requirements\nLots of text."}],
        })

        parsed_path = isolate_docs_dir / doc_hash / "parsed.json"
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.write_text(json.dumps(parsed))

        result = detect_sections(doc_hash)
        page_map = {t.page_number: t.section_title for t in result}

        assert page_map[1] == "Chapter 1"
        assert page_map[4] == "Parking Requirements"


# ---------------------------------------------------------------------------
# Test detect_sections — LLM fallback skipped in test env
# ---------------------------------------------------------------------------

class TestDetectSectionsLLMFallback:
    def test_llm_skipped_in_test_env(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """When API key is test key, LLM is skipped and default titles used."""
        doc_hash = "llm_test"
        parsed = {
            "sections": [],
            "pages": [
                {"page_num": 1, "elements": [{"type": "paragraph", "text": "Random page text."}]},
            ],
        }
        parsed_path = isolate_docs_dir / doc_hash / "parsed.json"
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.write_text(json.dumps(parsed))

        result = detect_sections(doc_hash)
        assert len(result) == 1
        assert result[0].section_title == "Page 1"


# ---------------------------------------------------------------------------
# Test _regex_section_id
# ---------------------------------------------------------------------------

class TestRegexSectionId:
    def test_id_is_stable(self) -> None:
        title = "Chapter 3 Zoning Regulations"
        id1 = _regex_section_id(title)
        id2 = _regex_section_id(title)
        assert id1 == id2

    def test_id_differs_for_different_titles(self) -> None:
        id1 = _regex_section_id("Chapter 1")
        id2 = _regex_section_id("Chapter 2")
        assert id1 != id2

    def test_id_starts_with_regex_prefix(self) -> None:
        id_ = _regex_section_id("Some Title")
        assert id_.startswith("regex_")


# ---------------------------------------------------------------------------
# Test _llm_section_id
# ---------------------------------------------------------------------------

class TestLlmSectionId:
    def test_llm_id_starts_with_llm_prefix(self) -> None:
        id_ = _llm_section_id("Some Title")
        assert id_.startswith("llm_")


# ---------------------------------------------------------------------------
# Test SectionTriple NamedTuple
# ---------------------------------------------------------------------------

class TestSectionTriple:
    def test_is_named_tuple(self) -> None:
        t = SectionTriple(5, "Chapter 2", "s002")
        assert t.page_number == 5
        assert t.section_title == "Chapter 2"
        assert t.section_id == "s002"

    def test_iterable(self) -> None:
        t = SectionTriple(1, "Title", "s001")
        a, b, c = t
        assert (a, b, c) == (1, "Title", "s001")
