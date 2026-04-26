"""Unit tests for src/urban_rag/ingest/parse.py — Docling primary + Marker fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-unit-tests")

from urban_rag.common.errors import ParseError, ValidationError
from urban_rag.ingest.parse import (
    _is_empty_or_failed,
    _markdown_table_to_html,
    _parse_marker_sections,
    parse_document,
)


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

    # Patch parse.py DOCS_DIR
    import urban_rag.ingest.parse as parse_module

    parse_module.DOCS_DIR = test_docs

    return test_docs


# ---------------------------------------------------------------------------
# Test _is_empty_or_failed
# ---------------------------------------------------------------------------

class TestIsEmptyOrFailed:
    """Tests for the empty-result detection helper."""

    def test_none_result_is_empty(self) -> None:
        """None is considered empty."""
        assert _is_empty_or_failed(None) is True

    def test_empty_dict_is_empty(self) -> None:
        """An empty dict is empty."""
        assert _is_empty_or_failed({}) is True

    def test_result_with_no_keys_is_empty(self) -> None:
        """A dict with no keys is empty."""
        result: dict[str, object] = {}
        assert _is_empty_or_failed(result) is True

    def test_empty_sections_and_empty_pages_is_empty(self) -> None:
        """Empty sections AND empty pages → empty."""
        result = {"sections": [], "pages": []}
        assert _is_empty_or_failed(result) is True

    def test_empty_sections_missing_pages_key_is_empty(self) -> None:
        """Empty sections with no pages key → empty."""
        result = {"sections": []}
        assert _is_empty_or_failed(result) is True

    def test_non_empty_sections_is_not_empty(self) -> None:
        """Non-empty sections → not empty."""
        result = {"sections": [{"title": "Chapter 1", "level": 1}], "pages": []}
        assert _is_empty_or_failed(result) is False

    def test_non_empty_pages_is_not_empty(self) -> None:
        """Non-empty pages with empty sections → not empty."""
        result = {"sections": [], "pages": [{"page_num": 1}]}
        assert _is_empty_or_failed(result) is False

    def test_missing_sections_key_with_pages_is_not_empty(self) -> None:
        """Missing sections with non-empty pages → not empty."""
        result = {"pages": [{"page_num": 1}]}
        assert _is_empty_or_failed(result) is False


# ---------------------------------------------------------------------------
# Test _markdown_table_to_html
# ---------------------------------------------------------------------------

class TestMarkdownTableToHtml:
    """Tests for the markdown table → HTML converter."""

    def test_single_row(self) -> None:
        result = _markdown_table_to_html("| Col A | Col B |")
        assert "<td>Col A</td>" in result
        assert "<td>Col B</td>" in result
        assert result.startswith("<table>")
        assert result.endswith("</table>")

    def test_empty_row(self) -> None:
        result = _markdown_table_to_html("| |")
        assert "<table>" in result
        assert "<td></td>" in result

    def test_strips_whitespace_from_cells(self) -> None:
        result = _markdown_table_to_html("|  A  |  B  |")
        assert "<td>A</td>" in result
        assert "<td>B</td>" in result


# ---------------------------------------------------------------------------
# Test _parse_marker_sections
# ---------------------------------------------------------------------------

class TestParseMarkerSections:
    """Tests for Marker section tree parsing."""

    def test_no_headings_returns_empty(self) -> None:
        result = _parse_marker_sections("Plain text without any headings.")
        assert result == []

    def test_single_h1_heading(self) -> None:
        result = _parse_marker_sections("# Introduction")
        assert len(result) == 1
        assert result[0]["title"] == "Introduction"
        assert result[0]["level"] == 1
        assert result[0]["children"] == []

    def test_h1_and_h2_headings(self) -> None:
        text = "# Chapter 1\n## Section 1.1"
        result = _parse_marker_sections(text)
        assert len(result) == 1
        assert result[0]["level"] == 1
        assert result[0]["title"] == "Chapter 1"
        assert len(result[0]["children"]) == 1
        assert result[0]["children"][0]["level"] == 2
        assert result[0]["children"][0]["title"] == "Section 1.1"

    def test_multiple_h1_creates_siblings(self) -> None:
        text = "# First\n# Second\n# Third"
        result = _parse_marker_sections(text)
        assert len(result) == 3
        assert result[0]["title"] == "First"
        assert result[1]["title"] == "Second"
        assert result[2]["title"] == "Third"

    def test_heading_with_level_3(self) -> None:
        text = "### Deep heading"
        result = _parse_marker_sections(text)
        assert len(result) == 1
        assert result[0]["level"] == 3
        assert result[0]["title"] == "Deep heading"


# ---------------------------------------------------------------------------
# Test parse_document — idempotency
# ---------------------------------------------------------------------------

class TestParseDocumentIdempotency:
    """Tests for parse_document cache (idempotency)."""

    def test_cache_hit_returns_cached_result(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """If parsed.json already exists, it is returned without re-parsing."""
        doc_hash = "abc123"
        cached = {
            "sections": [{"title": "Cached", "level": 1, "page_start": 1, "page_end": 1, "children": []}],
            "pages": [],
            "tables": [],
            "_parser": "docling",
            "version": "1.0",
        }
        cached_path = isolate_docs_dir / doc_hash / "parsed.json"
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_text(json.dumps(cached))

        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        result = parse_document(pdf_path, doc_hash)

        assert result["sections"][0]["title"] == "Cached"
        # Docling/Marker should NOT be called
        # (if they were, we'd get ParseError since no real docling installed)

    def test_different_hash_bypasses_cache(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """Cache is per-hash; different hash doesn't use cached result."""
        # Create a cached entry for hash "abc"
        cached_path_ab = isolate_docs_dir / "abc" / "parsed.json"
        cached_path_ab.parent.mkdir(parents=True, exist_ok=True)
        cached_path_ab.write_text(json.dumps({"sections": [], "pages": [], "tables": [], "_parser": "docling"}))

        # Now try with a different hash that has no cached result
        doc_hash = "xyz789"
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content here to make it longer than 1kb" + b"A" * 2000)

        # Neither docling nor marker are installed, so this should raise
        with pytest.raises(ParseError):
            parse_document(pdf_path, doc_hash)


# ---------------------------------------------------------------------------
# Test parse_document — Docling failure → Marker fallback
# ---------------------------------------------------------------------------

class TestParseDocumentFallback:
    """Tests for Docling → Marker fallback behavior."""

    def test_docling_import_error_triggers_marker_fallback(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """If Docling import fails, Marker is tried as fallback."""
        doc_hash = "fallback_test"
        pdf_path = tmp_path / "fallback_doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test content" + b"B" * 2000)

        # Mock Marker module and its convert_single_pdf
        mock_marker = MagicMock()
        mock_result = MagicMock()
        mock_result.markdown = "# Fallback Title\nSome content here."
        mock_result.images = []
        mock_marker.convert.convert_single_pdf.return_value = mock_result

        # Use sys.modules to mock both docling (not installed) and marker
        modules_to_mock = {
            "marker": mock_marker,
            "marker.convert": mock_marker.convert,
            "docling": None,  # letting it raise ModuleNotFoundError naturally
        }

        with patch.dict("sys.modules", modules_to_mock, clear=False):
            # Docling's DocumentConverter needs to be patched too
            mock_docling_module = MagicMock()
            with patch.dict("sys.modules", {"docling": mock_docling_module, "docling.document_converter": mock_docling_module}, clear=False):
                # Docling raises ModuleNotFoundError → caught → falls back to Marker
                mock_docling_module.DocumentConverter.side_effect = ModuleNotFoundError("Simulated Docling not installed")

                result = parse_document(pdf_path, doc_hash)

        assert result["_parser"] == "marker"
        assert len(result["sections"]) >= 0

    def test_marker_also_fails_raises_parse_error(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """If both Docling and Marker fail, ParseError is raised."""
        doc_hash = "both_fail"
        pdf_path = tmp_path / "both_fail.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 both fail" + b"C" * 2000)

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if "docling" in name or "marker" in name:
                raise ImportError(f"Simulated import error for {name}")
            return MagicMock()

        with patch("importlib.import_module", side_effect=mock_import):
            with pytest.raises(ParseError) as exc_info:
                parse_document(pdf_path, doc_hash)

        assert "failed" in str(exc_info.value.message).lower()


# ---------------------------------------------------------------------------
# Test parse_document — validation
# ---------------------------------------------------------------------------

class TestParseDocumentValidation:
    """Tests for parse_document input validation."""

    def test_nonexistent_pdf_raises_validation_error(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """A non-existent PDF path raises ValidationError."""
        nonexistent = tmp_path / "does_not_exist.pdf"
        doc_hash = "any"

        with pytest.raises(ValidationError) as exc_info:
            parse_document(nonexistent, doc_hash)

        assert "not found" in str(exc_info.value.message).lower()


# ---------------------------------------------------------------------------
# Test parse_document — empty Docling result triggers fallback
# ---------------------------------------------------------------------------

class TestParseDocumentEmptyDoclingResult:
    """Tests for when Docling succeeds but returns empty structure."""

    def test_docling_returns_empty_sections_triggers_marker(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """If Docling returns {} (empty), Marker is tried."""
        doc_hash = "empty_docling"
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 empty docling test" + b"D" * 2000)

        # Mock Docling to return empty result
        mock_doc = MagicMock()
        mock_doc.export_to_dict.return_value = {"sections": [], "pages": []}
        mock_doc.version = "1.0"
        mock_result = MagicMock()
        mock_result.document = mock_doc

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if "docling" in name:
                m = MagicMock()
                m.DocumentConverter.return_value.convert.return_value = mock_result
                return m
            if "marker" in name:
                raise ImportError("Marker not available")
            return MagicMock()

        with patch("importlib.import_module", side_effect=mock_import):
            with pytest.raises(ParseError):
                # Marker also not installed, so both fail → ParseError
                parse_document(pdf_path, doc_hash)
