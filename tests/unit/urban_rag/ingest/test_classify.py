"""Unit tests for src/urban_rag/ingest/classify.py — adaptive DPI page classification."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-unit-tests")

from urban_rag.ingest.classify import (
    DPI_TEXT,
    DPI_VISUAL,
    DRAWING_THRESHOLD,
    _build_page_records,
    _load_pages_jsonl,
    _page_has_images,
    _page_num_drawings,
    classify_pages,
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

    # Patch classify.py DOCS_DIR
    import urban_rag.ingest.classify as classify_module

    classify_module.DOCS_DIR = test_docs

    return test_docs


# ---------------------------------------------------------------------------
# Test DPI constants
# ---------------------------------------------------------------------------

class TestDpiConstants:
    """Verify DPI constants match PART V §5.3."""

    def test_dpi_visual_is_250(self) -> None:
        assert DPI_VISUAL == 250

    def test_dpi_text_is_100(self) -> None:
        assert DPI_TEXT == 100

    def test_drawing_threshold_is_40(self) -> None:
        assert DRAWING_THRESHOLD == 40


# ---------------------------------------------------------------------------
# Test _page_has_images
# ---------------------------------------------------------------------------

class TestPageHasImages:
    """Tests for the image detection helper."""

    def test_no_images_returns_false(self) -> None:
        page: dict = {"page_num": 1, "elements": []}
        assert _page_has_images(page) is False

    def test_empty_images_list_returns_false(self) -> None:
        page: dict = {"page_num": 1, "images": [], "elements": []}
        assert _page_has_images(page) is False

    def test_images_list_with_items_returns_true(self) -> None:
        page: dict = {"page_num": 1, "images": [{"path": "/img1.png"}], "elements": []}
        assert _page_has_images(page) is True

    def test_elements_with_image_type_returns_true(self) -> None:
        page: dict = {
            "page_num": 1,
            "elements": [{"type": "image", "src": "/img1.png"}],
        }
        assert _page_has_images(page) is True

    def test_elements_with_figure_type_returns_true(self) -> None:
        page: dict = {
            "page_num": 1,
            "elements": [{"type": "figure", "caption": "Fig 1"}],
        }
        assert _page_has_images(page) is True

    def test_elements_with_non_image_type_returns_false(self) -> None:
        page: dict = {
            "page_num": 1,
            "elements": [{"type": "paragraph", "text": "Hello"}],
        }
        assert _page_has_images(page) is False


# ---------------------------------------------------------------------------
# Test _page_num_drawings
# ---------------------------------------------------------------------------

class TestPageNumDrawings:
    """Tests for the drawing count helper."""

    def test_no_elements_returns_zero(self) -> None:
        page: dict = {"page_num": 1, "elements": []}
        assert _page_num_drawings(page) == 0

    def test_line_elements_counted(self) -> None:
        page: dict = {
            "page_num": 1,
            "elements": [
                {"type": "line", "points": [0, 0, 100, 0]},
                {"type": "line", "points": [0, 0, 0, 100]},
            ],
        }
        assert _page_num_drawings(page) == 2

    def test_rect_elements_counted(self) -> None:
        page: dict = {
            "page_num": 1,
            "elements": [{"type": "rect", "bbox": [0, 0, 100, 100]}],
        }
        assert _page_num_drawings(page) == 1

    def test_path_elements_counted(self) -> None:
        page: dict = {
            "page_num": 1,
            "elements": [{"type": "path", "d": "M0,0 L100,100"}],
        }
        assert _page_num_drawings(page) == 1

    def test_drawings_list_counted(self) -> None:
        page: dict = {
            "page_num": 1,
            "elements": [
                {
                    "type": "block",
                    "drawings": [{"type": "rect"}, {"type": "line"}],
                }
            ],
        }
        assert _page_num_drawings(page) == 2

    def test_paths_list_counted(self) -> None:
        page: dict = {
            "page_num": 1,
            "elements": [
                {
                    "type": "block",
                    "paths": [{"type": "polyline"}, {"type": "polyline"}, {"type": "polyline"}],
                }
            ],
        }
        assert _page_num_drawings(page) == 3

    def test_combined_drawing_types(self) -> None:
        """Multiple drawing types accumulate."""
        page: dict = {
            "page_num": 1,
            "elements": [
                {"type": "line", "points": [0, 0, 100, 0]},
                {"type": "rect", "bbox": [0, 0, 100, 100]},
                {"type": "path", "d": "M0,0"},
                {"type": "polygon", "points": [0, 0, 100, 0, 100, 100]},
                {"type": "drawing"},
            ],
        }
        assert _page_num_drawings(page) == 5

    def test_non_drawing_types_ignored(self) -> None:
        page: dict = {
            "page_num": 1,
            "elements": [
                {"type": "paragraph", "text": "Hello"},
                {"type": "heading", "level": 1},
                {"type": "table", "rows": []},
            ],
        }
        assert _page_num_drawings(page) == 0


# ---------------------------------------------------------------------------
# Test _build_page_records
# ---------------------------------------------------------------------------

class TestBuildPageRecords:
    """Tests for the page record builder."""

    def test_text_page_classification(self) -> None:
        """A page with no images and ≤40 drawings → TEXT, DPI 100."""
        parsed = {
            "pages": [
                {
                    "page_num": 1,
                    "elements": [{"type": "paragraph", "text": "Hello world"}],
                }
            ]
        }
        records = _build_page_records("abc123", parsed)
        assert len(records) == 1
        assert records[0].page_type == "TEXT"
        assert records[0].dpi_used == DPI_TEXT

    def test_visual_page_via_images(self) -> None:
        """A page with has_images=True → VISUAL, DPI 250."""
        parsed = {
            "pages": [
                {
                    "page_num": 1,
                    "images": [{"path": "/img1.png"}],
                    "elements": [],
                }
            ]
        }
        records = _build_page_records("abc123", parsed)
        assert len(records) == 1
        assert records[0].page_type == "VISUAL"
        assert records[0].dpi_used == DPI_VISUAL

    def test_visual_page_via_drawings_above_threshold(self) -> None:
        """A page with drawings > 40 → VISUAL, DPI 250."""
        # Create a page with 41 line elements
        elements = [{"type": "line"} for _ in range(41)]
        parsed = {"pages": [{"page_num": 1, "elements": elements}]}
        records = _build_page_records("abc123", parsed)
        assert len(records) == 1
        assert records[0].page_type == "VISUAL"
        assert records[0].dpi_used == DPI_VISUAL

    def test_text_page_at_drawing_threshold(self) -> None:
        """A page with drawings == 40 → TEXT, DPI 100 (not > 40)."""
        elements = [{"type": "line"} for _ in range(40)]
        parsed = {"pages": [{"page_num": 1, "elements": elements}]}
        records = _build_page_records("abc123", parsed)
        assert len(records) == 1
        assert records[0].page_type == "TEXT"
        assert records[0].dpi_used == DPI_TEXT

    def test_page_id_format(self) -> None:
        """Page IDs follow the f'{{doc_id}}#p{{page_num:04d}}' format."""
        parsed = {"pages": [{"page_num": 5, "elements": []}]}
        records = _build_page_records("myhash", parsed)
        assert records[0].page_id == "myhash#p0005"

    def test_doc_id_set_correctly(self) -> None:
        """doc_id is set to the doc_hash."""
        parsed = {"pages": [{"page_num": 1, "elements": []}]}
        records = _build_page_records("the_hash_abc", parsed)
        assert records[0].doc_id == "the_hash_abc"

    def test_page_num_preserved(self) -> None:
        """page_num is preserved from the source data."""
        parsed = {"pages": [{"page_num": 7, "elements": []}]}
        records = _build_page_records("hash", parsed)
        assert records[0].page_num == 7

    def test_multiple_pages(self) -> None:
        """Multiple pages each get their own record."""
        parsed = {
            "pages": [
                {"page_num": 1, "elements": []},
                {"page_num": 2, "elements": [{"type": "image"}]},
                {"page_num": 3, "elements": []},
            ]
        }
        records = _build_page_records("abc", parsed)
        assert len(records) == 3
        assert records[0].page_type == "TEXT"
        assert records[1].page_type == "VISUAL"
        assert records[2].page_type == "TEXT"

    def test_empty_pages_list(self) -> None:
        """An empty pages list returns empty records."""
        parsed: dict = {"pages": []}
        records = _build_page_records("abc", parsed)
        assert records == []

    def test_page_num_as_float(self) -> None:
        """page_num as float (Docling quirk) is coerced to int."""
        parsed = {"pages": [{"page_num": 3.0, "elements": []}]}
        records = _build_page_records("abc", parsed)
        assert records[0].page_num == 3

    def test_missing_page_num_defaults_to_1(self) -> None:
        """A page dict with no page_num defaults to 1."""
        parsed = {"pages": [{"elements": []}]}
        records = _build_page_records("abc", parsed)
        assert records[0].page_num == 1


# ---------------------------------------------------------------------------
# Test classify_pages — integration with filesystem
# ---------------------------------------------------------------------------

class TestClassifyPages:
    """Integration tests for classify_pages (full pipeline)."""

    def test_classify_writes_pages_jsonl(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """classify_pages writes a pages.jsonl file."""
        doc_hash = "test_hash_001"
        dest_dir = isolate_docs_dir / doc_hash
        dest_dir.mkdir(parents=True)
        parsed_json = dest_dir / "parsed.json"
        parsed_json.write_text(
            json.dumps({
                "pages": [
                    {"page_num": 1, "elements": []},
                    {"page_num": 2, "elements": [{"type": "image"}]},
                ]
            })
        )

        records = classify_pages(doc_hash)

        pages_jsonl = dest_dir / "pages.jsonl"
        assert pages_jsonl.exists()
        assert len(records) == 2
        assert records[0].page_type == "TEXT"
        assert records[1].page_type == "VISUAL"

    def test_classify_cache_hit_skips_reclassification(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """If pages.jsonl already exists, it is returned without re-parsing."""
        doc_hash = "cached_hash_002"
        dest_dir = isolate_docs_dir / doc_hash
        dest_dir.mkdir(parents=True)

        # Pre-create pages.jsonl
        pages_jsonl = dest_dir / "pages.jsonl"
        cached_record = {
            "page_id": f"{doc_hash}#p0001",
            "doc_id": doc_hash,
            "page_num": 1,
            "page_type": "TEXT",
            "dpi_used": 100,
            "image_uri": "",
            "extracted_text": "",
            "layout": [],
            "section_id": None,
            "section_title": None,
        }
        pages_jsonl.write_text(json.dumps(cached_record) + "\n")

        # Also create a parsed.json that would give different result
        parsed_json = dest_dir / "parsed.json"
        parsed_json.write_text(
            json.dumps({
                "pages": [{"page_num": 1, "images": [{"path": "/img.png"}], "elements": []}]
            })
        )

        records = classify_pages(doc_hash)

        # Returns cached data (TEXT), not re-parsed data (VISUAL)
        assert len(records) == 1
        assert records[0].page_type == "TEXT"
        assert records[0].dpi_used == DPI_TEXT

    def test_missing_parsed_json_raises_file_not_found_error(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """If parsed.json doesn't exist, classify_pages raises FileNotFoundError."""
        doc_hash = "no_parsed_hash"
        # Create docs dir but not the parsed.json
        (isolate_docs_dir / doc_hash).mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            classify_pages(doc_hash)

    def test_classify_produces_valid_jsonl(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """pages.jsonl contains one valid JSON line per page."""
        doc_hash = "valid_jsonl_hash"
        dest_dir = isolate_docs_dir / doc_hash
        dest_dir.mkdir(parents=True)
        parsed_json = dest_dir / "parsed.json"
        parsed_json.write_text(
            json.dumps({
                "pages": [
                    {"page_num": 1, "elements": []},
                    {"page_num": 2, "elements": []},
                ]
            })
        )

        classify_pages(doc_hash)

        pages_jsonl = dest_dir / "pages.jsonl"
        lines = pages_jsonl.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "page_id" in record
            assert "doc_id" in record
            assert "page_num" in record
            assert "page_type" in record
            assert "dpi_used" in record

    def test_classify_dpi_distribution(
        self, isolate_docs_dir: Path, tmp_path: Path
    ) -> None:
        """DPI distribution in records: TEXT=100, VISUAL=250."""
        doc_hash = "dpi_dist_hash"
        dest_dir = isolate_docs_dir / doc_hash
        dest_dir.mkdir(parents=True)
        parsed_json = dest_dir / "parsed.json"
        # 3 text pages (no visual content), 2 visual pages (have images)
        parsed_json.write_text(
            json.dumps({
                "pages": [
                    {"page_num": 1, "elements": []},  # TEXT
                    {"page_num": 2, "elements": []},  # TEXT
                    {"page_num": 3, "images": [{}], "elements": []},  # VISUAL
                    {"page_num": 4, "elements": [{"type": "rect"}]},  # TEXT (1 drawing)
                    {
                        "page_num": 5,
                        "elements": [{"type": "line"} for _ in range(41)],
                    },  # VISUAL (>40 drawings)
                ]
            })
        )

        records = classify_pages(doc_hash)

        assert len(records) == 5
        text_records = [r for r in records if r.page_type == "TEXT"]
        visual_records = [r for r in records if r.page_type == "VISUAL"]
        assert len(text_records) == 3
        assert len(visual_records) == 2
        for r in text_records:
            assert r.dpi_used == DPI_TEXT
        for r in visual_records:
            assert r.dpi_used == DPI_VISUAL


# ---------------------------------------------------------------------------
# Test _load_pages_jsonl
# ---------------------------------------------------------------------------

class TestLoadPagesJsonl:
    """Tests for the JSONL loader helper."""

    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        """A valid JSONL file loads all records."""
        path = tmp_path / "pages.jsonl"
        records = [
            {
                "page_id": "hash#p0001",
                "doc_id": "hash",
                "page_num": 1,
                "page_type": "TEXT",
                "dpi_used": 100,
                "image_uri": "",
                "extracted_text": "",
                "layout": [],
                "section_id": None,
                "section_title": None,
            },
            {
                "page_id": "hash#p0002",
                "doc_id": "hash",
                "page_num": 2,
                "page_type": "VISUAL",
                "dpi_used": 250,
                "image_uri": "",
                "extracted_text": "",
                "layout": [],
                "section_id": None,
                "section_title": None,
            },
        ]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        loaded = _load_pages_jsonl(path)

        assert len(loaded) == 2
        assert loaded[0].page_type == "TEXT"
        assert loaded[1].page_type == "VISUAL"

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        """Empty lines are skipped during loading."""
        path = tmp_path / "pages.jsonl"
        path.write_text(
            json.dumps({
                "page_id": "h#p0001",
                "doc_id": "h",
                "page_num": 1,
                "page_type": "TEXT",
                "dpi_used": 100,
                "image_uri": "",
                "extracted_text": "",
                "layout": [],
                "section_id": None,
                "section_title": None,
            })
            + "\n\n\n"
        )

        loaded = _load_pages_jsonl(path)

        assert len(loaded) == 1
