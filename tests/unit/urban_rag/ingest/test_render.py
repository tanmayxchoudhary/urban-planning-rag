"""Unit tests for src/urban_rag/ingest/render.py — PDF page rendering stage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-unit-tests")

from urban_rag.ingest.render import (
    DPI_TEXT,
    DPI_VISUAL,
    _render_page,
    render_document,
    sanitize_filename,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate tests with temporary docs and page_images directories."""
    test_docs = tmp_path / "docs"
    test_page_images = tmp_path / "page_images"
    test_docs.mkdir()
    test_page_images.mkdir()

    # Reset settings singleton
    import urban_rag.common.settings as settings_module

    settings_module._settings = None

    original_init = settings_module.Settings.__init__

    def patched_init(self: settings_module.Settings) -> None:
        original_init(self)
        object.__setattr__(self, "docs_dir", str(test_docs))
        object.__setattr__(self, "page_images_dir", str(test_page_images))

    monkeypatch.setattr(settings_module.Settings, "__init__", patched_init)

    # Patch render.py DOCS_DIR and PAGE_IMAGES_DIR
    import urban_rag.ingest.render as render_module

    render_module.DOCS_DIR = test_docs
    render_module.PAGE_IMAGES_DIR = test_page_images

    return tmp_path


# ---------------------------------------------------------------------------
# Test DPI constants
# ---------------------------------------------------------------------------

class TestDpiConstants:
    """Verify DPI constants match classify.py."""

    def test_dpi_text_is_100(self) -> None:
        assert DPI_TEXT == 100

    def test_dpi_visual_is_250(self) -> None:
        assert DPI_VISUAL == 250


# ---------------------------------------------------------------------------
# Test sanitize_filename
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_keeps_normal_filenames(self) -> None:
        assert sanitize_filename("report.pdf") == "report"

    def test_replaces_spaces_with_underscores(self) -> None:
        assert sanitize_filename("my report.pdf") == "my_report"

    def test_removes_multiple_underscores(self) -> None:
        assert sanitize_filename("my__report.pdf") == "my_report"

    def test_removes_trailing_underscores(self) -> None:
        assert sanitize_filename("report_.pdf") == "report"

    def test_removes_special_chars(self) -> None:
        # Hyphens are also replaced (only alphanumeric, underscore, hyphen allowed)
        result = sanitize_filename("report-2024_final.pdf")
        assert result == "report_2024_final"

    def test_strips_leading_underscores(self) -> None:
        assert sanitize_filename("_report.pdf") == "report"

    def test_empty_after_sanitize_returns_source(self) -> None:
        assert sanitize_filename(".pdf") == "source"

    def test_unicode_chars_removed(self) -> None:
        assert sanitize_filename("文档.pdf") == "source"


# ---------------------------------------------------------------------------
# Test _render_page
# ---------------------------------------------------------------------------

class TestRenderPage:
    """Tests for the low-level page rendering helper."""

    def test_renders_page_to_png_at_text_dpi(self, tmp_path: Path) -> None:
        """A TEXT page is rendered at 100 DPI."""
        pdf_path = tmp_path / "test.pdf"
        _create_simple_pdf(pdf_path)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result_path = _render_page(
            pdf_path, page_num=1, dpi=DPI_TEXT, output_dir=output_dir, base_name="test"
        )

        assert result_path.exists()
        assert result_path.name.startswith("test__page_0001")
        assert result_path.suffix == ".png"

    def test_renders_page_to_png_at_visual_dpi(self, tmp_path: Path) -> None:
        """A VISUAL page is rendered at 250 DPI."""
        pdf_path = tmp_path / "test.pdf"
        _create_simple_pdf(pdf_path)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result_path = _render_page(
            pdf_path, page_num=1, dpi=DPI_VISUAL, output_dir=output_dir, base_name="test"
        )

        assert result_path.exists()

    def test_output_path_format_is_correct(self, tmp_path: Path) -> None:
        """Output filename follows {base}__page_{pagenum:04d}.png pattern."""
        pdf_path = tmp_path / "test.pdf"
        _create_simple_pdf(pdf_path, pages=5)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result_path = _render_page(
            pdf_path, page_num=5, dpi=DPI_TEXT, output_dir=output_dir, base_name="mydoc"
        )

        assert result_path.name == "mydoc__page_0005.png"

    def test_page_number_padding_is_4_digits(self, tmp_path: Path) -> None:
        """Page numbers are zero-padded to 4 digits."""
        pdf_path = tmp_path / "test.pdf"
        _create_simple_pdf(pdf_path, pages=12)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        for page_num in [1, 5, 10, 12]:
            result_path = _render_page(
                pdf_path, page_num=page_num, dpi=DPI_TEXT, output_dir=output_dir, base_name="doc"
            )
            # Expected filename: doc__page_XXXX.png where XXXX is zero-padded
            assert result_path.name == f"doc__page_{page_num:04d}.png"

    def test_nonexistent_page_raises(self, tmp_path: Path) -> None:
        """Requesting a page beyond PDF page count raises IndexError."""
        pdf_path = tmp_path / "test.pdf"
        _create_simple_pdf(pdf_path, pages=2)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(IndexError):
            _render_page(
                pdf_path, page_num=99, dpi=DPI_TEXT, output_dir=output_dir, base_name="test"
            )


# ---------------------------------------------------------------------------
# Test render_document
# ---------------------------------------------------------------------------

class TestRenderDocument:
    """Integration tests for the full document render pipeline."""

    def test_renders_all_pages_from_pages_jsonl(self, isolate_settings: Path, tmp_path: Path) -> None:
        """render_document reads pages.jsonl and renders each page at the correct DPI."""
        # Setup: create doc dir with source.pdf and pages.jsonl
        doc_hash = "abc123def456"
        doc_dir = isolate_settings / "docs" / doc_hash
        doc_dir.mkdir(parents=True)

        # Create a 3-page PDF
        pdf_path = doc_dir / "source.pdf"
        _create_simple_pdf(pdf_path, pages=3)

        # Create pages.jsonl: 2 TEXT (100 DPI), 1 VISUAL (250 DPI)
        pages = [
            {
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
            },
            {
                "page_id": f"{doc_hash}#p0002",
                "doc_id": doc_hash,
                "page_num": 2,
                "page_type": "VISUAL",
                "dpi_used": 250,
                "image_uri": "",
                "extracted_text": "",
                "layout": [],
                "section_id": None,
                "section_title": None,
            },
            {
                "page_id": f"{doc_hash}#p0003",
                "doc_id": doc_hash,
                "page_num": 3,
                "page_type": "TEXT",
                "dpi_used": 100,
                "image_uri": "",
                "extracted_text": "",
                "layout": [],
                "section_id": None,
                "section_title": None,
            },
        ]
        pages_jsonl = doc_dir / "pages.jsonl"
        pages_jsonl.write_text("\n".join(json.dumps(p) for p in pages) + "\n")

        page_images_dir = isolate_settings / "page_images"
        page_images_dir.mkdir(parents=True, exist_ok=True)

        updated_records = render_document(doc_hash)

        # All 3 pages should be rendered
        assert len(updated_records) == 3

        # Check PNG files exist
        for i, record in enumerate(updated_records, start=1):
            assert record.image_uri != ""
            png_path = Path(record.image_uri)
            assert png_path.exists(), f"PNG not found: {png_path}"

        # VISUAL page (page 2) should have dpi_used=250; TEXT pages = 100
        visual_record = next(r for r in updated_records if r.page_num == 2)
        text_record = next(r for r in updated_records if r.page_num == 1)

        assert visual_record.dpi_used == 250
        assert visual_record.page_type == "VISUAL"
        assert text_record.dpi_used == 100
        assert text_record.page_type == "TEXT"

    def test_pages_jsonl_is_updated_with_image_uris(self, isolate_settings: Path, tmp_path: Path) -> None:
        """After rendering, pages.jsonl contains image_uri paths for all pages."""
        doc_hash = "xyz789abc123"
        doc_dir = isolate_settings / "docs" / doc_hash
        doc_dir.mkdir(parents=True)

        pdf_path = doc_dir / "source.pdf"
        _create_simple_pdf(pdf_path, pages=2)

        pages = [
            {
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
            },
            {
                "page_id": f"{doc_hash}#p0002",
                "doc_id": doc_hash,
                "page_num": 2,
                "page_type": "TEXT",
                "dpi_used": 100,
                "image_uri": "",
                "extracted_text": "",
                "layout": [],
                "section_id": None,
                "section_title": None,
            },
        ]
        pages_jsonl = doc_dir / "pages.jsonl"
        pages_jsonl.write_text("\n".join(json.dumps(p) for p in pages) + "\n")

        page_images_dir = isolate_settings / "page_images"
        page_images_dir.mkdir(parents=True, exist_ok=True)

        render_document(doc_hash)

        # Re-read pages.jsonl from disk
        lines = pages_jsonl.read_text().strip().split("\n")
        updated_pages = [json.loads(line) for line in lines if line.strip()]

        assert len(updated_pages) == 2
        for page in updated_pages:
            assert page["image_uri"] != ""
            assert Path(page["image_uri"]).exists()

    def test_source_pdf_not_in_page_images_dir(self, isolate_settings: Path, tmp_path: Path) -> None:
        """Rendered PNGs go to page_images dir, not the docs hash dir."""
        doc_hash = "pngtest123"
        doc_dir = isolate_settings / "docs" / doc_hash
        doc_dir.mkdir(parents=True)

        pdf_path = doc_dir / "source.pdf"
        _create_simple_pdf(pdf_path, pages=1)

        pages = [
            {
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
            },
        ]
        pages_jsonl = doc_dir / "pages.jsonl"
        pages_jsonl.write_text(json.dumps(pages[0]) + "\n")

        page_images_dir = isolate_settings / "page_images"
        page_images_dir.mkdir(parents=True, exist_ok=True)

        render_document(doc_hash)

        # No PNGs should be in the docs hash directory
        pngs_in_doc_dir = list(doc_dir.glob("*.png"))
        assert len(pngs_in_doc_dir) == 0

        # PNGs should be in page_images_dir
        pngs_in_page_images = list(page_images_dir.glob("*.png"))
        assert len(pngs_in_page_images) == 1

    def test_missing_source_pdf_raises_file_not_found(self, isolate_settings: Path, tmp_path: Path) -> None:
        """If source.pdf doesn't exist in the doc hash dir, raise FileNotFoundError."""
        doc_hash = "nosource123"
        doc_dir = isolate_settings / "docs" / doc_hash
        doc_dir.mkdir(parents=True)

        # Create pages.jsonl but no source.pdf
        pages = [
            {
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
            },
        ]
        pages_jsonl = doc_dir / "pages.jsonl"
        pages_jsonl.write_text(json.dumps(pages[0]) + "\n")

        page_images_dir = isolate_settings / "page_images"
        page_images_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError):
            render_document(doc_hash)

    def test_missing_pages_jsonl_raises_file_not_found(self, isolate_settings: Path, tmp_path: Path) -> None:
        """If pages.jsonl doesn't exist, raise FileNotFoundError."""
        doc_hash = "nopages123"
        doc_dir = isolate_settings / "docs" / doc_hash
        doc_dir.mkdir(parents=True)

        # Create source.pdf but no pages.jsonl
        pdf_path = doc_dir / "source.pdf"
        _create_simple_pdf(pdf_path, pages=1)

        page_images_dir = isolate_settings / "page_images"
        page_images_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError):
            render_document(doc_hash)

    def test_idempotent_rerender_updates_image_uri(self, isolate_settings: Path, tmp_path: Path) -> None:
        """Re-running render_document on already-rendered doc updates image_uri."""
        doc_hash = "rerender123"
        doc_dir = isolate_settings / "docs" / doc_hash
        doc_dir.mkdir(parents=True)

        pdf_path = doc_dir / "source.pdf"
        _create_simple_pdf(pdf_path, pages=1)

        pages = [
            {
                "page_id": f"{doc_hash}#p0001",
                "doc_id": doc_hash,
                "page_num": 1,
                "page_type": "TEXT",
                "dpi_used": 100,
                "image_uri": "",  # Empty — simulate pre-render state
                "extracted_text": "",
                "layout": [],
                "section_id": None,
                "section_title": None,
            },
        ]
        pages_jsonl = doc_dir / "pages.jsonl"
        pages_jsonl.write_text(json.dumps(pages[0]) + "\n")

        page_images_dir = isolate_settings / "page_images"
        page_images_dir.mkdir(parents=True, exist_ok=True)

        # First render
        records1 = render_document(doc_hash)
        assert len(records1) == 1
        uri1 = records1[0].image_uri

        # Second render — should also succeed (idempotent)
        records2 = render_document(doc_hash)
        assert len(records2) == 1
        uri2 = records2[0].image_uri

        # URI should be stable
        assert uri1 == uri2

    def test_image_uri_path_format(self, isolate_settings: Path, tmp_path: Path) -> None:
        """image_uri follows the pattern data/page_images/{sanitized}__page_{num:04d}.png."""
        doc_hash = "pathformat123"
        doc_dir = isolate_settings / "docs" / doc_hash
        doc_dir.mkdir(parents=True)

        pdf_path = doc_dir / "source.pdf"
        _create_simple_pdf(pdf_path, pages=1)

        pages = [
            {
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
            },
        ]
        pages_jsonl = doc_dir / "pages.jsonl"
        pages_jsonl.write_text(json.dumps(pages[0]) + "\n")

        page_images_dir = isolate_settings / "page_images"
        page_images_dir.mkdir(parents=True, exist_ok=True)

        records = render_document(doc_hash)
        image_uri = records[0].image_uri

        # Should be an absolute path or relative starting with data/page_images/
        assert "page_images" in image_uri
        assert "__page_" in image_uri
        assert image_uri.endswith(".png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_simple_pdf(path: Path, pages: int = 1) -> None:
    """Create a simple multi-page PDF using PyMuPDF for testing."""
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((50, 50), f"Test page {i + 1}\n" * 20, fontsize=10)
    doc.save(str(path))
    doc.close()
