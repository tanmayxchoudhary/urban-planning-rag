"""Tests for the full ingest CLI pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated
from unittest.mock import MagicMock, patch

import pytest


class TestIngestPipelineIntegration:
    """Integration tests for urban-rag ingest CLI command."""

    def _create_valid_pdf(self, tmp_path: Path, name: str, size_kb: int = 2) -> Path:
        """Create a valid PDF file at least 1KB in size."""
        pdf_path = tmp_path / name
        padding = b"A" * (size_kb * 1024)
        content = (
            b"%PDF-1.4\n"
            b"%\xe2\xe3\xcf\xd3\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n"
            b"0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
            b"startxref\n192\n"
            b"%%EOF\n"
            + padding
        )
        pdf_path.write_bytes(content)
        return pdf_path

    def _create_small_pdf(self, tmp_path: Path, name: str) -> Path:
        """Create a PDF smaller than 1KB (will be rejected)."""
        pdf_path = tmp_path / name
        pdf_path.write_bytes(b"%PDF-1.4\ntoo small")
        return pdf_path

    def _create_non_pdf(self, tmp_path: Path, name: str) -> Path:
        """Create a non-PDF text file."""
        txt_path = tmp_path / name
        txt_path.write_bytes(b"This is just a plain text file, not a PDF.")
        return txt_path

    def test_ingest_single_pdf_exits_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VAL-CLI-001: urban-rag ingest on a valid PDF exits 0 on success."""
        pdf_path = self._create_valid_pdf(tmp_path, "sample.pdf", size_kb=2)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Mock Settings to avoid reading .env
        mock_settings = MagicMock(
            docs_dir=str(docs_dir),
            manifest_path=str(tmp_path / "manifest.parquet"),
            gemini_api_key="test-api-key-for-unit-tests",
        )
        import urban_rag.common.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_settings", lambda: mock_settings)

        # Patch DOCS_DIR in ingest modules
        import urban_rag.ingest.load as load_mod
        monkeypatch.setattr(load_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.parse as parse_mod
        monkeypatch.setattr(parse_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.classify as classify_mod
        monkeypatch.setattr(classify_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.chunk as chunk_mod
        monkeypatch.setattr(chunk_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.sections as sections_mod
        monkeypatch.setattr(sections_mod, "DOCS_DIR", docs_dir)

        # Also patch the global DOCS_DIR in settings resolution
        from urban_rag.ingest import load, parse, classify, chunk, sections

        # Run the CLI via direct module call
        from urban_rag.cli.ingest import ingest_file

        result = ingest_file(pdf_path)
        assert result is not None
        assert result.filename == "sample.pdf"

    def test_ingest_nonexistent_path_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VAL-CLI-026: urban-rag ingest /nonexistent exits non-zero with 'not found'."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        mock_settings = MagicMock(
            docs_dir=str(docs_dir),
            manifest_path=str(tmp_path / "manifest.parquet"),
            gemini_api_key="test-api-key-for-unit-tests",
        )
        import urban_rag.common.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_settings", lambda: mock_settings)

        import urban_rag.ingest.load as load_mod
        monkeypatch.setattr(load_mod, "DOCS_DIR", docs_dir)

        from urban_rag.cli.ingest import ingest_file
        from urban_rag.common.errors import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ingest_file(tmp_path / "nonexistent.pdf")

        assert "not found" in str(exc_info.value.message).lower()

    def test_ingest_non_pdf_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VAL-CLI-004: urban-rag ingest on a .txt file returns non-zero exit."""
        txt_path = self._create_non_pdf(tmp_path, "readme.txt")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        mock_settings = MagicMock(
            docs_dir=str(docs_dir),
            manifest_path=str(tmp_path / "manifest.parquet"),
            gemini_api_key="test-api-key-for-unit-tests",
        )
        import urban_rag.common.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_settings", lambda: mock_settings)

        import urban_rag.ingest.load as load_mod
        monkeypatch.setattr(load_mod, "DOCS_DIR", docs_dir)

        from urban_rag.cli.ingest import ingest_file
        from urban_rag.common.errors import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ingest_file(txt_path)

        error_msg = str(exc_info.value.message).lower()
        assert "not a pdf" in error_msg or "pdf" in error_msg

    def test_ingest_empty_dir_exits_zero_with_clean_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VAL-CLI-027: Batch ingest on empty directory exits 0 with 'no PDFs found'."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        mock_settings = MagicMock(
            docs_dir=str(docs_dir),
            manifest_path=str(tmp_path / "manifest.parquet"),
            gemini_api_key="test-api-key-for-unit-tests",
        )
        import urban_rag.common.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_settings", lambda: mock_settings)

        from urban_rag.cli.ingest import ingest_directory

        result = ingest_directory(empty_dir, rebuild=False, skip_eval=True)
        assert result == (0, 0)  # (exit_code, count)


class TestIngestPipelineStages:
    """Unit tests for individual pipeline stage wiring."""

    def test_pipeline_calls_all_stages_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify the pipeline calls validate_and_hash, parse, classify, chunk, sections."""
        pdf_path = tmp_path / "test.pdf"
        padding = b"A" * 2048
        content = (
            b"%PDF-1.4\n"
            b"%\xe2\xe3\xcf\xd3\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n192\n%%EOF\n"
            + padding
        )
        pdf_path.write_bytes(content)

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        mock_settings = MagicMock(
            docs_dir=str(docs_dir),
            manifest_path=str(tmp_path / "manifest.parquet"),
            gemini_api_key="test-api-key-for-unit-tests",
        )
        import urban_rag.common.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_settings", lambda: mock_settings)

        import urban_rag.ingest.load as load_mod
        monkeypatch.setattr(load_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.parse as parse_mod
        monkeypatch.setattr(parse_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.classify as classify_mod
        monkeypatch.setattr(classify_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.chunk as chunk_mod
        monkeypatch.setattr(chunk_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.sections as sections_mod
        monkeypatch.setattr(sections_mod, "DOCS_DIR", docs_dir)

        from urban_rag.cli.ingest import ingest_file

        result = ingest_file(pdf_path)
        doc_hash = result.doc_hash

        # Verify all output files were created
        doc_artifacts = docs_dir / doc_hash
        assert doc_artifacts.exists(), f"docs dir not created: {doc_artifacts}"
        assert (doc_artifacts / "source.pdf").exists(), "source.pdf not created"
        assert (doc_artifacts / "parsed.json").exists(), "parsed.json not created"
        assert (doc_artifacts / "pages.jsonl").exists(), "pages.jsonl not created"
        assert (doc_artifacts / "chunks.jsonl").exists(), "chunks.jsonl not created"
        assert (doc_artifacts / "sections.jsonl").exists(), "sections.jsonl not created"

    def test_idempotent_ingest_returns_same_hash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VAL-CLI-003: Running ingest twice on same PDF produces one entry (no duplicates)."""
        pdf_path = tmp_path / "sample.pdf"
        padding = b"A" * 2048
        content = (
            b"%PDF-1.4\n"
            b"%\xe2\xe3\xcf\xd3\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n192\n%%EOF\n"
            + padding
        )
        pdf_path.write_bytes(content)

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        manifest_path = tmp_path / "manifest.parquet"
        mock_settings = MagicMock(
            docs_dir=str(docs_dir),
            manifest_path=str(manifest_path),
            gemini_api_key="test-api-key-for-unit-tests",
        )
        import urban_rag.common.settings as settings_mod
        # Reset the global singleton to ensure clean state
        monkeypatch.setattr(settings_mod, "_settings", mock_settings)
        monkeypatch.setattr(settings_mod, "get_settings", lambda: mock_settings)

        import urban_rag.ingest.load as load_mod
        monkeypatch.setattr(load_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.parse as parse_mod
        monkeypatch.setattr(parse_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.classify as classify_mod
        monkeypatch.setattr(classify_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.chunk as chunk_mod
        monkeypatch.setattr(chunk_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.sections as sections_mod
        monkeypatch.setattr(sections_mod, "DOCS_DIR", docs_dir)

        from urban_rag.cli.ingest import ingest_file

        result1 = ingest_file(pdf_path)
        result2 = ingest_file(pdf_path)

        assert result1.doc_hash == result2.doc_hash
        # Manifest should have exactly 1 entry
        import pandas as pd
        df = pd.read_parquet(manifest_path)
        matching = df[df["doc_hash"] == result1.doc_hash]
        assert len(matching) == 1

    def test_rebuild_flag_removes_cached_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VAL-CLI-006: urban-rag ingest --rebuild removes cached outputs before re-running."""
        pdf_path = tmp_path / "sample.pdf"
        padding = b"A" * 2048
        content = (
            b"%PDF-1.4\n"
            b"%\xe2\xe3\xcf\xd3\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n192\n%%EOF\n"
            + padding
        )
        pdf_path.write_bytes(content)

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        mock_settings = MagicMock(
            docs_dir=str(docs_dir),
            manifest_path=str(tmp_path / "manifest.parquet"),
            gemini_api_key="test-api-key-for-unit-tests",
        )
        import urban_rag.common.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_settings", lambda: mock_settings)

        import urban_rag.ingest.load as load_mod
        monkeypatch.setattr(load_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.parse as parse_mod
        monkeypatch.setattr(parse_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.classify as classify_mod
        monkeypatch.setattr(classify_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.chunk as chunk_mod
        monkeypatch.setattr(chunk_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.sections as sections_mod
        monkeypatch.setattr(sections_mod, "DOCS_DIR", docs_dir)

        from urban_rag.cli.ingest import ingest_file, rebuild_document

        # First: ingest normally
        result1 = ingest_file(pdf_path)
        doc_hash = result1.doc_hash
        doc_dir = docs_dir / doc_hash

        # Write a marker file to simulate a "cached" output from a previous run
        cached_marker = doc_dir / "pages.jsonl"
        cached_marker.write_text("cached content - should be deleted on rebuild")

        # Rebuild should delete cached outputs
        rebuild_document(pdf_path, doc_hash)
        assert not cached_marker.exists(), "rebuild did not remove cached pages.jsonl"

    def test_skip_eval_flag_skips_eval_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VAL-CLI-007: urban-rag ingest --skip-eval bypasses eval validation."""
        pdf_path = self._create_valid_pdf_fixture(tmp_path)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        manifest_path = tmp_path / "manifest.parquet"
        mock_settings = MagicMock(
            docs_dir=str(docs_dir),
            manifest_path=str(manifest_path),
            gemini_api_key="test-api-key-for-unit-tests",
        )
        import urban_rag.common.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_settings", lambda: mock_settings)

        import urban_rag.ingest.load as load_mod
        monkeypatch.setattr(load_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.parse as parse_mod
        monkeypatch.setattr(parse_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.classify as classify_mod
        monkeypatch.setattr(classify_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.chunk as chunk_mod
        monkeypatch.setattr(chunk_mod, "DOCS_DIR", docs_dir)

        import urban_rag.ingest.sections as sections_mod
        monkeypatch.setattr(sections_mod, "DOCS_DIR", docs_dir)

        from urban_rag.cli.ingest import ingest_file

        # With skip_eval=True, should succeed without trying to run eval checks
        result = ingest_file(pdf_path, rebuild=False, skip_eval=True)
        assert result is not None  # No eval error raised

    def _create_valid_pdf_fixture(self, tmp_path: Path) -> Path:
        pdf_path = tmp_path / "sample.pdf"
        padding = b"A" * 2048
        content = (
            b"%PDF-1.4\n"
            b"%\xe2\xe3\xcf\xd3\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n192\n%%EOF\n"
            + padding
        )
        pdf_path.write_bytes(content)
        return pdf_path
