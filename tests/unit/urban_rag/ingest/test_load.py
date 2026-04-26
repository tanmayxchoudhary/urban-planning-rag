"""Unit tests for ingest/load.py — validate_and_hash, content-addressed manifest."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure test environment has required settings
os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-unit-tests")

from urban_rag.common.errors import ValidationError
from urban_rag.ingest.load import validate_and_hash


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

MIN_PDF_SIZE = 1024  # 1 KB threshold


def _make_pdf(path: Path, content: bytes) -> bytes:
    """Write a valid minimal PDF with given content bytes and return them."""
    path.write_bytes(content)
    return content


def _create_large_pdf(path: Path) -> bytes:
    """Create a valid PDF (>1KB) with enough content to exceed MIN_PDF_SIZE."""
    padding = b"A" * 1500
    pdf_content = (
        b"%PDF-1.4\n"
        b"%\xe2\xe3\xcf\xd3\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\n"
        b"endobj\n"
        b"xref\n"
        b"0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n"
        b"<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n"
        b"192\n"
        b"%%EOF\n"
        + padding
    )
    return _make_pdf(path, pdf_content)


class TestValidateAndHash:
    """Tests for validate_and_hash()."""

    @pytest.fixture(autouse=True)
    def isolated_docs_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> Path:
        """Isolate tests with a temporary docs directory.

        We patch DOCS_DIR directly in the load module and reset the settings
        singleton so that validate_and_hash uses the temp directory.
        """
        test_docs = tmp_path / "docs"
        test_docs.mkdir()
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key-for-unit-tests")

        # Reset the settings singleton so it picks up env vars fresh
        import urban_rag.common.settings as settings_module

        settings_module._settings = None

        # Patch DOCS_DIR directly in the load module
        import urban_rag.ingest.load as load_module

        monkeypatch.setattr(load_module, "DOCS_DIR", test_docs)

        # Also patch settings' docs_dir to return our temp path
        # (so _get_docs_dir returns the right thing)
        original_settings_init = settings_module.Settings.__init__

        def patched_init(self: settings_module.Settings) -> None:
            original_settings_init(self)
            object.__setattr__(self, "docs_dir", str(test_docs))

        monkeypatch.setattr(
            settings_module.Settings, "__init__", patched_init
        )

        # Reset DOCS_DIR so _get_docs_dir re-resolves
        load_module.DOCS_DIR = test_docs

        return test_docs

    def test_valid_pdf_returns_document_record(
        self, isolated_docs_dir: Path, tmp_path: Path
    ) -> None:
        """A valid PDF > 1KB returns a DocumentRecord with the correct hash."""
        pdf_path = tmp_path / "test_doc.pdf"
        file_bytes = _create_large_pdf(pdf_path)

        result = validate_and_hash(pdf_path)

        assert result is not None
        assert result.doc_hash == hashlib.sha256(file_bytes).hexdigest()
        assert result.filename == "test_doc.pdf"
        assert result.page_count >= 0  # PyMuPDF may be absent in test env
        assert result.size_bytes == len(file_bytes)
        assert result.version == "1"

    def test_valid_pdf_writes_content_addressed_file(
        self, isolated_docs_dir: Path, tmp_path: Path
    ) -> None:
        """The source PDF is written to docs/<hash>/source.pdf."""
        pdf_path = tmp_path / "test_doc.pdf"
        file_bytes = _create_large_pdf(pdf_path)
        expected_hash = hashlib.sha256(file_bytes).hexdigest()

        validate_and_hash(pdf_path)

        dest = isolated_docs_dir / expected_hash / "source.pdf"
        assert dest.exists(), f"Expected {dest} to exist"
        assert dest.read_bytes() == file_bytes

    def test_idempotent_same_pdf_produces_one_manifest_entry(
        self, isolated_docs_dir: Path, tmp_path: Path
    ) -> None:
        """Calling validate_and_hash twice on the same PDF produces no duplicate."""
        pdf_path = tmp_path / "test_doc.pdf"
        _create_large_pdf(pdf_path)

        result1 = validate_and_hash(pdf_path)
        result2 = validate_and_hash(pdf_path)

        # Same hash returned both times
        assert result1.doc_hash == result2.doc_hash
        # docs/ directory contains exactly one hash entry
        hash_dirs = [d for d in isolated_docs_dir.iterdir() if d.is_dir()]
        assert len(hash_dirs) == 1
        assert hash_dirs[0].name == result1.doc_hash

    def test_rejects_non_pdf_file(
        self, isolated_docs_dir: Path, tmp_path: Path
    ) -> None:
        """A .txt or other non-PDF file raises ValidationError."""
        txt_path = tmp_path / "notapdf.txt"
        txt_path.write_text("This is plain text, not a PDF.")

        with pytest.raises(ValidationError) as exc_info:
            validate_and_hash(txt_path)

        assert "not a PDF" in str(exc_info.value.message).lower() or "pdf" in str(
            exc_info.value.message
        ).lower()

    def test_rejects_pdf_smaller_than_1kb(
        self, isolated_docs_dir: Path, tmp_path: Path
    ) -> None:
        """A PDF with size < 1 KB is rejected with a size-related ValidationError."""
        # Create a minimal PDF using bytes that look like a PDF header
        # but is definitely smaller than 1KB
        small_content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n" + b"X" * 100
        small_pdf = tmp_path / "tiny.pdf"
        small_pdf.write_bytes(small_content)

        with pytest.raises(ValidationError) as exc_info:
            validate_and_hash(small_pdf)

        assert "too small" in str(exc_info.value.message).lower() or "size" in str(
            exc_info.value.message
        ).lower()

    def test_rejects_nonexistent_path(
        self, isolated_docs_dir: Path, tmp_path: Path
    ) -> None:
        """A non-existent file path raises ValidationError with 'not found'."""
        nonexistent = tmp_path / "does_not_exist.pdf"

        with pytest.raises(ValidationError) as exc_info:
            validate_and_hash(nonexistent)

        assert "not found" in str(exc_info.value.message).lower() or "exist" in str(
            exc_info.value.message
        ).lower()

    def test_correct_sha256_used_as_content_key(
        self, isolated_docs_dir: Path, tmp_path: Path
    ) -> None:
        """The SHA256 of file bytes (not path/name) is used as the content key."""
        pdf_path = tmp_path / "renamed_later.pdf"
        file_bytes = _create_large_pdf(pdf_path)
        expected_hash = hashlib.sha256(file_bytes).hexdigest()

        result = validate_and_hash(pdf_path)

        assert result.doc_hash == expected_hash
        # Verify by reading the bytes back and re-hashing
        stored_bytes = (isolated_docs_dir / expected_hash / "source.pdf").read_bytes()
        assert hashlib.sha256(stored_bytes).hexdigest() == expected_hash

    def test_two_different_pdfs_produce_different_hashes(
        self, isolated_docs_dir: Path, tmp_path: Path
    ) -> None:
        """Two distinct PDFs produce distinct content hashes."""
        pdf1_path = tmp_path / "doc1.pdf"
        pdf2_path = tmp_path / "doc2.pdf"

        # Create two PDFs with different byte content
        content1 = (
            b"%PDF-1.4\n"
            b"%\xe2\xe3\xcf\xd3\n"
            b"1 0 obj\n"
            b"<< /Type /Catalog /Pages 2 0 R >>\n"
            b"endobj\n"
            b"2 0 obj\n"
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
            b"endobj\n"
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\n"
            b"endobj\n"
            b"xref\n"
            b"0 4\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"trailer\n"
            b"<< /Size 4 /Root 1 0 R >>\n"
            b"startxref\n"
            b"192\n"
            b"%%EOF\n"
            + b"DOC1_CONTENT_IDENTIFIER" + b"A" * 1400
        )
        content2 = (
            b"%PDF-1.4\n"
            b"%\xe2\xe3\xcf\xd3\n"
            b"1 0 obj\n"
            b"<< /Type /Catalog /Pages 2 0 R >>\n"
            b"endobj\n"
            b"2 0 obj\n"
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
            b"endobj\n"
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\n"
            b"endobj\n"
            b"xref\n"
            b"0 4\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"trailer\n"
            b"<< /Size 4 /Root 1 0 R >>\n"
            b"startxref\n"
            b"192\n"
            b"%%EOF\n"
            + b"DOC2_CONTENT_IDENTIFIER" + b"B" * 1400
        )

        _make_pdf(pdf1_path, content1)
        _make_pdf(pdf2_path, content2)

        result1 = validate_and_hash(pdf1_path)
        result2 = validate_and_hash(pdf2_path)

        assert result1.doc_hash != result2.doc_hash
        # Both stored separately
        assert (isolated_docs_dir / result1.doc_hash).exists()
        assert (isolated_docs_dir / result2.doc_hash).exists()
