"""Tests for ingest CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from urban_rag.common.errors import ValidationError


if TYPE_CHECKING:
    from pathlib import Path


class TestIngestFile:
    """Tests for ingest_file() function."""

    def test_ingest_nonexistent_path_raises_validation_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VAL-CLI-026: Ingest rejects nonexistent PDF path with ValidationError.

        A non-existent file path should raise ValidationError with 'not found' message.
        """
        # Ensure the path definitely does not exist
        nonexistent = tmp_path / "does_not_exist.pdf"
        assert not nonexistent.exists()

        # Patch DOCS_DIR in the load module to avoid actual filesystem operations
        import urban_rag.ingest.load as load_module

        test_docs = tmp_path / "docs"
        test_docs.mkdir()
        monkeypatch.setattr(load_module, "DOCS_DIR", test_docs)

        # Also ensure settings are patched
        import urban_rag.common.settings as settings_module

        monkeypatch.setattr(
            settings_module.Settings,
            "__init__",
            lambda self: None,
        )
        monkeypatch.setattr(
            settings_module,
            "get_settings",
            lambda: MagicMock(docs_dir=str(test_docs)),
        )

        # Import after patching
        from urban_rag.cli.ingest import ingest_file

        with pytest.raises(ValidationError) as exc_info:
            ingest_file(nonexistent)

        assert "not found" in str(exc_info.value.message).lower()

    def test_ingest_valid_pdf_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid PDF > 1KB returns a DocumentRecord successfully."""
        import hashlib

        # Create a valid PDF
        pdf_path = tmp_path / "test_doc.pdf"
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
        pdf_path.write_bytes(pdf_content)

        # Patch DOCS_DIR in the load module
        import urban_rag.ingest.load as load_module

        test_docs = tmp_path / "docs"
        test_docs.mkdir()
        monkeypatch.setattr(load_module, "DOCS_DIR", test_docs)

        # Also ensure settings are patched
        import urban_rag.common.settings as settings_module

        monkeypatch.setattr(
            settings_module.Settings,
            "__init__",
            lambda self: None,
        )
        monkeypatch.setattr(
            settings_module,
            "get_settings",
            lambda: MagicMock(docs_dir=str(test_docs)),
        )

        # Import after patching
        from urban_rag.cli.ingest import ingest_file

        result = ingest_file(pdf_path)

        assert result is not None
        assert result.doc_hash == hashlib.sha256(pdf_content).hexdigest()
        assert result.filename == "test_doc.pdf"
        assert result.size_bytes == len(pdf_content)
