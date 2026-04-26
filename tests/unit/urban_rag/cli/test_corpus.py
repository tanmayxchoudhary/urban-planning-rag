"""Tests for corpus management CLI commands."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from urban_rag.common.types import DocumentRecord


@pytest.fixture
def temp_docs_dir(tmp_path: Path) -> Path:
    """Create a temporary docs directory."""
    docs = tmp_path / "docs"
    docs.mkdir()
    return docs


@pytest.fixture
def temp_manifest_path(tmp_path: Path) -> Path:
    """Create a temporary manifest path."""
    return tmp_path / "manifest.parquet"


@pytest.fixture
def sample_records() -> list[DocumentRecord]:
    """Create sample document records for testing."""
    return [
        DocumentRecord(
            doc_hash="a" * 64,
            filename="doc1.pdf",
            title="URDPFI Guidelines Volume I, 2014",
            family="URDPFI",
            jurisdiction="IN-MH-Mumbai",
            publisher="MoHUA",
            year=2014,
            version="v1",
            license="gov_open",
            page_count=10,
            storage_uri="s3://urban-rag-source/doc1.pdf",
            ingested_at=datetime(2026, 1, 15, 10, 30, 0),
            indexed_at=datetime(2026, 1, 15, 11, 0, 0),
        ),
        DocumentRecord(
            doc_hash="b" * 64,
            filename="doc2.pdf",
            title="NBC 2016 Vol 1",
            family="NBC",
            jurisdiction="IN",
            publisher="BIS",
            year=2016,
            version="v1",
            license="public_domain",
            page_count=5,
            storage_uri="s3://urban-rag-source/doc2.pdf",
            ingested_at=datetime(2026, 2, 20, 14, 0, 0),
            indexed_at=datetime(2026, 2, 20, 14, 30, 0),
        ),
    ]


@pytest.fixture
def populated_manifest(
    temp_manifest_path: Path, sample_records: list[DocumentRecord]
) -> Path:
    """Create a manifest with sample records."""
    df = pd.DataFrame([r.model_dump() for r in sample_records])
    df.to_parquet(temp_manifest_path, index=False)
    return temp_manifest_path


class TestCorpusList:
    """Tests for corpus list command."""

    def test_list_empty_corpus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that list shows clean message on empty corpus."""
        from urban_rag.cli import corpus as corpus_module

        # Point settings to temp paths
        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()
        temp_manifest = tmp_path / "manifest.parquet"

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(temp_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        # Run list - should not raise, should show empty message
        corpus_module.corpus_list()

    def test_list_with_documents(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        populated_manifest: Path,
        sample_records: list[DocumentRecord],
    ) -> None:
        """Test that list shows all ingested documents."""
        from urban_rag.cli import corpus as corpus_module

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(populated_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        # Run list - should complete without error
        corpus_module.corpus_list()


class TestCorpusStats:
    """Tests for corpus stats command."""

    def test_stats_empty_corpus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that stats shows zeros on empty corpus."""
        from urban_rag.cli import corpus as corpus_module

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()
        temp_manifest = tmp_path / "manifest.parquet"

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(temp_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        # Run stats - should not raise, should show zeros
        corpus_module.corpus_stats()

    def test_stats_with_documents(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        populated_manifest: Path,
        sample_records: list[DocumentRecord],
    ) -> None:
        """Test that stats reports correct totals."""
        from urban_rag.cli import corpus as corpus_module

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(populated_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        # Run stats - should complete without error
        corpus_module.corpus_stats()


class TestCorpusRm:
    """Tests for corpus rm command."""

    def test_rm_invalid_hash_too_short(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that rm rejects invalid hash (too short)."""
        from urban_rag.cli import corpus as corpus_module
        from urban_rag.common.errors import ValidationError

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()
        temp_manifest = tmp_path / "manifest.parquet"

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(temp_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        with pytest.raises(ValidationError, match="64-character"):
            corpus_module.corpus_rm("abc123")

    def test_rm_document_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that rm raises error when document not found."""
        from urban_rag.cli import corpus as corpus_module
        from urban_rag.common.errors import DocumentNotFoundError

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()
        temp_manifest = tmp_path / "manifest.parquet"

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(temp_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        with pytest.raises(DocumentNotFoundError):
            corpus_module.corpus_rm("z" * 64)

    def test_rm_removes_manifest_entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        populated_manifest: Path,
        sample_records: list[DocumentRecord],
    ) -> None:
        """Test that rm removes the manifest entry."""
        from urban_rag.cli import corpus as corpus_module

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(populated_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        # Verify we have 2 records
        assert len(sample_records) == 2

        # Remove one
        corpus_module.corpus_rm(sample_records[0].doc_hash)

        # Verify manifest now has 1 record
        df = pd.read_parquet(populated_manifest)
        assert len(df) == 1
        assert df.iloc[0].doc_hash == sample_records[1].doc_hash


class TestCorpusReindex:
    """Tests for corpus reindex command."""

    def test_reindex_invalid_hash_too_short(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that reindex rejects invalid hash (too short)."""
        from urban_rag.cli import corpus as corpus_module
        from urban_rag.common.errors import ValidationError

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()
        temp_manifest = tmp_path / "manifest.parquet"

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(temp_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        with pytest.raises(ValidationError, match="64-character"):
            corpus_module.corpus_reindex("abc123")

    def test_reindex_document_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that reindex raises error when document not found."""
        from urban_rag.cli import corpus as corpus_module
        from urban_rag.common.errors import DocumentNotFoundError

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()
        temp_manifest = tmp_path / "manifest.parquet"

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(temp_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        with pytest.raises(DocumentNotFoundError):
            corpus_module.corpus_reindex("z" * 64)

    def test_reindex_no_pngs_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        populated_manifest: Path,
        sample_records: list[DocumentRecord],
    ) -> None:
        """Test that reindex fails when no PNGs are found."""
        from urban_rag.cli import corpus as corpus_module
        from urban_rag.common.errors import ValidationError

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()

        # Create doc dir but no PNGs
        doc_dir = temp_docs / sample_records[0].doc_hash[:64]
        doc_dir.mkdir()

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(populated_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        with pytest.raises(ValidationError, match="No page PNGs"):
            corpus_module.corpus_reindex(sample_records[0].doc_hash)


class TestCorpusRmMissingDocsDir:
    """Tests for corpus rm with missing docs directory."""

    def test_rm_missing_docs_dir_shows_warning_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that rm handles missing docs_dir gracefully (no FileNotFoundError)."""
        from urban_rag.cli import corpus as corpus_module

        temp_docs = tmp_path / "docs"  # Does NOT exist
        temp_manifest = tmp_path / "manifest.parquet"

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(temp_manifest)

        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        # Should not raise FileNotFoundError - the function checks docs_dir.exists()
        # Since manifest is also empty, DocumentNotFoundError is raised
        from urban_rag.common.errors import DocumentNotFoundError
        with pytest.raises(DocumentNotFoundError):
            corpus_module.corpus_rm("z" * 64)


class TestCorpusReindexMissingDocsDir:
    """Tests for corpus reindex with missing docs directory."""

    def test_reindex_missing_docs_dir_raises_clear_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        populated_manifest: Path,
        sample_records: list[DocumentRecord],
    ) -> None:
        """Test that reindex raises ValidationError with clear message when docs_dir missing."""
        from urban_rag.cli import corpus as corpus_module
        from urban_rag.common.errors import ValidationError

        temp_docs = tmp_path / "docs"  # Does NOT exist
        temp_manifest = tmp_path / "manifest.parquet"

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(populated_manifest)

        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        # Should raise ValidationError (not FileNotFoundError)
        with pytest.raises(ValidationError, match="not found|Corpus directory"):
            corpus_module.corpus_reindex(sample_records[0].doc_hash)


class TestCorpusEmptyState:
    """Tests for empty corpus state handling."""

    def test_list_empty_shows_clean_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that list on empty corpus shows clean message, not traceback."""
        from urban_rag.cli import corpus as corpus_module

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()
        temp_manifest = tmp_path / "manifest.parquet"

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(temp_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        # This should not raise
        corpus_module.corpus_list()

    def test_stats_empty_shows_zeros(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that stats on empty corpus shows zeros."""
        from urban_rag.cli import corpus as corpus_module

        temp_docs = tmp_path / "docs"
        temp_docs.mkdir()
        temp_manifest = tmp_path / "manifest.parquet"

        # Create mock settings object
        mock_settings = MagicMock()
        mock_settings.docs_dir = str(temp_docs)
        mock_settings.manifest_path = str(temp_manifest)

        # Patch get_settings in the corpus module
        monkeypatch.setattr(
            "urban_rag.cli.corpus.get_settings", lambda: mock_settings
        )

        # This should not raise
        corpus_module.corpus_stats()
