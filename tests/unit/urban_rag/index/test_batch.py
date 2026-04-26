"""Unit tests for index/batch.py — full corpus batch indexing job."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure test environment has required settings
os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-unit-tests")


class TestSanitizeForPath:
    """Tests for _sanitize_for_path helper."""

    def test_strips_spaces(self) -> None:
        """Spaces become underscores."""
        from urban_rag.index.batch import _sanitize_for_path

        assert _sanitize_for_path("swm 2016.pdf") == "swm_2016.pdf"
        assert _sanitize_for_path("URDPFI v2.pdf") == "URDPFI_v2.pdf"

    def test_strips_quotes(self) -> None:
        """Single and double quotes become underscores."""
        from urban_rag.index.batch import _sanitize_for_path

        assert _sanitize_for_path("doc's file.pdf") == "doc_s_file.pdf"
        assert _sanitize_for_path("say \"hello\".pdf") == "say_hello_.pdf"

    def test_strips_path_separators(self) -> None:
        """Path separators are removed."""
        from urban_rag.index.batch import _sanitize_for_path

        assert _sanitize_for_path("path/to/doc.pdf") == "path_to_doc.pdf"
        assert _sanitize_for_path("a\\b\\c.pdf") == "a_b_c.pdf"

    def test_collapse_multiple_underscores(self) -> None:
        """Multiple consecutive underscores are collapsed to one."""
        from urban_rag.index.batch import _sanitize_for_path

        assert _sanitize_for_path("a__b__c.pdf") == "a_b_c.pdf"
        assert _sanitize_for_path("file___name.pdf") == "file_name.pdf"  # 3+ collapsed


class TestBuildPoint:
    """Tests for _build_point helper."""

    def test_point_id_is_page_id(self) -> None:
        """Point ID is the deterministic page_id for idempotency."""
        from urban_rag.index.batch import PageIndexEntry, _build_point

        entry = PageIndexEntry(
            page_id="abc123#p0001",
            doc_hash="abc123",
            doc_filename="test.pdf",
            page_num=1,
            page_type="TEXT",
            dpi_used=100,
            image_path=Path("/tmp/test.png"),
            section_path=["Section 1"],
        )

        embeddings = [[0.1] * 128, [0.2] * 128]  # 2 patches, 128-dim
        point = _build_point(entry, embeddings)

        assert point["id"] == "abc123#p0001"

    def test_point_has_patches_and_pooled_vectors(self) -> None:
        """Point contains both patches (multivector) and pooled (single-vector)."""
        from urban_rag.index.batch import PageIndexEntry, _build_point

        entry = PageIndexEntry(
            page_id="abc#p0001",
            doc_hash="abc",
            doc_filename="doc.pdf",
            page_num=5,
            page_type="VISUAL",
            dpi_used=250,
            image_path=Path("/tmp/img.png"),
            section_path=["Chapter 1", "1.2"],
        )

        embeddings = [[0.1] * 128, [0.2] * 128, [0.3] * 128]
        point = _build_point(entry, embeddings)

        assert "patches" in point["vector"]
        assert "pooled" in point["vector"]
        assert len(point["vector"]["patches"]) == 3
        assert len(point["vector"]["pooled"]) == 128

    def test_pooled_is_mean_of_patches(self) -> None:
        """Pooled vector is element-wise mean of patch vectors."""
        from urban_rag.index.batch import PageIndexEntry, _build_point

        entry = PageIndexEntry(
            page_id="def#p0002",
            doc_hash="def",
            doc_filename="doc2.pdf",
            page_num=2,
            page_type="TEXT",
            dpi_used=100,
            image_path=Path("/tmp/p.png"),
            section_path=[],
        )

        embeddings = [[1.0] * 128, [1.0] * 128]
        point = _build_point(entry, embeddings)

        pooled = point["vector"]["pooled"]
        assert all(v == 1.0 for v in pooled)

    def test_payload_contains_required_fields(self) -> None:
        """Payload has doc_hash, doc_filename, page_num, page_type, dpi, section_path, image_uri."""
        from urban_rag.index.batch import PageIndexEntry, _build_point

        entry = PageIndexEntry(
            page_id="xyz#p0010",
            doc_hash="xyzhash",
            doc_filename="mumbai_dp.pdf",
            page_num=10,
            page_type="VISUAL",
            dpi_used=250,
            image_path=Path("/data/page_images/mumbai_dp__page_0010.png"),
            section_path=["Chapter 3", "3.1 FSI"],
        )

        embeddings = [[0.5] * 128]
        point = _build_point(entry, embeddings)
        payload = point["payload"]

        assert payload["doc_hash"] == "xyzhash"
        assert payload["doc_filename"] == "mumbai_dp.pdf"
        assert payload["page_num"] == 10
        assert payload["page_type"] == "VISUAL"
        assert payload["dpi"] == 250
        assert payload["section_path"] == ["Chapter 3", "3.1 FSI"]
        assert "image_uri" in payload
        assert "lance://" in payload["image_uri"]


class TestPageIndexEntry:
    """Tests for PageIndexEntry."""

    def test_image_uri_format(self) -> None:
        """image_uri uses lance:// prefix."""
        from urban_rag.index.batch import PageIndexEntry

        entry = PageIndexEntry(
            page_id="hsh#p0001",
            doc_hash="hsh",
            doc_filename="file.pdf",
            page_num=1,
            page_type="TEXT",
            dpi_used=100,
            image_path=Path("/data/docs/hsh/pages/p_0001.png"),
            section_path=[],
        )

        assert entry.image_uri == "lance:///data/docs/hsh/pages/p_0001.png"


class TestDiscoverPages:
    """Tests for _discover_pages page discovery logic."""

    def _make_temp_corpus(
        self, tmp_path: Path, doc_hashes: list[str], filenames: list[str]
    ) -> dict[str, Any]:
        """Helper to create a minimal test corpus with known documents and image files."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        manifest_rows = []

        for doc_hash, filename in zip(doc_hashes, filenames, strict=True):
            doc_dir = docs_dir / doc_hash
            doc_dir.mkdir(parents=True, exist_ok=True)

            num_pages = 3
            pages = []
            for page_num in range(1, num_pages + 1):
                page_id = f"{doc_hash}#p{page_num:04d}"
                pages.append({
                    "page_id": page_id,
                    "doc_id": doc_hash,
                    "page_num": page_num,
                    "page_type": "TEXT",
                    "dpi_used": 100,
                    "section_title": f"Section {page_num}",
                })

            with (doc_dir / "pages.jsonl").open("w") as f:
                for p in pages:
                    f.write(json.dumps(p) + "\n")

            manifest_rows.append({
                "doc_hash": doc_hash,
                "filename": filename,
                "title": filename,
                "family": "OTHER",
                "jurisdiction": "IN",
                "publisher": None,
                "year": 2024,
                "version": "v1",
                "license": "unknown",
                "page_count": num_pages,
                "size_bytes": 1024,
                "storage_uri": "",
                "ingested_at": "2024-01-01T00:00:00",
                "indexed_at": None,
            })

        import pandas as pd

        manifest_path = tmp_path / "manifest.parquet"
        pd.DataFrame(manifest_rows).to_parquet(str(manifest_path))

        page_images_dir = tmp_path / "page_images"
        page_images_dir.mkdir()

        # Create actual image files that match what _discover_pages will look for
        for _doc_hash, filename in zip(doc_hashes, filenames, strict=True):
            from urban_rag.index.batch import _sanitize_for_path
            sanitized = _sanitize_for_path(filename)
            for page_num in range(1, 4):
                image_file = page_images_dir / f"{sanitized}__page_{page_num:04d}.png"
                image_file.write_bytes(b"fake png")

        return {
            "docs_dir": docs_dir,
            "manifest_path": manifest_path,
            "page_images_dir": page_images_dir,
        }

    def test_discovers_all_pages_when_images_exist(self, tmp_path: Path) -> None:
        """All pages are discovered when corresponding image files exist."""
        from urban_rag.index.batch import _discover_pages

        corpus = self._make_temp_corpus(
            tmp_path,
            doc_hashes=["a" * 64, "b" * 64],
            filenames=["doc_a.pdf", "doc_b.pdf"],
        )

        pages = _discover_pages(
            docs_dir=corpus["docs_dir"],
            manifest_path=corpus["manifest_path"],
            page_images_dir=corpus["page_images_dir"],
        )

        assert len(pages) == 6  # 2 docs x 3 pages

    def test_missing_pages_jsonl_skips_document(self, tmp_path: Path) -> None:
        """Documents without pages.jsonl are skipped with a warning."""
        from urban_rag.index.batch import _discover_pages

        corpus = self._make_temp_corpus(
            tmp_path,
            doc_hashes=["c" * 64, "d" * 64],
            filenames=["doc_c.pdf", "doc_d.pdf"],
        )

        # Remove pages.jsonl from first doc
        first_doc = next(corpus["docs_dir"].iterdir())
        (first_doc / "pages.jsonl").unlink()

        pages = _discover_pages(
            docs_dir=corpus["docs_dir"],
            manifest_path=corpus["manifest_path"],
            page_images_dir=corpus["page_images_dir"],
        )

        # Only 3 pages from the second doc
        assert len(pages) == 3

    def test_missing_image_skips_page(self, tmp_path: Path) -> None:
        """Pages whose image files don't exist are skipped."""
        from urban_rag.index.batch import _discover_pages

        # Create corpus but don't create image files
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        import pandas as pd

        doc_hash = "e" * 64
        doc_dir = docs_dir / doc_hash
        doc_dir.mkdir(parents=True)

        pages = [{"page_id": f"{doc_hash}#p{i:04d}", "doc_id": doc_hash,
                  "page_num": i, "page_type": "TEXT", "dpi_used": 100,
                  "section_title": None} for i in range(1, 4)]
        with (doc_dir / "pages.jsonl").open("w") as f:
            for p in pages:
                f.write(json.dumps(p) + "\n")

        manifest_df = pd.DataFrame([{"doc_hash": doc_hash, "filename": "doc_e.pdf",
                                      "title": "", "family": "OTHER", "jurisdiction": "IN",
                                      "publisher": None, "year": 2024, "version": "v1",
                                      "license": "unknown", "page_count": 3,
                                      "size_bytes": 1024, "storage_uri": "",
                                      "ingested_at": "2024-01-01T00:00:00", "indexed_at": None}])
        manifest_df.to_parquet(str(tmp_path / "manifest.parquet"))

        page_images_dir = tmp_path / "page_images"
        page_images_dir.mkdir()  # No files created — images missing

        discovered = _discover_pages(
            docs_dir=docs_dir,
            manifest_path=tmp_path / "manifest.parquet",
            page_images_dir=page_images_dir,
        )

        assert len(discovered) == 0  # All skipped due to missing images

    def test_empty_manifest_raises(self, tmp_path: Path) -> None:
        """Empty manifest raises ValidationError."""
        from urban_rag.common.errors import ValidationError
        from urban_rag.index.batch import _discover_pages

        manifest_path = tmp_path / "manifest.parquet"
        import pandas as pd
        pd.DataFrame([]).to_parquet(str(manifest_path))

        with pytest.raises(ValidationError, match="empty"):
            _discover_pages(
                docs_dir=tmp_path / "docs",
                manifest_path=manifest_path,
                page_images_dir=tmp_path / "page_images",
            )

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        """Missing manifest raises ValidationError."""
        from urban_rag.common.errors import ValidationError
        from urban_rag.index.batch import _discover_pages

        with pytest.raises(ValidationError, match="Manifest not found"):
            _discover_pages(
                docs_dir=tmp_path / "docs",
                manifest_path=tmp_path / "nonexistent.parquet",
                page_images_dir=tmp_path / "page_images",
            )


class TestEmbedPagesViaHttp:
    """Tests for _embed_pages_via_http."""

    def test_calls_embed_endpoint(self) -> None:
        """Calls POST /embed with correct payload."""
        from urban_rag.index.batch import _embed_pages_via_http

        paths = [Path("/tmp/page1.png"), Path("/tmp/page2.png")]

        with ExitStack() as stack:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embeddings": [[[0.1] * 128], [[0.2] * 128]],
                "model_id": "vidore/colqwen2.5-v0.2",
                "device": "cuda",
                "batch_size": 2,
                "latency_ms": 100,
            }

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.post.return_value = mock_response

            stack.enter_context(
                patch("urban_rag.index.batch.httpx.Client", return_value=mock_client_instance)
            )

            result = _embed_pages_via_http(paths, "http://localhost:3102")

            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            assert call_args.args[0] == "http://localhost:3102/embed"
            assert call_args.kwargs["json"]["image_paths"] == ["/tmp/page1.png", "/tmp/page2.png"]
            assert len(result) == 2

    def test_service_unavailable_on_connect_error(self) -> None:
        """Transport error raises ServiceUnavailableError."""
        import httpx

        from urban_rag.common.errors import ServiceUnavailableError
        from urban_rag.index.batch import _embed_pages_via_http

        with ExitStack() as stack:
            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.post.side_effect = httpx.TransportError("connection refused")

            stack.enter_context(
                patch("urban_rag.index.batch.httpx.Client", return_value=mock_client_instance)
            )

            with pytest.raises(ServiceUnavailableError, match="Cannot connect"):
                _embed_pages_via_http([Path("/tmp/p.png")], "http://localhost:3102")

    def test_service_unavailable_on_timeout(self) -> None:
        """Timeout raises ServiceUnavailableError."""
        import httpx

        from urban_rag.common.errors import ServiceUnavailableError
        from urban_rag.index.batch import _embed_pages_via_http

        with ExitStack() as stack:
            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.post.side_effect = httpx.TimeoutException("timed out")

            stack.enter_context(
                patch("urban_rag.index.batch.httpx.Client", return_value=mock_client_instance)
            )

            with pytest.raises(ServiceUnavailableError, match="timed out"):
                _embed_pages_via_http([Path("/tmp/p.png")], "http://localhost:3102")

    def test_service_unavailable_on_non_200(self) -> None:
        """Non-200 response raises ServiceUnavailableError."""
        from urban_rag.common.errors import ServiceUnavailableError
        from urban_rag.index.batch import _embed_pages_via_http

        with ExitStack() as stack:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.post.return_value = mock_response

            stack.enter_context(
                patch("urban_rag.index.batch.httpx.Client", return_value=mock_client_instance)
            )

            with pytest.raises(ServiceUnavailableError, match="500"):
                _embed_pages_via_http([Path("/tmp/p.png")], "http://localhost:3102")
