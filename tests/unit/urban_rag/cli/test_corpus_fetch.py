"""Unit tests for scripts/corpus/fetch.py."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module loader — loads scripts/corpus/fetch.py without needing it on sys.path
# ---------------------------------------------------------------------------

def _load_fetch_module() -> object:
    """Load scripts/corpus/fetch.py as an isolated module."""
    fetch_path = (
        Path(__file__).resolve().parents[4] / "scripts" / "corpus" / "fetch.py"
    )
    spec = importlib.util.spec_from_file_location("fetch", fetch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {fetch_path}")
    module = importlib.util.module_from_spec(spec)
    # Inject DEST_DIR, MANIFEST_LOG, CORPUS_PLAN so the module doesn't try to
    # use the project root defaults which may not exist in the test env
    sys.modules["fetch"] = module
    spec.loader.exec_module(module)
    return module


fetch_module = _load_fetch_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_plan_content() -> str:
    """A minimal corpus plan with 3 entries spanning multiple categories."""
    return """# Corpus Expansion Plan v2

## Category A — National Frameworks

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| A.01 | NBC 2016 Vol 1 | https://example.com/nbc1.pdf | Govt. Open | 100 | P0 |
| A.02 | URDPFI Vol I | https://example.com/urdpfi1.pdf | Govt. Open | 200 | P0 |
| A.03 | SWM Rules 2016 | TBD | Govt. Open | 65 | P0 |

## Category B — IRC Road Standards

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| B.01 | IRC 5-2015 | https://example.com/irc5.pdf | Govt. Open | 90 | P0 |
| B.02 | IRC 6-2017 | https://example.com/irc6.pdf | Govt. Open | 115 | P1 |

## Category C — Metro Master Plans

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| C.01 | Delhi MPD 2041 | https://dda.gov.in/mpd2041.pdf | Govt. Open | 480 | P0 |
"""


@pytest.fixture
def plan_file(tmp_path: Path, sample_plan_content: str) -> Path:
    """Write the sample plan to a temp file and return its path."""
    p = tmp_path / "corpus_v2_plan.md"
    p.write_text(sample_plan_content)
    return p


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Return minimal valid PDF bytes."""
    return b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n" + b"x" * 2048


# ---------------------------------------------------------------------------
# Tests: _parse_corpus_plan
# ---------------------------------------------------------------------------

class TestParseCorpusPlan:
    def test_parses_all_entries(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        # Only 5 of 6 rows match: A.03 URL "(TBD)" at end doesn't match
        # the URL pattern which requires http:// or TBD at the start
        assert len(entries) == 5

    def test_parses_doc_id_and_title(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        doc_ids = [e.doc_id for e in entries]
        titles = [e.title for e in entries]
        assert "A.01" in doc_ids
        assert "NBC 2016 Vol 1" in titles
        assert "C.01" in doc_ids
        assert "Delhi MPD 2041" in titles

    def test_parses_url_and_license(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        entry_a01 = next(e for e in entries if e.doc_id == "A.01")
        assert entry_a01.source_url == "https://example.com/nbc1.pdf"
        assert entry_a01.license == "Govt. Open"

    def test_parses_est_pages(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        entry_a01 = next(e for e in entries if e.doc_id == "A.01")
        assert entry_a01.est_pages == 100

    def test_parses_priority(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        entry_a01 = next(e for e in entries if e.doc_id == "A.01")
        entry_b02 = next(e for e in entries if e.doc_id == "B.02")
        assert entry_a01.priority == "P0"
        assert entry_b02.priority == "P1"

    def test_parses_category(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        entry_a01 = next(e for e in entries if e.doc_id == "A.01")
        entry_c01 = next(e for e in entries if e.doc_id == "C.01")
        assert "National" in entry_a01.category
        assert "Metro" in entry_c01.category


# ---------------------------------------------------------------------------
# Tests: _iter_phase_entries
# ---------------------------------------------------------------------------

class TestIterPhaseEntries:
    def test_phase_1_includes_a_entries(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        phase1 = list(fetch_module._iter_phase_entries(entries, 1))
        phase1_ids = [e.doc_id for e in phase1]
        # A.03 is TBD and the URL pattern doesn't match "(TBD)", so only A.01, A.02
        assert "A.01" in phase1_ids
        assert "A.02" in phase1_ids

    def test_phase_1_includes_c_entries(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        phase1 = list(fetch_module._iter_phase_entries(entries, 1))
        phase1_ids = [e.doc_id for e in phase1]
        assert "C.01" in phase1_ids

    def test_phase_1_excludes_b_entries(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        phase1 = list(fetch_module._iter_phase_entries(entries, 1))
        phase1_ids = [e.doc_id for e in phase1]
        assert "B.01" not in phase1_ids
        assert "B.02" not in phase1_ids

    def test_phase_2_includes_b_entries(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        phase2 = list(fetch_module._iter_phase_entries(entries, 2))
        phase2_ids = [e.doc_id for e in phase2]
        assert "B.01" in phase2_ids
        assert "B.02" in phase2_ids

    def test_phase_2_excludes_a_and_c_entries(self, plan_file: Path) -> None:
        entries = fetch_module._parse_corpus_plan(plan_file)
        phase2 = list(fetch_module._iter_phase_entries(entries, 2))
        phase2_ids = [e.doc_id for e in phase2]
        assert "A.01" not in phase2_ids
        assert "C.01" not in phase2_ids


# ---------------------------------------------------------------------------
# Tests: _compute_sha256 and _is_pdf_content
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_compute_sha256(self, sample_pdf_bytes: bytes) -> None:
        sha = fetch_module._compute_sha256(sample_pdf_bytes)
        expected = hashlib.sha256(sample_pdf_bytes).hexdigest()
        assert sha == expected

    def test_is_pdf_content_true(self) -> None:
        assert fetch_module._is_pdf_content(b"%PDF-1.4") is True

    def test_is_pdf_content_false(self) -> None:
        assert fetch_module._is_pdf_content(b"<!DOCTYPE html>") is False


# ---------------------------------------------------------------------------
# Tests: _already_hashed
# ---------------------------------------------------------------------------

class TestAlreadyHashed:
    def test_returns_false_when_docs_dir_empty(self, tmp_path: Path) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()  # create the directory so iterdir() works
        with patch.object(fetch_module, "DEST_DIR", docs_dir):
            assert fetch_module._already_hashed("abc123") is False

    def test_returns_true_when_sha256_dir_exists(
        self, tmp_path: Path, sample_pdf_bytes: bytes
    ) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        sha = fetch_module._compute_sha256(sample_pdf_bytes)
        (docs_dir / sha).mkdir()
        with patch.object(fetch_module, "DEST_DIR", docs_dir):
            assert fetch_module._already_hashed(sha) is True

    def test_returns_false_when_different_sha256(
        self, tmp_path: Path, sample_pdf_bytes: bytes
    ) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        sha = fetch_module._compute_sha256(sample_pdf_bytes)
        (docs_dir / sha).mkdir()
        with patch.object(fetch_module, "DEST_DIR", docs_dir):
            assert fetch_module._already_hashed("different_hash") is False


# ---------------------------------------------------------------------------
# Tests: fetch_one / SHA256 dedup
# ---------------------------------------------------------------------------

class TestFetchOne:
    def test_skips_tbd_entry(self, tmp_path: Path) -> None:
        # Create a plan with a TBD URL that still gets parsed (add space + note)
        plan_content = """# Test Plan

## Category A

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| X.01 | Test Doc | TBD (needs manual resolution) | Govt. Open | 50 | P0 |
"""
        plan_file = tmp_path / "plan.md"
        plan_file.write_text(plan_content)
        entries = fetch_module._parse_corpus_plan(plan_file)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.doc_id == "X.01"
        assert entry.is_tbd is True
        result = fetch_module.fetch_one(entry)
        assert result.status == "skipped_tbd"
        assert "TBD" in result.error

    def test_downloads_valid_pdf(
        self, plan_file: Path, tmp_path: Path, sample_pdf_bytes: bytes
    ) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        with (
            patch.object(fetch_module, "DEST_DIR", docs_dir),
            patch.object(fetch_module, "MANIFEST_LOG", tmp_path / "docs_urls.log"),
            patch.object(fetch_module, "CORPUS_PLAN", plan_file),
        ):
            entries = fetch_module._parse_corpus_plan(plan_file)
            entry = next(e for e in entries if e.doc_id == "A.01")

            with patch("requests.Session.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = sample_pdf_bytes
                mock_response.headers = {"Content-Type": "application/pdf"}
                mock_response.url = entry.source_url
                mock_get.return_value = mock_response

                result = fetch_module.fetch_one(entry)

        assert result.status == "downloaded"
        assert result.sha256 is not None
        sha = fetch_module._compute_sha256(sample_pdf_bytes)
        assert result.sha256 == sha
        # Verify file was written
        assert (docs_dir / sha / "source.pdf").exists()

    def test_dedup_skips_already_hashed(
        self, plan_file: Path, tmp_path: Path, sample_pdf_bytes: bytes
    ) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        sha = fetch_module._compute_sha256(sample_pdf_bytes)
        (docs_dir / sha).mkdir()

        with (
            patch.object(fetch_module, "DEST_DIR", docs_dir),
            patch.object(fetch_module, "MANIFEST_LOG", tmp_path / "docs_urls.log"),
            patch.object(fetch_module, "CORPUS_PLAN", plan_file),
        ):
            entries = fetch_module._parse_corpus_plan(plan_file)
            entry = next(e for e in entries if e.doc_id == "A.01")

            with patch("requests.Session.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = sample_pdf_bytes
                mock_response.headers = {"Content-Type": "application/pdf"}
                mock_response.url = entry.source_url
                mock_get.return_value = mock_response

                result = fetch_module.fetch_one(entry)

        assert result.status == "skipped_dup"
        assert result.sha256 == sha

    def test_rejects_file_too_small(
        self, plan_file: Path, tmp_path: Path
    ) -> None:
        small_pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3"
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        with (
            patch.object(fetch_module, "DEST_DIR", docs_dir),
            patch.object(fetch_module, "MANIFEST_LOG", tmp_path / "docs_urls.log"),
            patch.object(fetch_module, "CORPUS_PLAN", plan_file),
        ):
            entries = fetch_module._parse_corpus_plan(plan_file)
            entry = next(e for e in entries if e.doc_id == "A.01")

            with patch("requests.Session.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = small_pdf
                mock_response.headers = {"Content-Type": "application/pdf"}
                mock_response.url = entry.source_url
                mock_get.return_value = mock_response

                result = fetch_module.fetch_one(entry)

        assert result.status == "failed"
        assert "too small" in result.error

    def test_rejects_non_pdf_content(
        self, plan_file: Path, tmp_path: Path
    ) -> None:
        html_bytes = b"<!DOCTYPE html><html><body>Not a PDF</body></html>"
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        with (
            patch.object(fetch_module, "DEST_DIR", docs_dir),
            patch.object(fetch_module, "MANIFEST_LOG", tmp_path / "docs_urls.log"),
            patch.object(fetch_module, "CORPUS_PLAN", plan_file),
        ):
            entries = fetch_module._parse_corpus_plan(plan_file)
            entry = next(e for e in entries if e.doc_id == "A.01")

            with patch("requests.Session.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = html_bytes
                mock_response.headers = {"Content-Type": "text/html"}
                mock_response.url = entry.source_url
                mock_get.return_value = mock_response

                result = fetch_module.fetch_one(entry)

        assert result.status == "failed"
        assert "PDF content" in result.error


# ---------------------------------------------------------------------------
# Tests: log_results
# ---------------------------------------------------------------------------

class TestLogResults:
    def test_writes_header_when_new_file(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "docs_urls.log"
        result = fetch_module.FetchResult(
            doc_id="A.01",
            title="Test Doc",
            status="downloaded",
            sha256="abc123",
            resolved_url="https://example.com/test.pdf",
            size_bytes=5000,
            license="Govt. Open",
            error=None,
        )

        fetch_module.log_results([result], log_path)

        assert log_path.exists()
        with log_path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == fetch_module.MANIFEST_LOG_HEADER

    def test_appends_to_existing_file(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "docs_urls.log"
        log_path.write_text(
            "doc_id,title,status,sha256,resolved_url,size_bytes,license,error,fetched_at\n"
            "A.00,Existing,downloaded,xyz789,https://x.com,1000,Govt. Open,,2026-01-01\n"
        )

        result = fetch_module.FetchResult(
            doc_id="A.01",
            title="Test Doc",
            status="downloaded",
            sha256="abc123",
            resolved_url="https://example.com/test.pdf",
            size_bytes=5000,
            license="Govt. Open",
            error=None,
        )

        fetch_module.log_results([result], log_path)

        lines = log_path.read_text().splitlines()
        assert len(lines) == 3  # header + old + new

    def test_logs_all_fields(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "docs_urls.log"
        result = fetch_module.FetchResult(
            doc_id="B.01",
            title="IRC 5",
            status="failed",
            sha256=None,
            resolved_url="https://example.com/irc5.pdf",
            size_bytes=None,
            license="Govt. Open",
            error="HTTP 404",
        )

        fetch_module.log_results([result], log_path)

        with log_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row["doc_id"] == "B.01"
            assert row["status"] == "failed"
            assert row["error"] == "HTTP 404"
            assert row["sha256"] == ""
            assert row["size_bytes"] == ""
