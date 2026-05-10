#!/usr/bin/env python3
"""scripts/corpus/fetch.py — Source acquisition automation.

Downloads PDFs from the corpus expansion plan (docs/corpus_v2_plan.md) to
`data/docs/`, deduplicates by SHA256, and logs licensing information to
`data/docs_urls.log`.

Usage:
    python scripts/corpus/fetch.py                    # fetch all Phase 1 docs
    python scripts/corpus/fetch.py --phase 2         # fetch Phase 2 docs
    python scripts/corpus/fetch.py --limit 50         # fetch at most 50 docs
    python scripts/corpus/fetch.py --dry-run          # show what would be fetched

The corpus plan is read directly from docs/corpus_v2_plan.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEST_DIR = Path("data/docs")
MANIFEST_LOG = Path("data/docs_urls.log")
CORPUS_PLAN = Path("docs/corpus_v2_plan.md")

MAX_WORKERS = 4
REQUEST_TIMEOUT_SECONDS = 60
RATE_LIMIT_PAUSE_SECONDS = 1.0  # polite pause between requests to same domain
MIN_PDF_SIZE_BYTES = 1024

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CorpusEntry:
    """A single document entry from the corpus plan."""

    doc_id: str
    title: str
    source_url: str
    license: str
    est_pages: int
    priority: str
    category: str = ""

    @property
    def is_tbd(self) -> bool:
        stripped = self.source_url.strip().upper()
        return stripped == "TBD" or stripped.startswith("TBD ")


@dataclass
class FetchResult:
    """Outcome of attempting to fetch one document."""

    doc_id: str
    title: str
    status: str  # "downloaded" | "skipped_dup" | "skipped_tbd" | "failed" | "exists"
    sha256: str | None
    resolved_url: str | None
    size_bytes: int | None
    license: str
    error: str | None


# ---------------------------------------------------------------------------
# Corpus plan parser
# ---------------------------------------------------------------------------


def _parse_corpus_plan(plan_path: Path) -> list[CorpusEntry]:
    """Parse docs/corpus_v2_plan.md and yield CorpusEntry objects.

    Parses the markdown tables in the corpus plan using regex. Each table row
    must contain a doc_id (e.g. "A.01", "B.02"), a title, a URL, a license,
    an estimated page count, and a priority band.
    """
    text = plan_path.read_text()

    # Pattern for table rows: | A.01 | Title | URL | License | Est. Pages | Priority |
    # We capture: doc_id, title, source_url, license, est_pages, priority
    row_pattern = re.compile(
        r"^\| ([A-Z]\.\d+) \| (.+?) \|"  # doc_id and title
        r" ((?:https?://|TBD)[^\|]+?) \|"  # URL (allow TBD)
        r" ([^\|]+?) \|"  # license
        r" (\d+) \|"  # est_pages
        r" (P\d) \|",  # priority
        re.IGNORECASE | re.MULTILINE,
    )

    entries: list[CorpusEntry] = []
    current_category = ""

    for line in text.splitlines():
        # Detect category headers (## Category A — ...)
        cat_match = re.match(r"^## Category [A-Z] — (.+)", line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            continue

        row_match = row_pattern.match(line)
        if row_match:
            doc_id, title, source_url, license, est_pages_str, priority = row_match.groups()
            entries.append(
                CorpusEntry(
                    doc_id=doc_id.strip(),
                    title=title.strip(),
                    source_url=source_url.strip(),
                    license=license.strip(),
                    est_pages=int(est_pages_str.strip()),
                    priority=priority.strip().upper(),
                    category=current_category,
                )
            )

    return entries


def _iter_phase_entries(entries: list[CorpusEntry], phase: int):
    """Yield entries for a specific acquisition phase, in priority order."""
    if phase == 1:
        # A.01-A.17 (Nat. frameworks), C.01-C.14 (Metro MP),
        # A.26-A.34 (MoHUA), E.01-E.04, J.01-J.07
        for e in entries:
            prefix = e.doc_id.split(".")[0]
            if prefix in ("C", "E", "J") or (
                prefix == "A"
                and (
                    re.match(r"A\.0[1-9]", e.doc_id)
                    or re.match(r"A\.(2[6-9]|[3-9]\d)", e.doc_id)
                )
            ):
                yield e
    elif phase == 2:
        # D.01-D.35, B.01-B.10, E.05-E.19, G.01-G.08, I.01-I.05
        for e in entries:
            prefix = e.doc_id.split(".")[0]
            in_d_or_b = prefix in ("D", "B")
            in_e_5_plus = prefix == "E" and re.match(r"E\.([5-9]|[1-9]\d)", e.doc_id)
            in_g_1_8 = prefix == "G" and re.match(r"G\.([1-8])", e.doc_id)
            in_i_1_5 = prefix == "I" and re.match(r"I\.([1-5])", e.doc_id)
            if in_d_or_b or in_e_5_plus or in_g_1_8 or in_i_1_5:
                yield e
    else:
        # All remaining entries
        seen_ids: set[str] = set()
        for phase_num in (1, 2):
            for e in _iter_phase_entries(entries, phase_num):
                seen_ids.add(e.doc_id)
        for e in entries:
            if e.doc_id not in seen_ids:
                yield e


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _compute_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _is_pdf_content(content: bytes) -> bool:
    return content[:4] == b"%PDF"


def _download_pdf(entry: CorpusEntry, timeout: int = REQUEST_TIMEOUT_SECONDS) -> tuple[bytes, str]:
    """Download a PDF from a URL and return (bytes, resolved_url).

    Raises:
        requests.HTTPError: on HTTP errors (4xx/5xx).
        requests.Timeout: if the request times out.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; UrbanPlanningRAG/1.0; "
                "+https://github.com/tanmaychoudhary/urban-planning-rag)"
            ),
            "Accept": "application/pdf,application/octet-stream,*/*",
        }
    )

    response = session.get(entry.source_url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if (
        content_type
        and "pdf" not in content_type.lower()
        and "octet" not in content_type.lower()
        and not _is_pdf_content(response.content)
    ):
        raise requests.HTTPError(
            f"URL did not return PDF content (Content-Type: {content_type})",
            response=response,
        )

    return response.content, response.url


# ---------------------------------------------------------------------------
# SHA256 dedup check
# ---------------------------------------------------------------------------


def _already_hashed(sha256: str) -> bool:
    """Check if a document with this SHA256 already exists in data/docs/."""
    return any(
        doc_dir.is_dir() and doc_dir.name == sha256 for doc_dir in DEST_DIR.iterdir()
    )


def _already_logged(doc_id: str) -> bool:
    """Check if a doc_id is already in the manifest log."""
    if not MANIFEST_LOG.exists():
        return False
    with MANIFEST_LOG.open() as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0] == doc_id:
                return True
    return False


# ---------------------------------------------------------------------------
# Main fetch logic
# ---------------------------------------------------------------------------

_fetch_interrupted = False


def _sigint_handler(signum, frame):
    global _fetch_interrupted
    _fetch_interrupted = True
    sys.stderr.write("\n[fetch] Interrupted — finishing current downloads, then exiting.\n")
    sys.stderr.flush()


def fetch_one(entry: CorpusEntry) -> FetchResult:
    """Attempt to fetch one corpus entry.

    Returns a FetchResult describing what happened.
    """
    global _fetch_interrupted

    if _fetch_interrupted:
        return FetchResult(
            doc_id=entry.doc_id,
            title=entry.title,
            status="skipped",
            sha256=None,
            resolved_url=None,
            size_bytes=None,
            license=entry.license,
            error="Skipped due to interrupt",
        )

    # Skip TBD entries (URLs that need manual resolution)
    if entry.is_tbd:
        return FetchResult(
            doc_id=entry.doc_id,
            title=entry.title,
            status="skipped_tbd",
            sha256=None,
            resolved_url=None,
            size_bytes=None,
            license=entry.license,
            error="URL is TBD — needs manual resolution",
        )

    # Check if already logged
    if _already_logged(entry.doc_id):
        return FetchResult(
            doc_id=entry.doc_id,
            title=entry.title,
            status="exists",
            sha256=None,
            resolved_url=None,
            size_bytes=None,
            license=entry.license,
            error=None,
        )

    try:
        file_bytes, resolved_url = _download_pdf(entry)

        if len(file_bytes) < MIN_PDF_SIZE_BYTES:
            return FetchResult(
                doc_id=entry.doc_id,
                title=entry.title,
                status="failed",
                sha256=None,
                resolved_url=resolved_url,
                size_bytes=len(file_bytes),
                license=entry.license,
                error=f"Downloaded file too small ({len(file_bytes)} bytes)",
            )

        if not _is_pdf_content(file_bytes):
            return FetchResult(
                doc_id=entry.doc_id,
                title=entry.title,
                status="failed",
                sha256=None,
                resolved_url=resolved_url,
                size_bytes=len(file_bytes),
                license=entry.license,
                error="Downloaded content is not a PDF (missing %PDF header)",
            )

        sha256 = _compute_sha256(file_bytes)

        # Dedup: if SHA256 already exists, skip
        if _already_hashed(sha256):
            return FetchResult(
                doc_id=entry.doc_id,
                title=entry.title,
                status="skipped_dup",
                sha256=sha256,
                resolved_url=resolved_url,
                size_bytes=len(file_bytes),
                license=entry.license,
                error=None,
            )

        # Write to data/docs/<sha256>/source.pdf
        doc_dir = DEST_DIR / sha256
        doc_dir.mkdir(parents=True, exist_ok=True)
        dest_pdf = doc_dir / "source.pdf"
        dest_pdf.write_bytes(file_bytes)

        return FetchResult(
            doc_id=entry.doc_id,
            title=entry.title,
            status="downloaded",
            sha256=sha256,
            resolved_url=resolved_url,
            size_bytes=len(file_bytes),
            license=entry.license,
            error=None,
        )

    except requests.HTTPError as e:
        return FetchResult(
            doc_id=entry.doc_id,
            title=entry.title,
            status="failed",
            sha256=None,
            resolved_url=entry.source_url,
            size_bytes=None,
            license=entry.license,
            error=f"HTTP {e.response.status_code}: {e}",
        )
    except requests.Timeout:
        return FetchResult(
            doc_id=entry.doc_id,
            title=entry.title,
            status="failed",
            sha256=None,
            resolved_url=entry.source_url,
            size_bytes=None,
            license=entry.license,
            error=f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s",
        )
    except requests.RequestException as e:
        return FetchResult(
            doc_id=entry.doc_id,
            title=entry.title,
            status="failed",
            sha256=None,
            resolved_url=entry.source_url,
            size_bytes=None,
            license=entry.license,
            error=str(e),
        )
    except Exception as e:
        return FetchResult(
            doc_id=entry.doc_id,
            title=entry.title,
            status="failed",
            sha256=None,
            resolved_url=entry.source_url,
            size_bytes=None,
            license=entry.license,
            error=f"Unexpected error: {e}",
        )


def fetch_all(
    entries: list[CorpusEntry],
    max_workers: int = MAX_WORKERS,
    rate_limit: float = RATE_LIMIT_PAUSE_SECONDS,
    dry_run: bool = False,
) -> list[FetchResult]:
    """Fetch all entries concurrently, with rate limiting per domain."""
    results: list[FetchResult] = []
    last_domain = ""
    pause_until = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, entry): entry for entry in entries}

        for future in as_completed(futures):
            if dry_run:
                entry = futures[future]
                sys.stderr.write(
                    f"[dry-run] Would fetch: {entry.doc_id} — "
                    f"{entry.title} ({entry.source_url})\n"
                )
                results.append(
                    FetchResult(
                        doc_id=entry.doc_id,
                        title=entry.title,
                        status="dry_run",
                        sha256=None,
                        resolved_url=entry.source_url,
                        size_bytes=None,
                        license=entry.license,
                        error=None,
                    )
                )
                continue

            result = future.result()
            results.append(result)

            # Log progress
            status_icon = {
                "downloaded": "SUCCESS",
                "skipped_dup": "SKIPPED_DUP",
                "skipped_tbd": "SKIPPED_TBD",
                "failed": "FAILED",
                "exists": "EXISTS",
            }.get(result.status, "UNKNOWN")

            msg = f"[{result.doc_id}] {result.status}"
            if result.error:
                msg += f" — {result.error}"
            sys.stderr.write(f"{status_icon}: {msg}\n")

            # Polite rate limiting
            if result.resolved_url:
                domain = urlparse(result.resolved_url).netloc
                if domain == last_domain and pause_until > 0:
                    sleep_time = rate_limit - (time.time() - pause_until)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                last_domain = domain
                pause_until = time.time()

            if _fetch_interrupted:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

    return results


# ---------------------------------------------------------------------------
# Manifest logging
# ---------------------------------------------------------------------------

MANIFEST_LOG_HEADER = [
    "doc_id",
    "title",
    "status",
    "sha256",
    "resolved_url",
    "size_bytes",
    "license",
    "error",
    "fetched_at",
]


def log_results(results: list[FetchResult], log_path: Path = MANIFEST_LOG) -> None:
    """Append fetch results to the manifest log CSV."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not log_path.exists()
    mode = "a" if log_path.exists() else "w"

    with log_path.open(mode) as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(MANIFEST_LOG_HEADER)

        now = datetime.now(tz=UTC).isoformat()
        for r in results:
            writer.writerow(
                [
                    r.doc_id,
                    r.title,
                    r.status,
                    r.sha256 or "",
                    r.resolved_url or "",
                    r.size_bytes or "",
                    r.license,
                    r.error or "",
                    now,
                ]
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Fetch PDFs from the corpus expansion plan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Corpus acquisition phase (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of documents to fetch (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without downloading anything",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of concurrent download workers (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if already in manifest log",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Set up signal handler for graceful interrupt
    signal.signal(signal.SIGINT, _sigint_handler)

    if not CORPUS_PLAN.exists():
        sys.stderr.write(f"ERROR: Corpus plan not found at {CORPUS_PLAN}\n")
        return 1

    sys.stderr.write(f"Reading corpus plan from {CORPUS_PLAN}...\n")
    all_entries = _parse_corpus_plan(CORPUS_PLAN)
    sys.stderr.write(f"Found {len(all_entries)} document entries in corpus plan.\n")

    # Get entries for the requested phase
    phase_entries = list(_iter_phase_entries(all_entries, args.phase))
    sys.stderr.write(f"Phase {args.phase} target: {len(phase_entries)} documents.\n")

    if args.limit > 0:
        phase_entries = phase_entries[: args.limit]
        sys.stderr.write(f"Limited to {args.limit} documents.\n")

    if not phase_entries:
        sys.stderr.write("No documents to fetch for this phase.\n")
        return 0

    sys.stderr.write(
        f"Fetching {len(phase_entries)} documents with "
        f"{args.workers} workers...\n"
    )

    results = fetch_all(phase_entries, max_workers=args.workers, dry_run=args.dry_run)

    if args.dry_run:
        return 0

    # Log results
    log_results(results)
    sys.stderr.write(f"Results logged to {MANIFEST_LOG}\n")

    # Summary
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    sys.stderr.write("\nSummary:\n")
    for status, count in sorted(counts.items()):
        sys.stderr.write(f"   {status}: {count}\n")

    total_downloaded = counts.get("downloaded", 0)
    total_failed = counts.get("failed", 0)
    overall = "SUCCESS" if total_failed == 0 else "PARTIAL"
    sys.stderr.write(
        f"\n{overall} Downloaded: {total_downloaded} | Failed: {total_failed}\n"
    )

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
