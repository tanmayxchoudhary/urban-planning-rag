"""Ingest CLI commands."""

from __future__ import annotations

from pathlib import Path

from urban_rag.common.logging import get_logger
from urban_rag.common.types import DocumentRecord
from urban_rag.ingest.load import validate_and_hash

log = get_logger(__name__, service="cli")


def ingest_file(path: Path) -> DocumentRecord:
    """Ingest a single PDF file into the corpus.

    Validates that the path exists and is a valid PDF before processing.
    This function is called by the CLI ingest command and can also be used
    directly for testing.

    Args:
        path: Path to the PDF file to ingest.

    Returns:
        DocumentRecord for the ingested document.

    Raises:
        ValidationError: If the file does not exist, is not a valid PDF,
            or fails validation.
    """
    log.info("ingest_file_started", path=str(path))

    # validate_and_hash handles:
    # - Path existence check (raises ValidationError if not found)
    # - PDF magic bytes verification
    # - Minimum file size check
    # - SHA256 content hash computation
    # - Content-addressed storage
    record = validate_and_hash(path)

    log.info(
        "ingest_file_completed",
        doc_hash=record.doc_hash,
        filename=record.filename,
        page_count=record.page_count,
    )

    return record
