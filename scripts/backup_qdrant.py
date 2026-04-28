#!/usr/bin/env python3
"""
Backup Qdrant collections using the create_snapshot API.

This script creates point-in-time snapshots of all Qdrant collections
(pages_visual, pages_text) and saves them to local disk with metadata
for later restore. It is idempotent — re-running will create new snapshots
with different timestamps.

Usage:
    # Backup all collections to default ./backups directory:
    python scripts/backup_qdrant.py

    # Backup to a custom directory:
    python scripts/backup_qdrant.py --output-dir /mnt/backups/qdrant

    # Backup specific collections only:
    python scripts/backup_qdrant.py --collections pages_visual

    # Dry-run (list collections but don't create snapshots):
    python scripts/backup_qdrant.py --dry-run

Exit codes:
    0 = success (snapshots created)
    1 = failure (connection error, missing collection, etc.)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

from urban_rag.common.errors import ServiceUnavailableError
from urban_rag.index.qdrant_client import (
    COLLECTION_PAGES_TEXT,
    COLLECTION_PAGES_VISUAL,
    get_qdrant_client,
    list_collections,
)

logger = structlog.get_logger(__name__, service="backup-qdrant")

# Default backup directory
DEFAULT_BACKUP_DIR = Path("data/backups")
METADATA_FILENAME = "backup_metadata.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def backup_collection_snapshot(
    client: Any,
    collection_name: str,
    output_dir: Path,
    wait: bool = True,
) -> dict[str, Any]:
    """Create a snapshot for a single collection and download to output_dir.

    Args:
        client: Qdrant client instance.
        collection_name: Name of the collection to snapshot.
        output_dir: Directory to save the snapshot tarball and metadata.
        wait: Whether to wait for snapshot creation to complete.

    Returns:
        Dict with keys: snapshot_name, collection_name, local_path, size_bytes,
        created_at, checksum (sha256).

    Raises:
        ServiceUnavailableError: If Qdrant is unreachable or collection missing.
    """
    log = logger.bind(collection=collection_name)

    # Check if collection exists
    if not client.collection_exists(collection_name):
        raise ServiceUnavailableError(
            f"Collection '{collection_name}' does not exist. Cannot backup."
        )

    # Create snapshot on the Qdrant server
    log.info("Creating snapshot on Qdrant server", collection=collection_name)
    try:
        snapshot_info = client.create_snapshot(
            collection_name=collection_name,
            wait=wait,
        )
    except Exception as e:
        log.error("Failed to create snapshot", error=str(e))
        raise ServiceUnavailableError(
            f"Failed to create snapshot for collection '{collection_name}': {e}"
        ) from e

    if snapshot_info is None:
        raise ServiceUnavailableError(
            f"Snapshot creation returned None for collection '{collection_name}'"
        )

    # snapshot_info is a SnapshotDescription object
    snapshot_name = (
        getattr(snapshot_info, "name", None)
        or getattr(snapshot_info, "snapshot_name", None)
    )
    if not snapshot_name:
        # Try to extract from the response dict
        snapshot_name = getattr(snapshot_info, "id", snapshot_info.get("name", "unknown"))

    log.info("Snapshot created on server", snapshot_name=snapshot_name)

    # Download the snapshot file from Qdrant to local disk
    # Qdrant serves snapshots at /collections/{collection}/snapshots/{name}
    output_dir.mkdir(parents=True, exist_ok=True)

    # We use the client's download_snapshot method
    local_tarball = output_dir / f"{collection_name}__{snapshot_name}"

    try:
        client.download_snapshot(
            collection_name=collection_name,
            snapshot_name=snapshot_name,
            destination=str(local_tarball),
            wait=wait,
        )
    except Exception as e:
        log.error("Failed to download snapshot", error=str(e), snapshot_name=snapshot_name)
        # Clean up partial download
        if local_tarball.exists():
            local_tarball.unlink()
        raise ServiceUnavailableError(
            f"Failed to download snapshot '{snapshot_name}' for collection '{collection_name}': {e}"
        ) from e

    if not local_tarball.exists():
        raise ServiceUnavailableError(
            f"Downloaded snapshot file not found at {local_tarball}"
        )

    size_bytes = local_tarball.stat().st_size
    checksum = compute_file_sha256(local_tarball)

    log.info(
        "Snapshot downloaded",
        local_path=str(local_tarball),
        size_bytes=size_bytes,
        checksum=checksum[:16] + "...",
    )

    return {
        "snapshot_name": snapshot_name,
        "collection_name": collection_name,
        "local_path": str(local_tarball),
        "size_bytes": size_bytes,
        "checksum": checksum,
        "created_at": utc_now(),
    }


def list_collection_snapshots(client: Any, collection_name: str) -> list[dict[str, Any]]:
    """List all server-side snapshots for a collection."""
    try:
        snapshots = client.list_snapshots(collection_name=collection_name)
        result = []
        for snap in snapshots:
            result.append({
                "name": getattr(snap, "name", None) or getattr(snap, "id", "unknown"),
                "creation_time": getattr(snap, "creation_time", None),
                "size": getattr(snap, "size", 0),
            })
        return result
    except Exception as e:
        logger.warning("Failed to list snapshots", collection=collection_name, error=str(e))
        return []


# ---------------------------------------------------------------------------
# Main backup logic
# ---------------------------------------------------------------------------

def run_backup(
    output_dir: Path | str | None = None,
    collections: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full backup process.

    Args:
        output_dir: Directory to save backups. Defaults to ./data/backups.
        collections: List of collection names to backup. Defaults to all.
        dry_run: If True, list collections but don't create snapshots.

    Returns:
        Summary dict with keys: collections, snapshots, backup_dir, duration_s.
    """
    start_time = time.monotonic()
    log = logger.bind()

    output_dir = Path(output_dir) if output_dir else DEFAULT_BACKUP_DIR
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    if dry_run:
        log.info("DRY RUN — listing collections only")
        client = get_qdrant_client()
        existing = list_collections(client)
        log.info("Collections found", collections=existing)
        return {
            "collections": existing,
            "snapshots": [],
            "backup_dir": str(output_dir),
            "duration_s": 0,
            "dry_run": True,
        }

    # Connect to Qdrant
    log.info("Connecting to Qdrant")
    try:
        client = get_qdrant_client()
        existing_collections = list_collections(client)
        log.info("Connected", collections=existing_collections)
    except Exception as e:
        logger.error("Failed to connect to Qdrant", error=str(e))
        raise ServiceUnavailableError(f"Qdrant connection failed: {e}") from e

    # Determine which collections to backup
    all_target_collections = {COLLECTION_PAGES_VISUAL, COLLECTION_PAGES_TEXT}
    if collections:
        target = set(collections) & all_target_collections
    else:
        target = all_target_collections

    # Filter to only collections that exist
    to_backup = [c for c in target if c in existing_collections]
    missing = target - set(existing_collections)
    if missing:
        logger.warning("Collections not found, skipping", missing=list(missing))

    if not to_backup:
        logger.warning("No target collections found to backup")
        return {
            "collections": list(existing_collections),
            "snapshots": [],
            "backup_dir": str(output_dir),
            "duration_s": time.monotonic() - start_time,
            "dry_run": False,
        }

    # Per-collection snapshot subdirectory
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    created_snapshots = []

    for coll in to_backup:
        try:
            meta = backup_collection_snapshot(
                client=client,
                collection_name=coll,
                output_dir=run_dir,
                wait=True,
            )
            created_snapshots.append(meta)
            logger.info("Collection backed up", collection=coll, size=meta["size_bytes"])
        except ServiceUnavailableError as e:
            logger.error("Skipping collection due to error", collection=coll, error=str(e))
            # Continue with other collections rather than failing entirely
        except Exception as e:
            logger.error("Unexpected error backing up collection", collection=coll, error=str(e))
            # Continue with other collections

    # Write metadata file alongside snapshots
    metadata = {
        "backup_id": timestamp,
        "backup_at": utc_now(),
        "collections_backed_up": [s["collection_name"] for s in created_snapshots],
        "snapshots": created_snapshots,
    }
    metadata_path = run_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Backup metadata written", path=str(metadata_path))

    duration_s = time.monotonic() - start_time

    logger.info(
        "Backup complete",
        collections=len(created_snapshots),
        backup_dir=str(run_dir),
        duration_s=round(duration_s, 1),
    )

    return {
        "collections": list(existing_collections),
        "snapshots": created_snapshots,
        "backup_dir": str(run_dir),
        "duration_s": round(duration_s, 1),
        "dry_run": False,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point for backup script."""
    import typer

    app = typer.Typer(
        name="backup-qdrant",
        help="Backup Qdrant collections to local disk using create_snapshot API",
        no_args_is_help=True,
    )

    @app.command()
    def run(
        output_dir: Path | None = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Directory to save backups. Defaults to data/backups.",
        ),
        collections: list[str] | None = typer.Option(
            None,
            "--collections",
            "-c",
            help="Collection names to backup. Defaults to all (pages_visual, pages_text).",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="List collections but don't create snapshots.",
        ),
    ) -> None:
        """Run Qdrant backup."""
        try:
            result = run_backup(
                output_dir=output_dir,
                collections=collections,
                dry_run=dry_run,
            )
            if result["dry_run"]:
                logger.info("Dry run complete — no changes made")
            else:
                logger.info(
                    "Backup done",
                    collections=len(result["snapshots"]),
                    backup_dir=result["backup_dir"],
                    duration_s=result["duration_s"],
                )
            raise typer.Exit(code=0)
        except ServiceUnavailableError as e:
            logger.error("Backup failed", error=str(e))
            raise typer.Exit(code=1) from e
        except Exception as e:
            logger.error("Unexpected error", error=str(e))
            raise typer.Exit(code=1) from e

    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
