#!/usr/bin/env python3
"""
Restore Qdrant collections from local snapshot backups.

This script restores collections from snapshots created by backup_qdrant.py.
It uses Qdrant's recover_snapshot API to restore collection data from
snapshot tarballs stored on local disk.

Usage:
    # Restore all collections from latest backup:
    python scripts/restore_qdrant.py

    # Restore from a specific backup directory:
    python scripts/restore_qdrant.py --backup-dir data/backups/run_20260428_120000

    # Restore specific collections:
    python scripts/restore_qdrant.py --collections pages_visual

    # Dry-run (validate backup metadata without restoring):
    python scripts/restore_qdrant.py --dry-run

    # Restore and wait for completion:
    python scripts/restore_qdrant.py --wait

Exit codes:
    0 = success (collections restored)
    1 = failure (missing backup, connection error, etc.)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import UTC

import structlog

from urban_rag.common.errors import ServiceUnavailableError
from urban_rag.index.qdrant_client import (
    COLLECTION_PAGES_TEXT,
    COLLECTION_PAGES_VISUAL,
    create_collections,
    get_qdrant_client,
    list_collections,
)

logger = structlog.get_logger(__name__, service="restore-qdrant")

METADATA_FILENAME = "backup_metadata.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    from datetime import datetime
    return datetime.now(UTC).isoformat()


def find_latest_backup(backups_root: Path) -> Path | None:
    """Find the most recent backup run directory under backups_root.

    Args:
        backups_root: Root directory containing run_* subdirectories.

    Returns:
        Path to the latest run directory, or None if none found.
    """
    if not backups_root.exists():
        return None

    run_dirs = sorted(
        [d for d in backups_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def load_backup_metadata(backup_dir: Path) -> dict[str, Any] | None:
    """Load and validate backup metadata from backup_dir.

    Args:
        backup_dir: Directory containing backup snapshot files and metadata.

    Returns:
        Metadata dict, or None if metadata file is missing/invalid.
    """
    metadata_path = backup_dir / METADATA_FILENAME
    if not metadata_path.exists():
        logger.warning("Backup metadata file not found", path=str(metadata_path))
        return None

    try:
        raw = metadata_path.read_text()
        meta = json.loads(raw)
        # Validate required fields
        if "snapshots" not in meta or "backup_id" not in meta:
            logger.warning("Backup metadata missing required fields", path=str(metadata_path))
            return None
        return meta
    except json.JSONDecodeError as e:
        logger.warning("Backup metadata is not valid JSON", path=str(metadata_path), error=str(e))
        return None


def validate_snapshot_file(snapshot_meta: dict[str, Any]) -> tuple[bool, str]:
    """Validate that a snapshot file exists and has correct checksum.

    Args:
        snapshot_meta: Dict with 'local_path' and 'checksum' keys.

    Returns:
        (is_valid, error_message) tuple.
    """
    local_path = Path(snapshot_meta["local_path"])
    if not local_path.exists():
        return False, f"Snapshot file not found: {local_path}"

    import hashlib
    expected_checksum = snapshot_meta["checksum"]
    actual_checksum = hashlib.sha256(local_path.read_bytes()).hexdigest()
    if actual_checksum != expected_checksum:
        return (
            False,
            f"Checksum mismatch for {local_path}: "
            f"expected {expected_checksum[:16]}..., got {actual_checksum[:16]}...",
        )

    return True, ""


def restore_collection_from_snapshot(
    client: Any,
    collection_name: str,
    snapshot_path: Path,
    wait: bool = True,
    priority: int | None = None,
) -> bool:
    """Restore a collection from a local snapshot tarball.

    Args:
        client: Qdrant client instance.
        collection_name: Name of the collection to restore.
        snapshot_path: Path to the snapshot tarball file.
        wait: Whether to wait for restore to complete.
        priority: Optional recovery priority (0=high, 1=medium, 2=low).

    Returns:
        True if restore succeeded, False otherwise.
    """
    log = logger.bind(collection=collection_name)

    if not snapshot_path.exists():
        log.error("Snapshot file not found", path=str(snapshot_path))
        return False

    # The recover_snapshot API on QdrantClient takes:
    # collection_name, location (URL or path), api_key, checksum, priority, wait
    # For local files, we use file:// URIs or absolute paths
    location = str(snapshot_path.resolve())

    priority_obj = None
    if priority is not None:
        from qdrant_client import models
        # Priority: 0=HIGH(snapshot), 1=MEDIUM(replica), 2=LOW(no_sync)
        priority_map = {
            0: models.SnapshotPriority.SNAPSHOT,
            1: models.SnapshotPriority.REPLICA,
            2: models.SnapshotPriority.NO_SYNC,
        }
        priority_obj = priority_map.get(priority)

    log.info("Starting restore", location=location, wait=wait, priority=priority)
    try:
        result = client.recover_snapshot(
            collection_name=collection_name,
            location=location,
            wait=wait,
            priority=priority_obj,
        )
        log.info("Restore completed", result=str(result))
        return True
    except Exception as e:
        log.error("Restore failed", error=str(e))
        return False


# ---------------------------------------------------------------------------
# Main restore logic
# ---------------------------------------------------------------------------

def run_restore(
    backup_dir: Path | str | None = None,
    collections: list[str] | None = None,
    dry_run: bool = False,
    wait: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Run the full restore process.

    Args:
        backup_dir: Directory containing backup snapshots and metadata.
                    If None, auto-detects latest from data/backups.
        collections: List of collection names to restore. Defaults to all in backup.
        dry_run: If True, validate backup without restoring.
        wait: Whether to wait for restore operations to complete.
        force: If True, overwrite existing collections without confirmation.

    Returns:
        Summary dict with restoration results.
    """
    start_time = time.monotonic()
    log = logger.bind()

    # Resolve backup directory
    if backup_dir is None:
        backup_dir = Path("data/backups")
        log.info("No backup directory specified, looking for latest in data/backups")

    backup_dir = Path(backup_dir)

    # If backup_dir is a run_* directory containing metadata, use it directly.
    # Otherwise, if it's the backups root, find the latest run.
    if not (backup_dir / METADATA_FILENAME).exists():
        latest = find_latest_backup(backup_dir)
        if latest is None:
            raise ServiceUnavailableError(
                f"No backup found in {backup_dir}. Run backup_qdrant.py first."
            )
        backup_dir = latest
        log.info("Using latest backup", backup_dir=str(backup_dir))

    log.info("Loading backup metadata", backup_dir=str(backup_dir))
    meta = load_backup_metadata(backup_dir)
    if meta is None:
        raise ServiceUnavailableError(
            f"Invalid or missing backup metadata in {backup_dir}. "
            "Run backup_qdrant.py first to create a valid backup."
        )

    # Connect to Qdrant
    log.info("Connecting to Qdrant")
    try:
        client = get_qdrant_client()
        existing = list_collections(client)
        log.info("Connected", collections=existing)
    except Exception as e:
        logger.error("Failed to connect to Qdrant", error=str(e))
        raise ServiceUnavailableError(f"Qdrant connection failed: {e}") from e

    # Determine which snapshots to restore
    snapshots = meta.get("snapshots", [])
    if not snapshots:
        raise ServiceUnavailableError("No snapshots found in backup metadata")

    # Filter to target collections
    all_targets = {COLLECTION_PAGES_VISUAL, COLLECTION_PAGES_TEXT}
    target_collections = set(collections) if collections else all_targets

    to_restore = [s for s in snapshots if s["collection_name"] in target_collections]

    if not to_restore:
        log.warning("No matching snapshots to restore", requested=list(target_collections))
        return {
            "backup_id": meta.get("backup_id"),
            "backup_dir": str(backup_dir),
            "collections_restored": [],
            "skipped": [],
            "errors": [],
            "duration_s": 0,
            "dry_run": dry_run,
        }

    if dry_run:
        log.info("DRY RUN — validating backup without restoring")
        validation_results = []
        for snap in to_restore:
            is_valid, error = validate_snapshot_file(snap)
            validation_results.append({
                "collection": snap["collection_name"],
                "snapshot_name": snap["snapshot_name"],
                "valid": is_valid,
                "error": error,
                "local_path": snap["local_path"],
                "size_bytes": snap.get("size_bytes", 0),
            })
            if is_valid:
                log.info(
                    "Snapshot valid",
                    collection=snap["collection_name"],
                    path=snap["local_path"],
                )
            else:
                log.error(
                    "Snapshot invalid",
                    collection=snap["collection_name"],
                    error=error,
                )

        return {
            "backup_id": meta.get("backup_id"),
            "backup_dir": str(backup_dir),
            "validation": validation_results,
            "dry_run": True,
        }

    # Restore each collection
    restored = []
    errors = []

    for snap in to_restore:
        collection_name = snap["collection_name"]
        snapshot_path = Path(snap["local_path"])

        log.info("Restoring collection", collection=collection_name, from_path=str(snapshot_path))

        # Validate file exists and checksum matches before restoring
        is_valid, error = validate_snapshot_file(snap)
        if not is_valid:
            log.error(
                "Skipping restore — snapshot invalid",
                collection=collection_name,
                error=error,
            )
            errors.append({"collection": collection_name, "error": error})
            continue

        # If collection already exists and force=False, warn
        if collection_name in existing and not force:
            log.warning(
                "Collection already exists. Use --force to overwrite.",
                collection=collection_name,
            )
            errors.append({
                "collection": collection_name,
                "error": (
                    f"Collection '{collection_name}' already exists. "
                    "Use --force to overwrite."
                ),
            })
            continue

        # Ensure collection schema exists in Qdrant (create if missing)
        if collection_name not in existing:
            log.info(
                    "Collection does not exist, creating schema",
                    collection=collection_name,
                )
            try:
                create_collections(client)
                # Note: create_collections creates all known collections.
                # If it's a custom collection not in our schema, we may need
                # to handle differently.
                existing_now = list_collections(client)
                if collection_name not in existing_now:
                    log.warning(
                        "Collection schema not created automatically. "
                        "You may need to create it first.",
                        collection=collection_name,
                    )
            except Exception as e:
                log.error(
                    "Failed to create collection schema",
                    collection=collection_name,
                    error=str(e),
                )

        success = restore_collection_from_snapshot(
            client=client,
            collection_name=collection_name,
            snapshot_path=snapshot_path,
            wait=wait,
        )

        if success:
            restored.append(collection_name)
            log.info("Collection restored", collection=collection_name)
        else:
            errors.append({
                "collection": collection_name,
                "error": "Restore operation returned failure",
            })

    duration_s = time.monotonic() - start_time

    logger.info(
        "Restore complete",
        restored=len(restored),
        errors=len(errors),
        duration_s=round(duration_s, 1),
    )

    return {
        "backup_id": meta.get("backup_id"),
        "backup_dir": str(backup_dir),
        "collections_restored": restored,
        "skipped": [
            s["collection_name"]
            for s in to_restore
            if s["collection_name"] not in restored
            and s["collection_name"] not in [e["collection"] for e in errors]
        ],
        "errors": errors,
        "duration_s": round(duration_s, 1),
        "dry_run": False,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point for restore script."""
    import typer

    app = typer.Typer(
        name="restore-qdrant",
        help="Restore Qdrant collections from local snapshot backups",
        no_args_is_help=True,
    )

    @app.command()
    def run(
        backup_dir: Path | None = typer.Option(
            None,
            "--backup-dir",
            "-b",
            help=(
                "Directory containing backup snapshots and metadata. "
                "Defaults to data/backups/run_<latest>. "
                "Can be a specific run directory (e.g. data/backups/run_20260428_120000) "
                "or the backups root (will auto-detect latest run)."
            ),
        ),
        collections: list[str] | None = typer.Option(
            None,
            "--collections",
            "-c",
            help=(
                "Collection names to restore. Defaults to all collections in backup "
                "(pages_visual, pages_text)."
            ),
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Validate backup metadata and snapshot files without restoring.",
        ),
        wait: bool = typer.Option(
            True,
            "--wait/--no-wait",
            help="Wait for restore operations to complete. Default: wait.",
        ),
        force: bool = typer.Option(
            False,
            "--force",
            "-f",
            help="Overwrite existing collections without confirmation.",
        ),
    ) -> None:
        """Restore Qdrant collections from local backups."""
        try:
            result = run_restore(
                backup_dir=backup_dir,
                collections=collections,
                dry_run=dry_run,
                wait=wait,
                force=force,
            )

            if result.get("dry_run"):
                validations = result.get("validation", [])
                all_valid = all(v["valid"] for v in validations)
                if all_valid:
                    logger.info(
                        "Dry run PASSED — all snapshots valid and ready to restore",
                        collections=[v["collection"] for v in validations],
                    )
                else:
                    logger.warning(
                        "Dry run FAILED — some snapshots are invalid",
                        invalid=[v["collection"] for v in validations if not v["valid"]],
                    )
                    raise typer.Exit(code=1)
            else:
                restored = result.get("collections_restored", [])
                errors = result.get("errors", [])
                if errors and not restored:
                    logger.error("Restore failed", errors=errors)
                    raise typer.Exit(code=1)
                if restored:
                    logger.info(
                        "Restore complete",
                        restored=restored,
                        errors=len(errors),
                        duration_s=result.get("duration_s"),
                    )
                else:
                    logger.warning("No collections were restored")

            raise typer.Exit(code=0)

        except ServiceUnavailableError as e:
            logger.error("Restore failed", error=str(e))
            raise typer.Exit(code=1) from e
        except Exception as e:
            logger.error("Unexpected error", error=str(e))
            raise typer.Exit(code=1) from e

    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
