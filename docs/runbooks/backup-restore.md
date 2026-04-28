# Qdrant Backup & Restore Runbook

**Purpose**: Document the Qdrant backup and restore procedure for the Urban Planning RAG system.
**SLO**: Restore drill completes within 30 minutes and passes smoke health checks.
**Scope**: Qdrant collections (`pages_visual`, `pages_text`) and metadata DB backup/restore.

---

## Overview

Qdrant is the vector database for the Urban Planning RAG retrieval system. The backup strategy uses Qdrant's native **snapshot API** (`create_snapshot` / `recover_snapshot`) to create tarball archives of each collection and restore them to staging or production.

### Key Characteristics

- **Snapshot format**: Qdrant collection snapshots are gzipped tar archives (`.snapshot`) containing data and configuration.
- **Consistency**: Snapshots are point-in-time and reflect the state at creation time.
- **Storage**: Snapshots are stored locally on the Qdrant node. For production, they should be copied to S3/GCS for durability.
- **Restore behavior**: Restoring a snapshot **overwrites** the existing collection data. Use with caution in production.

### Backup Schedule

| Frequency | Type | Retention |
|-----------|------|-----------|
| Hourly (via Qdrant auto-snapshot) | Automatic | 7 days |
| Daily (manual) | `backup_qdrant.py` → S3 | 90 days |
| Pre-deployment | `backup_qdrant.py` → local | Until replaced |

---

## Backup Procedure

### Prerequisites

- Qdrant is running and accessible (`QDRANT_URL`, `QDRANT_API_KEY` set)
- `qdrant-client` Python package installed
- Sufficient disk space on the Qdrant node for snapshot storage

### 1. Manual Backup (backup_qdrant.py)

The `scripts/backup_qdrant.py` script creates point-in-time snapshots of all collections.

```bash
# Install dependencies (should already be installed via uv sync)
uv sync

# Backup all collections to default ./data/backups directory:
python scripts/backup_qdrant.py

# Backup to a custom directory:
python scripts/backup_qdrant.py --output-dir /mnt/backups/qdrant

# Backup specific collections only:
python scripts/backup_qdrant.py --collections pages_visual

# Dry-run (list collections but don't create snapshots):
python scripts/backup_qdrant.py --dry-run
```

**Output structure:**
```
data/backups/
└── run_20260428_120000/           ← timestamped run directory
    ├── backup_metadata.json        ← snapshot registry
    ├── pages_visual__snapshot-abc123.snapshot   ← collection snapshot
    └── pages_text__snapshot-def456.snapshot
```

**Metadata file contents:**
```json
{
  "backup_id": "20260428_120000",
  "backup_at": "2026-04-28T12:00:00+00:00",
  "collections_backed_up": ["pages_visual", "pages_text"],
  "snapshots": [
    {
      "snapshot_name": "snapshot-abc123",
      "collection_name": "pages_visual",
      "local_path": "data/backups/run_20260428_120000/pages_visual__snapshot-abc123.snapshot",
      "size_bytes": 1073741824,
      "checksum": "sha256:abc123...",
      "created_at": "2026-04-28T12:00:05+00:00"
    }
  ]
}
```

### 2. Copy to Cloud Storage (Production)

After creating local snapshots, copy them to cloud storage for durability:

```bash
# Example: Copy to S3
aws s3 cp --recursive data/backups/ s3://urban-rag-backups/qdrant/

# Or with rclone for multi-cloud support
rclone copy data/backups/ remote:urban-rag-backups/qdrant/
```

### 3. Verify Backup Integrity

```bash
# Run smoke test to verify collections are intact after backup
python scripts/smoke_qdrant_cloud.py

# Compare point counts before and after backup:
curl -s http://localhost:6333/collections/pages_visual/points | jq '.result.points_count'
```

---

## Restore Procedure

### Prerequisites

- Qdrant is running and accessible
- Backup directory contains snapshot files and `backup_metadata.json`
- Collection schemas must exist (they will be auto-created for known collections)

### 1. Restore from Local Backup

The `scripts/restore_qdrant.py` script restores collections from local snapshot backups.

```bash
# Restore all collections from latest backup:
python scripts/restore_qdrant.py

# Restore from a specific backup directory:
python scripts/restore_qdrant.py --backup-dir data/backups/run_20260428_120000

# Restore specific collections:
python scripts/restore_qdrant.py --collections pages_visual

# Dry-run (validate backup without restoring):
python scripts/restore_qdrant.py --dry-run

# Restore without waiting for completion (faster, less safe):
python scripts/restore_qdrant.py --no-wait

# Force overwrite existing collections:
python scripts/restore_qdrant.py --force
```

### 2. Restore from Cloud Storage (Production)

If backups are stored in cloud storage, download them first:

```bash
# Download from S3
aws s3 cp --recursive s3://urban-rag-backups/qdrant/ data/backups/

# Then restore
python scripts/restore_qdrant.py --backup-dir data/backups/run_20260428_120000
```

### 3. Verify Restore Success

After restore, verify the collections are healthy:

```bash
# Check collection info
curl -s http://localhost:6333/collections/pages_visual | jq '.result'

# Run smoke test
python scripts/smoke_qdrant_cloud.py

# Verify point counts match expected values
curl -s http://localhost:6333/collections/pages_visual/points | jq '.result.points_count'
curl -s http://localhost:6333/collections/pages_text/points | jq '.result.points_count'
```

---

## Restore Drill (Staging Environment)

A scheduled restore drill should be performed monthly to verify backup/restore operability.

### Step-by-Step Drill

```bash
# 1. Create a fresh backup of production
python scripts/backup_qdrant.py --output-dir data/backups/drill_$(date +%Y%m%d)

# 2. Deploy a staging Qdrant instance
docker run -d --name urban-rag-qdrant-staging -p 3104:6333 qdrant/qdrant

# 3. Restore to staging
QDRANT_URL=http://localhost:3104 python scripts/restore_qdrant.py \
  --backup-dir data/backups/drill_20260428 \
  --collections pages_visual,pages_text

# 4. Run smoke health checks against staging
QDRANT_URL=http://localhost:3104 QDRANT_API_KEY= python scripts/smoke_qdrant_cloud.py

# 5. Verify collection point counts match production
# Compare counts from production vs staging

# 6. Clean up staging
docker stop urban-rag-qdrant-staging && docker rm urban-rag-qdrant-staging
```

### Restore Drill Verification Checklist

- [ ] Staging Qdrant starts successfully
- [ ] `restore_qdrant.py --dry-run` passes for all collections
- [ ] `restore_qdrant.py` completes without errors
- [ ] Collection point counts match between production and staging
- [ ] Smoke test passes against restored staging instance
- [ ] No data loss or corruption observed
- [ ] Drill completes within 30-minute SLO

---

## Val-OPS-014: Backup Restore Path Validation

The `VAL-OPS-014` assertion requires that "A scheduled restore drill can recover Qdrant snapshot and metadata DB to a staging environment and pass smoke health checks."

**Verification steps:**
1. Run `python scripts/backup_qdrant.py --output-dir /tmp/val_backup`
2. Deploy staging Qdrant on port 3104
3. Run `QDRANT_URL=http://localhost:3104 python scripts/restore_qdrant.py --backup-dir /tmp/val_backup --dry-run`
4. Run `QDRANT_URL=http://localhost:3104 QDRANT_API_KEY= python scripts/smoke_qdrant_cloud.py`
5. Verify all checks pass and restore completes within 30 minutes

---

## Troubleshooting

### Snapshot creation fails with "Collection not found"

**Cause**: The collection doesn't exist yet in Qdrant.

**Resolution**: Ensure collections are created first (via the indexing pipeline or `create_collections()` in `src/urban_rag/index/qdrant_client.py`).

### Restore fails with "Collection already exists"

**Cause**: Attempting to restore over an existing collection without `--force`.

**Resolution**: Use `--force` flag to overwrite existing collections, or delete the collection first:
```bash
curl -X DELETE http://localhost:6333/collections/pages_visual
```

### Snapshot download returns empty file

**Cause**: Qdrant's `download_snapshot` may not work in all configurations (e.g., Qdrant Cloud with signed URLs).

**Resolution**: For Qdrant Cloud, snapshots are accessible via signed URLs. Use `recover_from_url` with the snapshot URL instead of local file path. See [Qdrant Cloud documentation](https://qdrant.tech/documentation/cloud/backup/) for the correct procedure.

### Checksum mismatch on restore

**Cause**: Snapshot file was corrupted or incompletely downloaded.

**Resolution**: Re-run the backup to create a fresh snapshot, then re-download. Verify the backup succeeded before copying to cloud storage.

### Restore exceeds 30-minute SLO

**Cause**: Large collections with millions of points take time to restore.

**Resolution**: For large production collections, consider using Qdrant's incremental backup approach or streaming restore. Ensure the staging environment has equivalent resources (CPU, memory, disk IOPS) to production.

---

## Related Documents

- [VAL-OPS-014 assertion specification](../validation-contract.md#VAL-OPS-014)
- [rollback.md](./rollback.md) — Lightning deployment rollback procedure
- [smoke_qdrant_cloud.py](../../scripts/smoke_qdrant_cloud.py) — Qdrant smoke test
- [qdrant_client.py](../../src/urban_rag/index/qdrant_client.py) — Collection schema management
- [Qdrant snapshot documentation](https://qdrant.tech/documentation/database-tutorials/create-snapshot/)
