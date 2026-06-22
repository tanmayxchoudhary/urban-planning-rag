"""Phase 0 live verification — Urban RAG v1.

Single Modal job that produces fresh, auditable receipts for the Phase 0 gate.
It verifies the asset and the citation contract only; it makes NO claim about
retrieval quality (that is Phase 1).

Ground truth verified live on 2026-06-21:
- metadata_tomoro320_full.json contains only {source, page, total_pages}
- it does NOT contain native page_id or image_path fields
- the real citation contract is derived deterministically from source+page:
    page_id   = {source_stem}#p{NNNN}
    png_name  = {source_stem}__page_{NNNN}.png

This script exports:
1. Recursive volume inventory (with sizes) written to /data/phase0_volume_inventory.json
2. Canonical tensor facts + metadata facts
3. Full per-row mapping written to /data/phase0_row_mapping.json
4. Live uniqueness/existence assertions over derived page_id + PNG path

Everything heavy runs on Modal (remote). Nothing is installed locally.

Run:
  cd /root/projects/urban-planning-rag && .venv/bin/modal run scripts/phase0_verify.py
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone

import modal

VOLUME_NAME = "urban-rag-g1"
TENSOR_PATH = "/data/embeddings_tomoro320_full.pt"
META_PATH = "/data/metadata_tomoro320_full.json"
PAGE_IMAGES_DIR = "/data/page_images"
INVENTORY_RECEIPT = "/data/phase0_volume_inventory.json"
ROW_MAPPING_RECEIPT = "/data/phase0_row_mapping.json"

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")
app = modal.App("urban-rag-phase0-verify")
volume = modal.Volume.from_name(VOLUME_NAME)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _walk(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            out.append(os.path.join(dirpath, name))
    return sorted(out)


def _source_stem(source: str) -> str:
    return source[:-4] if source.lower().endswith(".pdf") else source


def _page_number(meta: dict) -> int | None:
    raw = meta.get("page", meta.get("page_number"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _derive_page_id(meta: dict) -> str:
    source = str(meta.get("source", ""))
    page = _page_number(meta)
    if not source or page is None:
        return ""
    return f"{_source_stem(source)}#p{page:04d}"


def _derive_png_name(meta: dict) -> str:
    source = str(meta.get("source", ""))
    page = _page_number(meta)
    if not source or page is None:
        return ""
    return f"{_source_stem(source)}__page_{page:04d}.png"


def _dups(values: list[str]) -> list[str]:
    return [k for k, c in Counter(v for v in values if v).items() if c > 1]


@app.function(image=image, volumes={"/data": volume}, timeout=900)
def verify() -> dict:
    import torch

    report: dict = {
        "gate": "PHASE0_VERIFY",
        "started_at": _utcnow(),
        "volume": VOLUME_NAME,
    }

    # --- 1. Recursive volume inventory ---
    top = sorted(os.listdir("/data"))
    all_files = _walk("/data")
    inventory = []
    for path in all_files:
        rel = os.path.relpath(path, "/data")
        try:
            size = os.path.getsize(path)
        except OSError:
            size = None
        inventory.append({"path": rel, "size_bytes": size})
    report["volume_top_level"] = top
    report["volume_recursive_file_count"] = len(inventory)
    report["volume_inventory_head"] = inventory[:10]
    report["volume_inventory_tail"] = inventory[-10:]

    # --- 2. Canonical tensor + metadata facts ---
    if not os.path.exists(TENSOR_PATH):
        report["error"] = f"tensor missing at {TENSOR_PATH}"
        report["passed"] = False
        return report

    # Safe here: private asset on our own Modal volume, not user-supplied.
    embeddings = torch.load(TENSOR_PATH, map_location="cpu", weights_only=False)
    report["tensor_name"] = os.path.basename(TENSOR_PATH)
    report["tensor_shape"] = list(embeddings.shape)
    report["tensor_dtype"] = str(embeddings.dtype)
    tensor_rows = int(embeddings.shape[0])
    try:
        head = embeddings[0].float().flatten()[:1024].numpy().tobytes()
        report["tensor_row0_head_sha256"] = hashlib.sha256(head).hexdigest()
    except Exception as exc:  # noqa: BLE001
        report["tensor_row0_head_sha256"] = f"err: {exc}"

    if not os.path.exists(META_PATH):
        report["error"] = f"metadata missing at {META_PATH}"
        report["passed"] = False
        return report
    with open(META_PATH, "rb") as f:
        raw = f.read()
    metadata = json.loads(raw)
    report["metadata_name"] = os.path.basename(META_PATH)
    report["metadata_sha256"] = hashlib.sha256(raw).hexdigest()
    report["metadata_rows"] = len(metadata)
    report["metadata_keys"] = sorted(metadata[0].keys()) if metadata else []
    sample_idxs = [0, 45, 80, 100, len(metadata) - 1] if len(metadata) > 100 else [0]
    report["metadata_samples"] = {
        str(i): metadata[i] for i in sample_idxs if 0 <= i < len(metadata)
    }
    report["distinct_sources"] = sorted({str(m.get("source", "")) for m in metadata})

    # --- 3. Full per-row mapping + contract assertions ---
    png_files = []
    if os.path.isdir(PAGE_IMAGES_DIR):
        png_files = [p for p in all_files if p.startswith(PAGE_IMAGES_DIR) and p.lower().endswith(".png")]
    png_basenames = {os.path.basename(p) for p in png_files}
    report["page_image_png_count"] = len(png_files)

    rows = []
    page_ids: list[str] = []
    png_names: list[str] = []
    missing_png: list[str] = []
    for row_idx, meta in enumerate(metadata):
        page_id = _derive_page_id(meta)
        png_name = _derive_png_name(meta)
        png_exists = png_name in png_basenames if png_name else False
        if png_name and not png_exists:
            missing_png.append(png_name)
        page_ids.append(page_id)
        png_names.append(png_name)
        rows.append(
            {
                "row_idx": row_idx,
                "page_id": page_id,
                "source": str(meta.get("source", "")),
                "page_number": _page_number(meta),
                "png": png_name,
                "png_exists": png_exists,
            }
        )

    checks = {
        "tensor_rows == metadata_rows": tensor_rows == len(metadata),
        "png_count >= metadata_rows": len(png_files) >= len(metadata),
        "page_id_unique": len(_dups(page_ids)) == 0,
        "page_id_nonempty_all": all(page_ids) and len(page_ids) == len(metadata),
        "png_path_unique": len(_dups(png_names)) == 0,
        "all_metadata_pngs_exist": len(missing_png) == 0,
    }
    report["checks"] = checks
    report["dup_page_id_examples"] = _dups(page_ids)[:10]
    report["dup_png_examples"] = _dups(png_names)[:10]
    report["missing_png_examples"] = missing_png[:10]
    report["row_mapping_head"] = rows[:5]
    report["row_mapping_tail"] = rows[-3:]

    # Durable receipts on volume
    with open(INVENTORY_RECEIPT, "w") as f:
        json.dump(inventory, f, indent=2)
    with open(ROW_MAPPING_RECEIPT, "w") as f:
        json.dump(rows, f, indent=2)
    volume.commit()
    report["volume_inventory_receipt"] = INVENTORY_RECEIPT
    report["row_mapping_receipt"] = ROW_MAPPING_RECEIPT

    report["passed"] = all(checks.values())
    report["finished_at"] = _utcnow()
    return report


@app.local_entrypoint()
def main() -> None:
    result = verify.remote()
    print(json.dumps(result, indent=2, default=str))
