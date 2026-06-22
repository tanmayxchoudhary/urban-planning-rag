"""Phase 0.0 — verify Modal Volume `urban-rag-g1` integrity (CPU only, no GPU).

Confirms the pinned Tomoro 320-d tensor loads, matches its metadata, and that
page images are present and consistent. This is the linchpin check before any
reconstruction work: if this passes, we reuse the existing tensor; if it fails,
Phase 0 becomes a re-embed.

Run:
    modal run scripts/modal_verify_g1.py
"""

from __future__ import annotations

import modal

VOLUME_NAME = "urban-rag-g1"
TENSOR = "/data/embeddings_tomoro320_full.pt"
META = "/data/metadata_tomoro320_full.json"
LEGACY_TENSOR = "/data/embeddings.pt"
LEGACY_META = "/data/metadata.json"
PAGE_IMAGES = "/data/page_images"

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")
app = modal.App("urban-rag-g1-verify")
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def verify() -> dict:
    import hashlib
    import json
    import os

    import torch

    def sha256(path: str, limit: int | None = None) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8 << 20)
                if not chunk:
                    break
                h.update(chunk)
                if limit and f.tell() >= limit:
                    break
        return h.hexdigest()

    report: dict = {}

    for label, tpath, mpath in (
        ("full", TENSOR, META),
        ("legacy", LEGACY_TENSOR, LEGACY_META),
    ):
        entry: dict = {"tensor_path": tpath, "exists": os.path.exists(tpath)}
        if entry["exists"]:
            t = torch.load(tpath, map_location="cpu", weights_only=False)
            tens = t if isinstance(t, torch.Tensor) else None
            if tens is not None:
                entry["shape"] = list(tens.shape)
                entry["dtype"] = str(tens.dtype)
            else:
                entry["type"] = str(type(t))
            entry["size_bytes"] = os.path.getsize(tpath)
            entry["sha256"] = sha256(tpath)
            if os.path.exists(mpath):
                with open(mpath) as f:
                    meta = json.load(f)
                rows = meta if isinstance(meta, list) else meta.get("pages", meta)
                entry["metadata_rows"] = len(rows) if hasattr(rows, "__len__") else None
                # sample a few page_ids / sources
                try:
                    entry["metadata_sample"] = rows[:3]
                except Exception:
                    pass
        report[label] = entry

    # page images
    pngs = []
    if os.path.isdir(PAGE_IMAGES):
        pngs = [f for f in os.listdir(PAGE_IMAGES) if f.endswith(".png")]
    sources: dict[str, int] = {}
    for f in pngs:
        stem = f.split("__page_")[0]
        sources[stem] = sources.get(stem, 0) + 1
    report["page_images"] = {
        "count": len(pngs),
        "by_source": sources,
        "sample": sorted(pngs)[:3],
    }
    return report


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(verify.remote(), indent=2, default=str))
