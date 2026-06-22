"""Phase 0 — Tomoro v1 search service (Modal in-memory MaxSim).

Reconstructed from log evidence and corrected against live metadata receipts.
The metadata does not contain native page_id/image_path fields; those are
DERIVED deterministically from source+page at runtime.

Derived contract:
- page_id   = {source_stem}#p{NNNN}
- image_path = /page_images/{source_stem}__page_{NNNN}.png

Run:
  cd /root/projects/urban-planning-rag && .venv/bin/modal run scripts/modal_search_service.py
"""

from __future__ import annotations

import json
import os
import time

import modal

VOLUME_NAME = "urban-rag-g1"
TENSOR_PATH = "/data/embeddings_tomoro320_full.pt"
META_PATH = "/data/metadata_tomoro320_full.json"
MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"
REVISION = "bf790bd8780b098b86453444632a184bb770be1a"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "pillow==12.2.0",
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==4.57.6",
        "accelerate==1.13.0",
    )
)

app = modal.App("tomoro-search")
volume = modal.Volume.from_name(VOLUME_NAME)


def _source_stem(source: str) -> str:
    return source[:-4] if source.lower().endswith(".pdf") else source


def _page_number(meta: dict) -> int:
    raw = meta.get("page", meta.get("page_number"))
    if raw is None:
        raise ValueError(f"metadata row missing page number: {meta}")
    return int(raw)


def _derive_page_id(meta: dict) -> str:
    return f"{_source_stem(str(meta['source']))}#p{_page_number(meta):04d}"


def _derive_image_path(meta: dict) -> str:
    return f"/page_images/{_source_stem(str(meta['source']))}__page_{_page_number(meta):04d}.png"


@app.cls(image=image, volumes={"/data": volume}, gpu="any", timeout=600)
class TomoroSearch:
    """Tomoro ColQwen3-4B search over the legacy 738-row tensor."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        load_start = time.time()
        print(f"Loading processor/model: {MODEL_ID}@{REVISION}")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        load_time = time.time() - load_start
        print(f"TomoroSearch loaded in {load_time:.2f}s")

        print(f"Loading tensor: {TENSOR_PATH}")
        # Safe here: private asset on our own Modal volume, not user-supplied.
        self.embeddings = torch.load(
            TENSOR_PATH, map_location="cpu", weights_only=False
        )

        if not os.path.exists(META_PATH):
            raise FileNotFoundError(
                f"Metadata not found at {META_PATH}. Refusing to fabricate citations."
            )
        with open(META_PATH) as f:
            self.metadata = json.load(f)
        if len(self.metadata) != int(self.embeddings.shape[0]):
            raise ValueError(
                f"Metadata rows ({len(self.metadata)}) != tensor rows "
                f"({int(self.embeddings.shape[0])}); aborting to avoid false citations."
            )

    def _score_multi_vector(
        self, q_emb: "torch.Tensor", p_embs: "torch.Tensor"
    ) -> "torch.Tensor":
        """Vectorized MaxSim consistent with ColQwen/ColPali late interaction."""
        import torch

        q = torch.nn.functional.normalize(q_emb[0].float(), p=2, dim=-1)
        p = torch.nn.functional.normalize(p_embs.float(), p=2, dim=-1)
        sims = torch.einsum("qd,pkd->pqk", q, p)
        max_per_q = torch.max(sims, dim=-1).values
        scores = torch.sum(max_per_q, dim=-1)
        return scores

    @modal.method()
    def search(self, query: str, top_k: int = 3) -> dict:
        import torch

        start = time.time()
        batch = self.processor.process_texts([query])
        batch = {
            k: v.to(self.model.device)
            for k, v in batch.items()
            if hasattr(v, "to")
        }

        with torch.no_grad():
            outputs = self.model(**batch)
            q_emb = outputs.embeddings
            if q_emb.dim() == 2:
                q_emb = q_emb.unsqueeze(0)

        scores = self._score_multi_vector(q_emb.cpu(), self.embeddings)
        k = min(top_k, scores.shape[0])
        top_scores, top_idx = torch.topk(scores, k)

        results = []
        for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_idx.tolist()), start=1):
            meta = self.metadata[idx]
            results.append(
                {
                    "rank": rank,
                    "row_idx": idx,
                    "page_id": _derive_page_id(meta),
                    "source": meta["source"],
                    "page": _page_number(meta),
                    "score": round(float(score), 4),
                    "image_path": _derive_image_path(meta),
                }
            )

        return {
            "query": query,
            "top_k": top_k,
            "model": MODEL_ID,
            "revision": REVISION,
            "results": results,
            "seconds": round(time.time() - start, 6),
        }


@app.local_entrypoint()
def main() -> None:
    service = TomoroSearch()
    result = service.search.remote("What does URDPFI say about road widths?", top_k=3)
    print(json.dumps(result, indent=2, default=str))
