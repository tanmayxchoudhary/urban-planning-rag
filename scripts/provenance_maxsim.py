"""Phase 0 — Tomoro provenance MaxSim gate (G1).

Honest provenance gate using the OCR-verified query set from the
visual-rag-provenance skill reference. The metadata does not contain native
page_id fields, so expected pages are asserted by exact (source, page) match,
and the external page_id is derived as {source_stem}#p{NNNN}.

Gate rule:
- pass if at least 4/5 expected pages appear in the top 5 results
- fail closed to REEMBED otherwise

Run:
  cd /root/projects/urban-planning-rag && .venv/bin/modal run scripts/provenance_maxsim.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import modal

VOLUME_NAME = "urban-rag-g1"
TENSOR_PATH = "/data/embeddings_tomoro320_full.pt"
META_PATH = "/data/metadata_tomoro320_full.json"
MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"
REVISION = "bf790bd8780b098b86453444632a184bb770be1a"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy==2.4.6",
    "pillow==12.2.0",
    "torch==2.10.0",
    "torchvision==0.25.0",
    "transformers==4.57.6",
    "accelerate==1.13.0",
)

app = modal.App("tomoro-provenance")
volume = modal.Volume.from_name(VOLUME_NAME)

KNOWN_QUERIES = [
    {
        "id": "vol1_p0045_planning_process",
        "question": "What does URDPFI say about simplifying the planning process and modifying Town and Country Planning Acts?",
        "expected_source": "urdpfi_vol1.pdf",
        "expected_page": 45,
    },
    {
        "id": "vol1_p0080_transport_plans",
        "question": "Which transport plans does URDPFI list for road network development, including hierarchical road network and arterial road construction?",
        "expected_source": "urdpfi_vol1.pdf",
        "expected_page": 80,
    },
    {
        "id": "vol1_p0100_ews_far",
        "question": "What percentage of FAR does URDPFI cite in DDA norms for private developers providing EWS housing?",
        "expected_source": "urdpfi_vol1.pdf",
        "expected_page": 100,
    },
    {
        "id": "vol2_p0080_environment_clearances",
        "question": "What statutory clearances does URDPFI list for environment clearances under EIA notification 2006?",
        "expected_source": "urdpfi_vol2.pdf",
        "expected_page": 80,
    },
    {
        "id": "vol2_p0150_sustainability_definitions",
        "question": "How does URDPFI define buffer zones and climate change in Appendix B sustainability definitions?",
        "expected_source": "urdpfi_vol2.pdf",
        "expected_page": 150,
    },
]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_stem(source: str) -> str:
    return source[:-4] if source.lower().endswith(".pdf") else source


def _page_number(meta: dict) -> int:
    raw = meta.get("page", meta.get("page_number"))
    if raw is None:
        raise ValueError(f"metadata row missing page number: {meta}")
    return int(raw)


def _derive_page_id(meta: dict) -> str:
    return f"{_source_stem(str(meta['source']))}#p{_page_number(meta):04d}"


def _score_multi_vector(q_emb, p_embs):
    """Vectorized MaxSim consistent with modal_search_service.py."""
    import torch

    q = torch.nn.functional.normalize(q_emb[0].float(), p=2, dim=-1)
    p = torch.nn.functional.normalize(p_embs.float(), p=2, dim=-1)
    sims = torch.einsum("qd,pkd->pqk", q, p)
    max_per_q = torch.max(sims, dim=-1).values
    scores = torch.sum(max_per_q, dim=-1)
    return scores


@app.function(image=image, volumes={"/data": volume}, gpu="any", timeout=600)
def run_provenance() -> dict:
    import torch
    from transformers import AutoModel, AutoProcessor

    print(f"Loading tensor: {TENSOR_PATH}")
    # Safe here: private asset on our own Modal volume, not user-supplied.
    embeddings = torch.load(TENSOR_PATH, map_location="cpu", weights_only=False)

    print(f"Loading processor/model: {MODEL_ID}@{REVISION}")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Metadata not found at {META_PATH}. Refusing provenance gate without exact row mapping."
        )
    with open(META_PATH) as f:
        metadata = json.load(f)
    if len(metadata) != int(embeddings.shape[0]):
        raise ValueError(
            f"Metadata rows ({len(metadata)}) != tensor rows ({int(embeddings.shape[0])}); aborting provenance gate."
        )

    expected_lookup: dict[tuple[str, int], int] = {}
    for idx, meta in enumerate(metadata):
        expected_lookup[(str(meta.get("source", "")), _page_number(meta))] = idx

    query_results = []
    top1_hits = 0
    top5_hits = 0

    for item in KNOWN_QUERIES:
        expected_key = (item["expected_source"], int(item["expected_page"]))
        if expected_key not in expected_lookup:
            raise ValueError(f"Expected page missing from metadata: {expected_key}")
        expected_idx = expected_lookup[expected_key]
        expected_meta = metadata[expected_idx]
        expected_page_id = _derive_page_id(expected_meta)

        batch = processor.process_texts([item["question"]])
        batch = {
            k: v.to(model.device)
            for k, v in batch.items()
            if hasattr(v, "to")
        }
        with torch.no_grad():
            outputs = model(**batch)
            q_emb = outputs.embeddings
            if q_emb.dim() == 2:
                q_emb = q_emb.unsqueeze(0)

        scores = _score_multi_vector(q_emb.cpu(), embeddings)
        top_scores, top_idx = torch.topk(scores, k=min(5, scores.shape[0]))
        sorted_idx = torch.argsort(scores, descending=True)
        expected_rank = int((sorted_idx == expected_idx).nonzero(as_tuple=True)[0][0].item()) + 1
        expected_score = float(scores[expected_idx].item())

        top_candidates = []
        for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_idx.tolist()), start=1):
            meta = metadata[idx]
            top_candidates.append(
                {
                    "rank": rank,
                    "row_idx": idx,
                    "page_id": _derive_page_id(meta),
                    "source": meta["source"],
                    "page": _page_number(meta),
                    "score": round(float(score), 4),
                }
            )

        top1_hit = bool(top_idx[0].item() == expected_idx)
        top5_hit = bool(any(int(idx) == expected_idx for idx in top_idx.tolist()))
        if top1_hit:
            top1_hits += 1
        if top5_hit:
            top5_hits += 1

        print(
            f"{item['id']}: expected={expected_page_id} rank={expected_rank} top1={top1_hit} top5={top5_hit}"
        )
        query_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_page_id": expected_page_id,
                "expected_row_idx": expected_idx,
                "expected_rank": expected_rank,
                "expected_score": round(expected_score, 4),
                "top1_hit": top1_hit,
                "top5_hit": top5_hit,
                "top_candidates": top_candidates,
            }
        )

    report = {
        "gate": "G1_PROVENANCE",
        "started_at": _utcnow(),
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "tensor_name": "embeddings_tomoro320_full.pt",
        "legacy_tensor_shape": list(embeddings.shape),
        "known_queries": len(KNOWN_QUERIES),
        "top1_hits": top1_hits,
        "top5_hits": top5_hits,
        "pass_threshold": "4/5 top5",
        "passed": top5_hits >= 4,
        "decision": "REUSE" if top5_hits >= 4 else "REEMBED",
        "query_results": query_results,
        "finished_at": _utcnow(),
    }
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main() -> None:
    result = run_provenance.remote()
    print(json.dumps(result, indent=2, default=str))
