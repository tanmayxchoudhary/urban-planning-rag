# Tomoro v1 Runtime — Urban RAG Phase 0 Recovery

**Status:** Phase 0 IN PROGRESS — reconstruction drafted, not yet live-verified.
Reconstructed runtime scripts from log receipts (no in-tree Tomoro runtime existed
prior; only logs mentioned it). The reconstruction reproduces the exact shapes/outputs
recorded in the logs, but the PLAN's Phase 0 gate is **not** yet met.

## Phase 0 gate status (per PLAN.md)
- Retriever source in git: ✗ scripts still show as untracked in `git status`
- Fresh recursive volume inventory receipt: ✗ only stale `modal_volume_ls.log` exists
- Canonical tensor named: ✓ `embeddings_tomoro320_full.pt`
- Live reconstructed search run from committed code: ✗ not re-run this pass
- Page-id mapping assertion: ✗ not asserted live this pass

Do **not** mark Phase 0 complete until every item above is green with fresh receipts in
`launch/artifacts/`.

## Ground Truth
- v1 uses **TomoroAI/tomoro-colqwen3-embed-4b** (revision `bf790bd8780b098b86453444632a184bb770be1a`) with in-memory MaxSim over the legacy 738-row visual tensor.
- Tensor shape: `[738, 1271, 320]`
- Corpus: 738 legacy rows (`urdpfi_vol1`: 447, `urdpfi_vol2`: 250, `swm_2016`: 41) as defined in `docs/corpus_ledger.md`. 5 extra PNGs are test pages and excluded.
- **No Qdrant, no hybrid (ColQwen2.5 + GTE + BM25 + RRF), no 8B models** for v1. Modal scale-to-zero in-memory retrieval only.
- Receipts: `launch/artifacts/g3_modal_search_test2.log`, `launch/artifacts/provenance_tomoro320_full.log`
- Canonical plan: `PLAN.md`

## Runtime Scripts (Reconstructed)
- `scripts/modal_search_service.py`: Defines `TomoroSearch` Modal class. Uses `process_texts` + model forward + `score_multi_vector` (MaxSim). Returns structured results with `page_id`, `source`, `page`, `score`, `image_path`. Matches output in `g3_modal_search_test2.log` (e.g., road-widths query, ~1.37s, scores ~13.68).
- `scripts/provenance_maxsim.py`: Defines `run_provenance`. Loads tensor + model, runs 5 known queries, verifies rank-1 hits, emits `G1_PROVENANCE` gate JSON (5/5 hits, `REUSE` decision). Matches `provenance_tomoro320_full.log`.
- `scripts/modal_verify_g1.py`: Existing CPU-only verifier for volume `urban-rag-g1` integrity (tensors, metadata, page images). Kept sane and consistent; references Tomoro 320-d paths.

## Volume Layout (urban-rag-g1)
- `/data/embeddings_tomoro320_full.pt`
- `/data/metadata_tomoro320_full.json`
- `/data/page_images/` (738 legacy PNGs + 5 tests)
- (Legacy paths also present for compatibility)

## Query Contract
```python
service.search.remote("What does URDPFI say about road widths?", top_k=3)
# returns: {query, top_k, model, revision, results: [{rank, page_id, source, page, score, image_path}], seconds}
```

## Provenance Gate (G1)
- 5 known queries must achieve rank=1 hit on expected pages.
- Threshold: 4/5 → `passed: true`, `decision: "REUSE"`
- If fails → re-embed path (not taken).

## How v1 Differs from April Hybrid (README)
The April hybrid architecture (visual ColQwen2.5 + text GTE-ModernColBERT + BM25 sparse, Qdrant collections, RRF k=60 fusion, Gemini 2.5 Flash rerank) is **not** the live v1 path. It is stale/broken and archived for v1. 

v1 is deliberately minimal: pure visual Tomoro MaxSim on Modal for fast green demo with grounded citations. Full hybrid, Qdrant, and generation layers come later (post-v1).

## Readiness
- Phase 0: scripts reconstructed + syntax verified + consistent with logs/PLAN.
- Phase 0 gate is **not yet met**: the scripts are still untracked, no fresh live Modal receipts were produced this pass, and the page-id mapping was not asserted from live artifacts.
- Current state: honest recovery groundwork only. Adversarial eval/API/generation cleanup exists, but no claim of a live public demo or production-ready retrieval should be made yet.
- Do not overclaim: no live public demo, no Qdrant index, retrieval quality not yet measured beyond smoke/provenance receipts.

See `PLAN.md` for locked decisions and build order.
