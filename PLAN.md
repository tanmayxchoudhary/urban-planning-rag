# Urban RAG v1 — Final Execution Plan

> **Purpose:** This is the execution-grade source of truth for getting Urban RAG from its current broken repo state to a real green demo.
> **Status:** Finalized by Niney + Claudey on 2026-06-19 from live repo inspection, log receipts, and adversarial review.
> **Boundary:** This plan is for **v1 green demo only**. The broader ship vision lives separately and must not pollute v1 execution.

---

## 0. TL;DR

- The **April hybrid in the repo is not the path**. It is stale, broken, and not to be debugged for v1.
- The **Tomoro 4B visual retriever is the right v1 direction**, but its **source code is not in the repo** today. We only have run logs and a local verifier script.
- The **738-row legacy tensor is the v1 corpus**: 447 `urdpfi_vol1` + 250 `urdpfi_vol2` + 41 `swm_2016`. The extra 5 PNGs are test pages and are **not** part of the embedded corpus.
- The first real task is **not UI work**. It is **Phase 0: reconstruct + re-verify the retriever from logs, then commit the missing source and receipts**.
- v1 runtime is locked to **Modal in-memory MaxSim** over the existing tensor. **No Qdrant. No hybrid. No 8B churn.**
- v1 is judged by one thing: **does it answer real in-corpus questions with grounded citations and refuse out-of-corpus bullshit cleanly?**
- The build order is locked: **recover asset → adversarial eval → generation → API → web → public**.
- Nothing gets marked "verified" without a live receipt in the repo.

---

## 1. Ground truth

### 1.1 Reality table

| Area | State | Confidence | Receipt | What it means |
|---|---|---|---|---|
| README architecture | Still describes April hybrid (`ColQwen2.5 + GTE + BM25 + Qdrant + RRF + Gemini`) | **verified-live** | `README.md` | README is stale and misleading. |
| Current repo runtime path | April hybrid code is what's committed under `src/urban_rag/` | **verified-live** | `src/urban_rag/`, `README.md`, `common/settings.py` | Current repo path is not the v1 path. |
| Tomoro retriever source | `TomoroSearch`, `process_texts`, `score_multi_vector`, `modal_search_service.py`, `provenance_maxsim.py` are **absent from the repo** | **verified-live** | repo search + `git status` + log mount paths | Phase 0 is a reconstruction from logs, not a simple wiring task. |
| Modal verifier | `scripts/modal_verify_g1.py` exists locally but is **untracked** | **verified-live** | `git status` | Stop claiming it is committed until it actually is. |
| One real search smoke | Road-width query returned relevant pages in **1.37s** | **log-only** | `launch/artifacts/g3_modal_search_test2.log` | Encouraging but not proof of retrieval quality. |
| Score stability | Rank 1 and rank 2 tied at **13.6875** on the smoke query | **log-only** | `g3_modal_search_test2.log` | Gate-A margin may be weak or unusable; must be calibrated, not assumed. |
| Provenance smoke | 5/5 known-query hits on `embeddings_tomoro320_full.pt` | **log-only** | `launch/artifacts/provenance_tomoro320_full.log` | Good provenance smoke, **not** a real eval. |
| Live volume listing | Stale listing shows only `metadata.json` and `embeddings.pt` | **stale-log** | `launch/artifacts/modal_volume_ls.log` | Fresh volume inventory is required in Phase 0. |
| Legacy corpus definition | 738 embedded rows = 447 `urdpfi_vol1` + 250 `urdpfi_vol2` + 41 `swm_2016`; 5 extra PNGs are tests | **verified-live** | `docs/corpus_ledger.md` | v1 corpus must be explicitly defined as these 738 rows. |
| Local data state | Current April ingest outputs are not ready for visual retrieval; many blank `image_uri` values | **verified-live** | `docs/corpus_ledger.md`, `data/docs/*` | Do not build v1 on current ingest outputs. |
| CI/CD reality | `.github/` does not exist at current HEAD | **verified-live** | repo search | Ignore old claims of working CI/CD until reintroduced later. |

### 1.2 v1 corpus definition

For v1, the active corpus is locked to the **existing 738-row legacy visual tensor**:

- `urdpfi_vol1`: 447 pages
- `urdpfi_vol2`: 250 pages
- `swm_2016`: 41 pages

Important:
- This is **not** "738 URDPFI pages" in a strict sense.
- It is **697 URDPFI pages + 41 SWM pages**.
- `743 PNGs` includes **5 test pages** that must be excluded from the citation-serving path.

### 1.3 Current thesis

The honest thesis is:

1. The repo's committed runtime is the wrong system for v1.
2. The Tomoro direction is correct.
3. The tensor likely survives and one smoke query looks promising.
4. The retriever **must be reconstructed from logs and then re-verified live** before anything else.
5. Retrieval quality is **not yet measured**; it must earn trust through the adversarial eval, not through hand-picked log receipts.

---

## 2. Locked v1 decisions

| Decision | Locked choice | Reason |
|---|---|---|
| Retriever model | `TomoroAI/tomoro-colqwen3-embed-4b` @ `bf790bd8` | Correct direction, existing tensor/log receipts, no model churn before green. |
| Query API | `process_texts` → model forward → `score_multi_vector` | Replaces the stale `process_queries` path. |
| Vector store | **None for v1** — Modal in-memory MaxSim over the 738-row tensor | Small corpus; avoids Qdrant complexity and already-failed infra. |
| Serving warmth | **Scale-to-zero by default**; warm only during demo windows | No 24/7 idle L4 burn for a demo. |
| Gateway | FastAPI on Hermes/VPS owns SSE, generation, citations, image serving, auth, rate-limit, noindex | Keeps public contract off Modal and centralizes cost/risk controls. |
| Page-image serving | One-time `modal volume get` to VPS; serve static/cacheable files locally | Avoids per-click Modal round-trips. |
| Generation | Gemini 2.5 Flash via official `google-genai` SDK over **downscaled page images** | Visual RAG must send real images, but with sane payload control. |
| Refusal | Two-gate system: **Gate A** retrieval-confidence, **Gate B** grounded structured evidence | Prompt-only refusal is fake safety. |
| Legacy hybrid | Archive as `legacy_hybrid`; fail closed unless explicitly selected | Preserve history without letting it contaminate runtime. |
| Scope boundary | No Qdrant, no hybrid retrieval, no 8B, no corpus expansion in v1 | Prevent scope drift and fake progress. |
| Sacred invariant | `tensor row ↔ page_id ↔ PNG filename ↔ cited page number` must remain 1:1 | Break this and the whole product becomes a liar. |

### 2.1 Retrieval profile contract

```yaml
profile_id: urdpfi_v1__tomoro_colqwen3_4b__202606
corpus_version: legacy_visual_738_rows
runtime: modal_in_memory_maxsim
visual:
  model_id: TomoroAI/tomoro-colqwen3-embed-4b
  model_revision: bf790bd8780b098b86453444632a184bb770be1a
  embedding_dim: 320
  query_api: process_texts + score_multi_vector
  tensor: embeddings_tomoro320_full.pt   # confirm canon in Phase 0
  page_count: 738
  test_pngs_excluded: 5
retrieval_gate:
  min_top1_score: <calibrate>
  min_top1_minus_top2_margin: <calibrate or disable if ties dominate>
  fallback_if_margin_unusable: domain_classifier + absolute score + topk dispersion
generator:
  model_id: gemini-2.5-flash
  input: downscaled_page_image_bytes
  image_budget:
    default_k: 3
    max_k: 5
    long_edge_px: 1024
    format: webp
  structured_output:
    - answer
    - cited_pages
    - visible_evidence_per_page
    - confidence
    - refusal_reason
```

---

## 3. Phase plan with gates

## Phase 0 — Recover and re-verify the asset

**Goal:** turn the current log-only Tomoro path into committed source + fresh receipts.

### Tasks
1. Reconstruct and commit:
   - `scripts/modal_search_service.py`
   - `scripts/provenance_maxsim.py`
   - `scripts/modal_verify_g1.py`
2. Run a **fresh** recursive volume inventory and save the receipt under `launch/artifacts/`.
3. Confirm which tensor is canonical for v1:
   - `embeddings_tomoro320_full.pt`
   - `embeddings.pt`
   - metadata filename(s)
   - shape / sha / row count
4. Reproduce the road-width smoke query from committed code.
5. Assert the sacred mapping:
   - tensor row ↔ metadata row ↔ page_id ↔ PNG filename
6. Add `docs/tomoro_v1_runtime.md` describing the real v1 architecture.
7. Update `README.md` so it stops pretending the April hybrid is current.

### Gate
Phase 0 is complete only when all of these are true:
- retriever source exists **in git**
- fresh volume inventory exists **in repo receipts**
- canonical tensor is named explicitly in this plan and in runtime docs
- one reconstructed search run reproduces sensible results
- page-id mapping assertion passes

### If Phase 0 fails
- If the tensor is missing or corrupted: stop calling the asset healthy; the plan changes to controlled re-embed.
- If the code cannot be reconstructed cleanly from logs: reduce to a 5-page or 20-page canary and prove the API path before touching web.
- If page-id mapping is inconsistent: stop everything until fixed; citations are the product.

---

## Phase 1 — Adversarial eval harness and Gate-A calibration

**Goal:** measure retrieval honestly before generation or UI polish.

### Tasks
1. Build `eval/urdpfi_v1_matrix.jsonl` with at least these buckets:
   - 20 direct answerable queries
   - 20 paraphrased answerable queries
   - 10 table / figure / layout queries
   - 10 near-miss in-domain queries
   - 20 out-of-corpus queries
   - 10 citation-trap queries
   - 10 ambiguity queries
2. Define the labeling method in the eval README:
   - who authored each question
   - how gold pages were chosen
   - how oracle contamination was prevented
3. Log these metrics:
   - recall@1 / @3 / @5
   - refusal precision / recall
   - citation-page correctness
   - top-1 / top-2 / top-5 score distributions
   - p50 / p95 latency
   - cold vs warm latency
4. Calibrate Gate A from real negatives.
5. Decide whether top-score margin is usable. If ties dominate, switch Gate A to absolute score + domain classifier + dispersion features.

### Gate
Phase 1 is complete only when:
- answerable recall@5 ≥ 80%
- out-of-corpus refusal ≥ 90%
- citation-page correctness ≥ 85%
- Gate A has a justified signal, not a decorative threshold

### If Phase 1 fails
- If recall is weak: investigate tensor, query preprocessing, page order, or whether 738-row corpus is too noisy.
- If refusal is weak: improve Gate A before asking Gemini to save us.
- If score ties make margin useless: document it and change the design instead of pretending otherwise.

---

## Phase 2 — Generation and Gate-B grounding

**Goal:** add answer generation without turning retrieval into a hallucination vending machine.

### Tasks
1. Wire Gemini 2.5 Flash via `google-genai` SDK.
2. Send real page images with adaptive budget:
   - default top-3
   - expand to top-5 only when justified
   - downscale to ~1024px long edge
   - use webp / sane quality settings
3. Enforce structured output:
   - `answer`
   - `cited_pages`
   - `visible_evidence_per_page`
   - `confidence`
   - `refusal_reason`
4. Reject or downgrade answers when evidence is absent or unsupported.
5. Re-run the full eval matrix end-to-end.

### Gate
Phase 2 is complete only when:
- Phase 1 thresholds still hold with generation on
- zero fabricated numeric claims appear in the adversarial set
- latency and payload cost are acceptable for demo use

---

## Phase 3 — API surgery

**Goal:** replace the repo's broken runtime path with the real v1 path.

### Tasks
1. Route `/v1/ask` to the Tomoro retriever + Gates A/B.
2. Confirm and fix the prior-audit bugs that still matter to v1:
   - duplicate `event_stream` execution
   - image route stub
   - Gemini stream parsing path
   - any inherited readiness / telemetry bugs on the live path
3. Implement static page-image serving from synced files.
4. Add demo token + rate limiting.
5. Add explicit `warming` SSE event.

### Gate
Phase 3 is complete only when:
- one query causes exactly one retrieval run
- one curl request returns a non-empty grounded answer with citations
- one out-of-corpus query refuses cleanly
- citation image route resolves to the correct page image

---

## Phase 4 — Web demo loop

**Goal:** prove the product loop in the browser.

### Tasks
1. Point Next.js to `/v1/ask`.
2. Wire citation chips to the local page-image route.
3. Implement page-image lightbox.
4. Show `warming` state cleanly.
5. Keep `noindex`, but do not pretend it is protection.

### Gate
Phase 4 is complete only when:
- browser query returns grounded answer + citations
- clicking a citation opens the correct page image
- out-of-corpus query refuses clearly
- browser console is clean

---

## Phase 5 — Public demo (RED gate)

**Goal:** expose the demo only after the loop is trustworthy.

### Tasks
1. Freeze config and runtime docs.
2. Re-run full eval + health checks.
3. Serve behind demo token + rate limits.
4. Only then consider DNS / public exposure.

### Gate
Phase 5 requires explicit approval for:
- Modal GPU spend
- VPS public hosting
- DNS / public exposure

---

## 4. Non-goals / kill list

- Do **not** debug or ship the April hybrid for v1.
- Do **not** introduce Qdrant into v1 runtime.
- Do **not** switch to Tomoro 8B before green.
- Do **not** add OCR/text/sparse/hybrid retrieval in v1.
- Do **not** expand the corpus before the 738-row demo works.
- Do **not** treat hand-picked known queries as evaluation.
- Do **not** trust prompt wording to enforce refusal.
- Do **not** expose static page images publicly without access control.
- Do **not** mark receipts as verified unless they are fresh and saved in the repo.
- Do **not** let the long-term ship ambition hijack v1 execution.

---

## 5. SHIP north-star positioning

There is a broader ambition beyond v1:
- `urban-rag.tanmaychoudhary.com`
- ≥200 documents
- ≥50,000 pages
- p95 ≤ 4s warm
- `IndianPlanningBench v0.1`

That ambition matters, but it is **not** the acceptance criteria for this file.

This plan only exists to get a trustworthy green demo over the current 738-row corpus.
The broader ship roadmap should live in a separate `SHIP_PLAN.md` and should begin **after** this plan is green.

The bridge between v1 and SHIP is simple:
- the adversarial eval matrix from Phase 1 becomes the seed of `IndianPlanningBench`
- every future model / DPI / corpus change must live behind a versioned retrieval profile
- page/citation contract must never break while scaling

---

## Appendix A — Acceptance checklist

Before calling v1 green, all must be true:

- [ ] Tomoro retriever source is committed
- [ ] fresh volume receipts are in `launch/artifacts/`
- [ ] canonical tensor and metadata are explicitly named
- [ ] 738-row corpus definition is explicit and test PNGs are excluded
- [ ] adversarial eval exists and is reproducible
- [ ] Gate A is calibrated from real negatives
- [ ] Gate B blocks unsupported answers
- [ ] `/v1/ask` hits Tomoro, not April hybrid
- [ ] citation chips open the correct page image
- [ ] OOD queries refuse cleanly
- [ ] public exposure remains gated until approval

## Appendix B — Operational principle

The durable rule for Urban RAG is:

**Every embedding, DPI, or corpus change must live behind a versioned retrieval profile and must never break the page/citation contract.**

That is how v1 becomes a product instead of another reset.
