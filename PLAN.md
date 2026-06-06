# Urban RAG v1 Plan — Public Demo First

> **For Hermes / Niney:** SHIP_PLAN.md is the long-range launch vision. This PLAN.md is the current architecture + execution source of truth for getting a narrow public demo working first.

**Goal:** Ship a beautiful, reliable public demo over the existing URDPFI corpus before expanding documents.

**Primary v1 promise:** A user asks an Indian urban planning question, gets a grounded answer with clickable citations, and each citation opens the exact source page image.

**Current v1 corpus:** Existing URDPFI page-image corpus only. No expansion until the end-to-end path is green.

---

## 1. Decisions Locked

### 1.1 Product priority

Public demo first.

We are not optimizing v1 around 50k pages, IndianPlanningBench, full launch marketing, or all-domain regulatory coverage. Those matter later. First, prove the demo loop:

```text
question → retrieval → grounded answer → citation chips → page image lightbox
```

### 1.2 Corpus scope

Use the existing URDPFI corpus.

Observed local state during recon:

- `data/page_images/`: 743 PNG renders.
- `data/embeddings/embeddings.pt`: 572.51 MB legacy embedding artifact; loaded as `torch.Tensor` with shape `[738, 1271, 320]`, dtype `bfloat16`.
- `data/embeddings/metadata.json`: 738 legacy per-page metadata rows: 447 `urdpfi_vol1.pdf`, 250 `urdpfi_vol2.pdf`, 41 `swm_2016.pdf`.
- `urban-rag corpus stats`: 11 documents / 325 pages in the current parsed manifest path.
- `data/docs/`: 8 document-hash directories with `source.pdf`, `pages.jsonl`, `chunks.jsonl`, `sections.jsonl`.

This means the project has two partially divergent corpus states:

1. **Legacy visual corpus** — 738 embedded rows (`[738, 1271, 320]`) and 738 matching page images under `urdpfi_vol1`, `urdpfi_vol2`, `swm_2016`, plus 5 extra hash-prefixed page images.
2. **Current April pipeline corpus** — parsed manifest reports 325 pages / `data/docs` currently has 322 `pages.jsonl` rows.

Important: the legacy embedding dimension is `320`, while current `src/urban_rag/embed/colqwen.py` declares ColQwen2.5 output as `128`. Those artifacts are not interchangeable. Either index/search the legacy artifact with its original model contract, or re-embed all active pages with the current pinned model.

### 1.3 Model / embedding contract

The visual model is part of the index identity.

If the visual model changes, all visual embeddings must be recomputed. The retrieval query encoder and indexed page encoder must be the same compatible model family. Do not mix 8B/4B/ColQwen variants inside one collection.

Every index version must record:

- `corpus_version`
- `embed_model_id`
- `embed_model_revision` if available
- `embedding_dim`
- `index_created_at`
- `page_count`
- `source_manifest_hash`

### 1.4 GPU strategy

Use Modal serverless GPUs if GPU is needed.

Lightning Studio history is useful context, but new GPU work should target Modal unless there is a specific reason not to. Modal should host the visual embedding service and batch re-index job, not the whole web/API stack.

### 1.5 Hybrid codebase strategy

Keep proven modules from current HEAD. Rewrite or simplify confusing/fragile parts.

Keep candidates:

- `src/urban_rag/ingest/render.py` — tested render path.
- `src/urban_rag/ingest/load.py`, `parse.py`, `chunk.py`, `sections.py` — keep where tests pass, but audit against real URDPFI data.
- `src/urban_rag/api/main.py` — API shape is useful, but must be wired to real retrieval.
- `web/` — Next.js build passes; citation UI exists.
- `src/urban_rag/index/qdrant_client.py` — Qdrant schema direction is good.
- `src/urban_rag/retrieve/fusion.py` and typed common models — useful.

Rewrite/repair candidates:

- Eval runner currently uses synthetic perfect candidates. It is not a real smoke eval.
- Corpus state is split between legacy artifacts and current manifest.
- Qdrant on localhost currently has no `pages_visual` / `pages_text` collections.
- Services manifest still references Lightning commands; Modal path needs to become first-class.
- Root `PLAN.md` was ignored by `.gitignore`; this must stop. Architecture docs are product artifacts, not disposable agent scratch.

---

## 2. Target v1 Architecture

```text
          ┌─────────────────────┐
          │ Existing URDPFI PDFs │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Corpus Ledger        │
          │ docs + pages + imgs  │
          └──────────┬──────────┘
                     │ page images
                     ▼
          ┌─────────────────────┐
          │ Modal GPU Embed Job  │
          │ visual model pinned  │
          └──────────┬──────────┘
                     │ page vectors + metadata
                     ▼
          ┌─────────────────────┐
          │ Qdrant pages_visual │
          │ versioned alias     │
          └──────────┬──────────┘
                     │
query ───────────────┘
  │
  ▼
┌─────────────────────┐
│ API /v1/ask          │
│ query embed → search │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Gemini grounded gen  │
│ answer + citations   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Next.js public demo  │
│ citation lightbox    │
└─────────────────────┘
```

### 2.1 Retrieval shape for v1

v1 should be **visual-first**.

Initial retrieval path:

1. Encode query with the same visual model used for page embeddings.
2. Search Qdrant `pages_visual`.
3. Return top-k page candidates with image URI, document title, page number, and text excerpt if available.
4. Generate answer with page images + excerpts.
5. Require inline citations that map back to candidate IDs.

Do not block v1 on text+sparse+VLM rerank if visual-only retrieval is good enough for the URDPFI demo. Hybrid retrieval becomes v1.1 once the narrow demo is already working.

### 2.2 Index versioning

Use physical collection names and stable aliases:

```text
pages_visual__urdpfi_v1__<model_slug>__<yyyymmdd>
pages_visual_current -> active collection
```

Do not write a new model into `pages_visual_current` directly. Build a new collection, smoke it, then switch alias.

### 2.3 Model-independent application boundary

The demo/frontend can be model-independent. The embeddings cannot.

Hard rule: any model change still requires a new derived index, because query vectors and document vectors must live in the same embedding space. The correct abstraction is not "no re-embedding ever"; it is "re-embedding happens behind a versioned retrieval profile and never breaks the product contract."

Create a first-class `RetrievalProfile`:

```yaml
profile_id: urdpfi_v1__tomoro_colqwen3_4b__202606
corpus_version: urdpfi_v1_738_pages
visual:
  model_id: TomoroAI/tomoro-colqwen3-embed-4b
  model_revision: <pinned revision>
  embedding_dim: 320
  vector_schema: multivector_late_interaction
  collection: pages_visual__urdpfi_v1__tomoro_colqwen3_4b__202606
text:
  model_id: lightonai/GTE-ModernColBERT-v1
  embedding_dim: 768
sparse:
  model_id: bm25_v1
reranker:
  model_id: gemini-2.5-flash
active_aliases:
  visual: pages_visual_current
  text: pages_text_current
```

The application only consumes stable page/citation fields:

```json
{
  "page_id": "urdpfi_vol1:p042",
  "doc_id": "urdpfi_vol1",
  "doc_title": "URDPFI Guidelines Vol I",
  "page_number": 42,
  "image_uri": "/v1/corpus/urdpfi_vol1/pages/42/image",
  "excerpt": "...",
  "score": 0.031,
  "provenance": {
    "profile_id": "urdpfi_v1__tomoro_colqwen3_4b__202606",
    "channels": ["visual", "text", "sparse"]
  }
}
```

Model IDs, dimensions, patch counts, and collection names belong in diagnostics and index manifests, not frontend logic.

### 2.4 ViDoRe v3 / pipeline leaderboard stance

Checked on 2026-06-06 from the ViDoRe leaderboard source, MTEB `ViDoRe(v3)` definitions/results, and `illuin-tech/vidore-benchmark` pipeline metrics.

Current standalone ViDoRe v3 leaders are still modern ColQwen-family models:

1. `TomoroAI/tomoro-colqwen3-embed-8b` — mean nDCG@10 around `61.59`.
2. `athrael-soju/colqwen3.5-4.5B-v3` — around `61.46`.
3. `OpenSearch-AI/Ops-Colqwen3-4B` — around `61.17`.
4. `TomoroAI/tomoro-colqwen3-embed-4b` — around `60.20`.

Pipeline leaderboard is not a simple model leaderboard. Top score is an agentic `ColEmbed-VL-8B + Opus 4.5` style pipeline around `69.22`, but with self-reported query latency around `135s/query`, which is wrong for the public demo. More practical entries include Jina text + reranker and NVIDIA ColEmbed/Nemotron variants in the `3–9s/query` range.

Decision: do not blindly chase the top leaderboard row. For v1, pick one visual retriever profile, build a real 738-page index, and evaluate on planning-specific smoke queries. Leaderboard rank is a candidate generator; our own corpus eval is the decision maker. There, fixed the shiny leaderboard rabbit hole before it ate the week.

### 2.5 Page image serving

v1 can serve page images from the API filesystem route or static object storage. The user-facing contract is more important than storage choice:

```json
{
  "page_id": "urdpfi-vol1:p042",
  "doc_title": "URDPFI Guidelines Vol I",
  "page_number": 42,
  "image_url": "/v1/pages/urdpfi-vol1/p042.png",
  "excerpt": "..."
}
```

Citation chips in the web UI must resolve to this image URL without guessing local paths.

---

## 3. Current Verification Baseline

Run on 2026-06-06 in `/root/projects/urban-planning-rag`.

### Passing

- Imports pass:
  - `urban_rag.common.settings`
  - `urban_rag.api.main`
  - `urban_rag.retrieve.orchestrator`
  - `urban_rag.embed.serve`
- Typecheck passes:
  - `uv run pyright src/urban_rag/` → `0 errors, 0 warnings, 0 informations`
- Focused backend test slice passes:
  - `uv run pytest tests/unit/urban_rag/common tests/unit/urban_rag/ingest tests/unit/urban_rag/retrieve/test_fusion.py tests/unit/urban_rag/eval/metrics/test_retrieval.py -q --maxfail=10`
  - Result: `290 passed, 10 warnings in 71.26s`
- Web production build passes after installing deps:
  - `cd web && npm ci && npm run build`
  - Result: Next.js build successful; one `<img>` performance warning.
- Synthetic smoke eval passes:
  - `uv run python -m urban_rag.eval run --dataset smoke --tag live-recon-20260606-140131`
  - Result: `25 passed / 0 failed`

### Not passing / not trustworthy yet

- Full unit run aborted because the disk hit 100% during pytest cache write. This is an environment issue first, not yet a proven test failure.
- `ruff check src/urban_rag tests/unit` fails mostly on test hygiene: unused imports, import ordering, long lines, blind `pytest.raises(Exception)`, and Bandit-style temp path warnings in tests.
- Web install reports `6 vulnerabilities (2 moderate, 4 high)` from npm audit.
- Synthetic eval is not evidence of retrieval quality. It creates perfect fake candidates from expected pages.
- Local Qdrant container is running for other infrastructure, but collections are only memory-related; no `pages_visual` or `pages_text` collections are present.

---

## 4. Phase Plan

## Phase 0 — Stabilize Ground Truth

**Goal:** Make the repo tell one truth about corpus, model, and index state.

### Task 0.1 — Stop ignoring PLAN.md

- Modify `.gitignore` to stop ignoring `PLAN.md`.
- Commit this file as the architecture source of truth.

Verification:

```bash
git check-ignore PLAN.md || true
# Expected: no output
```

### Task 0.2 — Create corpus ledger

Create `data/corpus_ledger.json` or `docs/corpus_ledger.md` that reconciles:

- `data/page_images/*.png`
- `data/docs/*/pages.jsonl`
- `data/manifest.parquet`
- `data/embeddings/metadata.json`
- source PDFs

Output table:

| doc_id | title | source_pdf | pages_jsonl | page_images | embedding_rows | status |
|---|---|---|---:|---:|---:|---|

Gate:

- We can explain exactly why page images are 743 while `corpus stats` says 325.
- Every page image either maps to a ledger page or is marked legacy/orphan.

### Task 0.3 — Decide active corpus version

Create a version label:

```text
urdpfi-v1-existing
```

Gate:

- Active page count is explicit.
- Active source PDFs are explicit.
- Orphan/legacy files are not deleted yet, just labelled.

---

## Phase 1 — Real Visual Index

**Goal:** Build a queryable visual index for the existing URDPFI corpus.

### Task 1.1 — Modal embed service skeleton

Create a Modal app that can:

- Load pinned visual model.
- Embed a batch of page image paths or uploaded image bytes.
- Embed a query with the same model.
- Return model metadata with every response.

Do not support multiple models dynamically in v1. One pinned model per deployed service.

### Task 1.2 — Re-index active corpus

Batch embed all active page images and write to a new Qdrant collection:

```text
pages_visual__urdpfi_v1_existing__<model_slug>__<date>
```

Gate:

- Qdrant count equals active ledger page count.
- Random 10 payloads have valid `page_id`, `doc_title`, `page_number`, `image_url`.
- Model metadata saved beside index manifest.

### Task 1.3 — Query smoke

Write a real retrieval smoke script:

```bash
uv run python scripts/smoke_visual_retrieval.py
```

It should run 5 URDPFI-only queries and print top-5 pages with page IDs and image paths.

Gate:

- At least 4/5 queries retrieve plausible pages by human inspection.
- No Gemini generation involved yet.

---

## Phase 2 — API End-to-End

**Goal:** Local API returns real answer + citations over real retrieval.

### Task 2.1 — Wire `/v1/ask` to real visual retrieval

For v1, the API may bypass text/sparse/rerank if those are not indexed.

Add a config flag:

```text
RETRIEVAL_MODE=visual_only|hybrid
```

Default v1: `visual_only`.

Gate:

```bash
curl -N http://localhost:3100/v1/ask/<id>/stream
```

Streams retrieval started → candidates → generation tokens → completed.

### Task 2.2 — Citation contract

Every generated citation must map to a returned candidate.

Gate:

- No citation chip can point to a missing page.
- If Gemini cites outside retrieved candidates, the API rejects/repairs the citation before returning.

### Task 2.3 — Page image route

Add `/v1/pages/{page_id}` or equivalent route.

Gate:

```bash
curl -I http://localhost:3100/v1/pages/<known_page_id>
# Expected: 200 image/png
```

---

## Phase 3 — Web Demo

**Goal:** Public-demo UI feels real, not like a backend test harness wearing lipstick.

### Task 3.1 — Landing query flow

The root page should support:

- Example URDPFI queries.
- Streaming answer.
- Citation chips.
- Clear error states.

Gate:

- `npm run build` passes.
- Manual browser query works locally.

### Task 3.2 — Citation lightbox polish

Keep existing `CitationChip.tsx` and `CitationLightbox.tsx` if they work. Polish only what breaks the demo.

Gate:

- Click citation → page image opens.
- Mobile viewport works.
- Back/close keyboard path works.

### Task 3.3 — Demo deployment

Deploy API + web behind a private/noindex URL first.

Public URL target remains:

```text
urban-rag.tanmaychoudhary.com
```

Public exposure/DNS remains a RED-line action: recommend and wait for explicit approval before changing DNS/firewall.

---

## Phase 4 — Real Smoke Eval

**Goal:** Replace fake green checks with checks that can catch broken retrieval.

### Task 4.1 — URDPFI-only smoke set

Create `eval/urdpfi_smoke.jsonl` with 10-15 questions that are actually answerable from the active corpus.

Each entry needs:

- question
- expected page(s) or doc section(s)
- required answer keywords
- rubric

### Task 4.2 — Live retrieval eval

Modify eval runner or add a new runner so it calls real retrieval instead of `_synthetic_candidates_for_entry`.

Gate:

```bash
uv run python -m urban_rag.eval run-live --dataset urdpfi_smoke
```

Minimum v1 gate:

- recall@5 ≥ 0.70 while tuning.
- recall@5 ≥ 0.85 before public demo.

---

## 5. Kill Criteria / Anti-Procrastination Rules

- Do not ingest new documents until a real URDPFI query returns citations + page images locally.
- Do not switch visual models without creating a new index version.
- Do not build a broader benchmark before live retrieval works.
- Do not optimize text/sparse/hybrid retrieval until visual-only has a measured baseline.
- Do not deploy publicly until page-image citations are reliable.

---

## 6. Immediate Next Work Order

1. Patch `.gitignore` so `PLAN.md` can be tracked.
2. Generate corpus ledger and explain the 743 vs 325 mismatch.
3. Create `scripts/smoke_visual_retrieval.py` skeleton against Qdrant/Modal interface.
4. Add Modal app skeleton for pinned visual model.
5. Rebuild visual index for the active corpus.
6. Wire local API to visual-only retrieval.
7. Run one browser query end-to-end.

That is the spine. Everything else is decoration until this works.
