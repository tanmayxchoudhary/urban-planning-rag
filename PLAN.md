# Urban Planning RAG — Comprehensive Revival Plan

**Status**: Draft v1.0 · **Author**: Niney (with Tanmay) · **Date**: 2026-04-26
**Target**: production-grade visual RAG over hundreds of Indian urban-planning documents, queryable through a simple web interface, deployed on rented GPUs (Lightning AI primary), executed by 24/7 autonomous agents.

This document is the single source of truth for the rebuild. It supersedes `README.md`, `HYBRID_RAG_V4.md`, and any `vN.x` claims in code comments. Everything in this plan is meant to be acted on by autonomous agents — every task has acceptance criteria and verification steps so an agent can check its own work.

---

## Table of Contents

- [Part 0 — Executive Summary](#part-0--executive-summary)
- [Part I — Audit Findings & Reset](#part-i--audit-findings--reset)
- [Part II — Vision, Goals, Non-Goals](#part-ii--vision-goals-non-goals)
- [Part III — SOTA Survey: Visual RAG, April 2026](#part-iii--sota-survey-visual-rag-april-2026)
- [Part IV — Architectural Decisions](#part-iv--architectural-decisions)
- [Part V — Document Ingestion Pipeline](#part-v--document-ingestion-pipeline)
- [Part VI — Embedding & Indexing Stack](#part-vi--embedding--indexing-stack)
- [Part VII — Retrieval Pipeline](#part-vii--retrieval-pipeline)
- [Part VIII — Generation & Answer Synthesis](#part-viii--generation--answer-synthesis)
- [Part IX — Web Interface](#part-ix--web-interface)
- [Part X — Evaluation Framework](#part-x--evaluation-framework)
- [Part XI — Deployment on Lightning AI](#part-xi--deployment-on-lightning-ai)
- [Part XII — Engineering Standards](#part-xii--engineering-standards)
- [Part XIII — Observability & Telemetry](#part-xiii--observability--telemetry)
- [Part XIV — Agent Execution Roadmap](#part-xiv--agent-execution-roadmap)
- [Appendix A — Dead Code Removal Manifest](#appendix-a--dead-code-removal-manifest)
- [Appendix B — Document & Data Schemas](#appendix-b--document--data-schemas)
- [Appendix C — API Spec](#appendix-c--api-spec)
- [Appendix D — Cost & Capacity Model](#appendix-d--cost--capacity-model)
- [Appendix E — Risk Register](#appendix-e--risk-register)
- [Appendix F — Glossary & References](#appendix-f--glossary--references)

---

## Part 0 — Executive Summary

### 0.1 The product

A queryable visual RAG system for Indian urban planning regulations. Users ask questions in natural language ("What is the FSI for residential zones in Mumbai?"). The system retrieves the most relevant pages from a corpus of hundreds of planning documents (NBC, URDPFI, state master plans, GDCRs, DCRs, SWM, BIS standards, IRC codes, MoHUA guidance), grounds an answer in those pages with precise citations, and renders both the answer and the source page images in a clean web interface.

### 0.2 The shape of the rebuild

The current repo is a Frankenstein of five competing pipelines from January–February 2026, only one of which (the `cli.py` / `src/rag.py` v3 path) ever worked end-to-end, and even that one only on 738 pages (URDPFI v1+v2 + SWM). The README falsely claims NBC is indexed; it isn't. The `src/hybrid_rag/` v4 scaffold was never wired up.

We are not patching this. We **fork forward**: keep what's salvageable (the embeddings, page images, the embedding logic), and rebuild everything else against a clean architecture grounded in April-2026 SOTA practices.

### 0.3 The architecture (one paragraph)

Documents flow through a **layout-aware parser** (Docling primary, Marker fallback) that produces both rendered page images and structured markdown with hierarchical sections. The page images are embedded with a **late-interaction multi-vector model** (ColQwen2.5 → ColNomic-7B as upgrade target) into a **Qdrant** collection that supports server-side multi-vector MaxSim. The structured text is embedded with a **dense text encoder** (BGE-M3 or GTE-ModernColBERT) and indexed alongside as a complementary view. **Retrieval** is two-stage: server-side ANN candidate generation (top-200) → MaxSim rerank (top-20) → optional **VLM cross-encoder rerank** (top-5) using Gemini 2.5 Flash or Qwen2.5-VL. **Generation** sends the top pages as images to a **VLM** (Gemini 2.5 Pro for accuracy, Gemini 2.5 Flash for speed) with a structured citation prompt. The whole thing sits behind a **FastAPI** server deployed on Lightning AI Studios via **LitServe**, fronted by a minimal **Next.js** chat UI. **Evaluation** runs continuously through RAGAS (metrics) + DeepEval (CI gates) + TruLens (production monitoring) on a curated golden set of urban-planning Q&A pairs.

### 0.4 What "done" looks like

- 500+ Indian urban-planning documents indexed end-to-end (target after one ingestion sweep)
- Median retrieval `nDCG@10 ≥ 0.65` on the golden set (baseline ColQwen2.5 territory; SOTA is 0.63 on ViDoRe V3 across all domains, our domain-specific should beat that)
- Median answer-faithfulness (RAGAS) `≥ 0.85` on the golden set
- p95 query latency `≤ 4 seconds` on a warm L4/L40S endpoint
- A web URL Tanmay can share where anyone types a question and gets an answer with cited page images
- Zero dead code paths in `main` branch
- A green CI pipeline that runs lint, type-check, unit tests, retrieval eval, and answer eval on every PR

### 0.5 What we're explicitly NOT doing in v1

- Hindi / regional-language input (English-only at launch)
- User accounts, history, or personalization
- Real-time document upload from the web UI (ingestion is a backend pipeline only)
- Multi-tenancy
- The PageIndex / NetworkX-KG fusion stack from v4 — we keep the *pattern* (multi-source fusion) but drop those specific components, since neither was ever validated and both add operational complexity

---

## Part I — Audit Findings & Reset

### 1.1 What exists on disk

```
urban-planning-rag/
├── .env                     # secrets: GEMINI_API_KEY, LIGHTNING_API_KEY (gitignored ✓)
├── .gitignore               # sensible
├── README.md                # outdated, claims 3,047 pages, actually 738
├── HYBRID_RAG_V4.md         # speculative v4 design, never validated
├── cli.py                   # v3 CLI, calls src/rag.py
├── query_simple.py          # CPU sentence-transformers + Gemini, mean-pools embeddings (lossy)
├── query_vision.py          # random-page → Gemini, no retrieval at all
├── requirements.txt         # ColQwen + Chroma + Gemini, Python 3.14 era
├── requirements-hybrid.txt  # adds qdrant + networkx + pageindex
├── requirements-lightning.txt
├── data/
│   ├── embeddings/embeddings.pt   # 573 MB, 738 pages × variable patches × 320-dim
│   ├── embeddings/metadata.json   # 738 entries
│   ├── page_images/                # 738 PNGs, adaptive 100/250 DPI
│   └── lance_db/                   # 980 KB stub, never finished
├── src/
│   ├── __init__.py
│   ├── rag.py                # v3 RAG class, ColQwen + ChromaDB + MaxSim, GPU-required at query time
│   ├── indexer_optimized.py  # 5–10× faster ChromaDB indexer
│   ├── api.py                # FastAPI + Qdrant, query vector hardcoded to embs[0] ← broken
│   ├── query.py              # placeholder Qdrant client, query also broken
│   ├── query_encoder.py      # MiniLM-L6 sentence-transformer, dimension mismatch with ColQwen
│   └── hybrid_rag/           # v4 scaffold, none of it wired up
│       ├── __init__.py       # HybridVisualRAG facade
│       ├── pageindex_integration.py   # external API, requires PAGEINDEX_API_KEY
│       ├── qdrant_store.py
│       ├── knowledge_graph.py         # NetworkX MultiDiGraph
│       ├── fusion_reranker.py
│       ├── pdf_extractor.py
│       ├── entity_extractor.py
│       └── api.py
└── scripts/
    ├── README.md
    ├── embed.py              # v3.0.0, the embedding workhorse, sound code
    ├── pipeline.py           # orchestrator, mostly works
    ├── build_index.py        # legacy
    ├── check_docs.py
    ├── test_gemini.py
    ├── test_lightning_setup.py
    ├── test_validation.py
    ├── index_lancedb.py      # competing index path
    ├── index_qdrant.py       # competing index path
    ├── migrate_to_v4.py      # never run
    ├── deploy_demo.sh
    ├── send_emails.py        # ?? portfolio outreach? unrelated
    ├── track_leads.py        # ??
    └── archive/              # 18 abandoned embed_vNN.py + benchmark scripts
```

### 1.2 The five competing pipelines

| Path | Entry | Embed model | Retrieval | Generation | Status |
|---|---|---|---|---|---|
| **A. v3 ColQwen + Chroma** | `cli.py` → `src/rag.py` | ColQwen3-4B | Multi-query expansion + MaxSim rerank | Gemini Vision | Untested in current env (no torch installed); GPU-required at query time |
| **B. CPU MiniLM** | `query_simple.py` | sentence-transformers MiniLM-L6 (384-dim) over **mean-pooled** ColQwen embeddings | Cosine over mean-pool | Gemini 2.0 Flash | Runs but retrieval is degraded — mean-pooling kills late interaction |
| **C. Vision-only** | `query_vision.py` | none | random sample | Gemini 2.0 Flash | Toy / not retrieval |
| **D. Qdrant API** | `src/api.py`, `src/query.py` | none at query time | Qdrant ANN with **`embs[0]` as query** ← bug | none | Broken |
| **E. v4 hybrid** | `src/hybrid_rag/` | ColQwen + PageIndex API + KG | Fusion of three | Placeholder | Scaffolded only, never run end-to-end |

### 1.3 Honest assessment

The codebase has good ideas scattered across it (adaptive DPI, two-stage retrieval, optimized indexer, fusion concept) but no single path that runs the full loop. The `src/hybrid_rag/` module is a 1,500-line spec disguised as code — every class has a `print("⚠️ X not available")` fallback and a TODO. It needs to be deleted and rebuilt with discipline.

The 573 MB of existing embeddings on URDPFI + SWM is **salvageable**, but only if we keep using ColQwen3-embed-4B at query time. Since we're upgrading the embedding model anyway (ColNomic-7B is the front runner), we will **re-embed everything**. The 738 page PNGs at adaptive DPI are also salvageable as ground truth for evaluation.

### 1.4 Reset decisions

| Decision | Rationale |
|---|---|
| Keep: `scripts/embed.py` (rewrite the model loader, keep the structure) | Adaptive DPI + PyMuPDF backbone is sound |
| Keep: `scripts/pipeline.py` (rewrite as `cli.py`-equivalent) | Orchestration shape is right |
| Keep: `data/page_images/` for golden-set evaluation | 738 known-good pages |
| Delete: `query_simple.py`, `query_vision.py`, `src/api.py`, `src/query.py`, `src/query_encoder.py`, `src/hybrid_rag/`, `src/indexer_optimized.py`, all `scripts/archive/`, `scripts/send_emails.py`, `scripts/track_leads.py`, `scripts/index_lancedb.py`, `scripts/migrate_to_v4.py`, `scripts/build_index.py`, `requirements-hybrid.txt`, `requirements-lightning.txt`, `HYBRID_RAG_V4.md` | Dead code |
| Replace: `README.md` | Outdated and misleading |
| Initialize: git history starting from this rebuild | The repo has no commits — the existing files become the "before" snapshot in commit 1, then we rebuild |
| Rotate: secrets in `.env` (Gemini + Lightning keys are committed in git tree even though gitignored — the `.env` is on disk and was previously checked into the working tree at one point) | Treat as compromised, rotate both before any deploy |

### 1.5 What we're keeping from v3

The two ideas worth porting forward into the new system:
1. **Adaptive DPI**: classify pages by `has_images || num_drawings > 40` → text@100 DPI, visual@250 DPI. Saves storage and cuts patch count for text-heavy pages without sacrificing diagram fidelity.
2. **Two-stage retrieval**: ANN candidate retrieval → MaxSim rerank. This is canonical late-interaction practice and aligns with SOTA.

What we're dropping from v3:
- ChromaDB (replacing with Qdrant for native multi-vector + filtering at scale)
- The MiniLM query encoder (dimension mismatch was the bug; we use the same ColQwen at query time, period)
- Multi-query token expansion as currently implemented (we replace with proper MaxSim everywhere, since Qdrant supports it server-side now)

---

## Part II — Vision, Goals, Non-Goals

### 2.1 The user

Two primary users:
1. **Tanmay** (you), planning your thesis and exploring regulatory landscapes; you ask spec-grade questions ("what's the differential FSI between TOD zones in Delhi vs. Bangalore?") and want grounded, citable answers.
2. **A planning student / junior architect** elsewhere in India who lands on the public URL because they can't navigate 2,000-page PDFs and needs an answer in 30 seconds.

Secondary: any researcher, journalist, urban-policy blogger, or activist who needs to cite Indian planning regulations.

### 2.2 Goals (in priority order)

1. **Correctness with evidence**. Every answer comes with page-image citations the user can click and read. Hallucinations are caught at evaluation, not in production.
2. **Coverage**. The corpus must be deep — at least all the central docs (NBC vols 1–2, URDPFI vols 1–2, SWM rules, BIS structural codes, IRC road codes, MoHUA guidance), plus state master plans for the 8 largest metros (Delhi, Mumbai, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad).
3. **Speed**. Sub-4-second p95 query → answer.
4. **Cost discipline**. Embedding is one-time. Retrieval is server-cheap (Qdrant). VLM generation is the recurring cost; budget assumes Gemini 2.5 Flash by default with Pro escalation only when explicit.
5. **Maintainability**. The system must be operable by a single autonomous agent on a 24/7 cron without human babysitting. That means: clean separation of stages, idempotent pipelines, structured logs, alerting, automatic rollback on regression.
6. **Defensibility**. The corpus and evaluation set are the moat. Both must be versioned, reproducible, and documented.

### 2.3 Non-goals (explicit)

- Multi-language input/output (English only at launch; Hindi via translation layer is a v2 task)
- User accounts, conversation history, fine-grained access control
- A general legal-research tool (this is **planning regulations**, not all of Indian law)
- Real-time document upload from end users (corpus is curated; ingestion is offline)
- An agent that drafts compliance reports / generates planning documents (we retrieve and answer, we don't write planning docs)
- Mobile native app
- Real-time collaboration

### 2.4 Quality bar (concrete)

| Metric | Target | How measured |
|---|---|---|
| Document coverage | ≥ 500 docs, ≥ 75,000 pages | manifest count |
| Retrieval `nDCG@10` (golden set) | ≥ 0.65 | RAGAS / custom eval harness |
| Retrieval `Recall@5` | ≥ 0.80 | golden set |
| Answer faithfulness (RAGAS) | ≥ 0.85 median | RAGAS faithfulness metric on golden set |
| Answer relevance (RAGAS) | ≥ 0.80 median | RAGAS answer relevance |
| Citation accuracy | ≥ 0.95 | manually-graded subset; cited page must contain the answer |
| p50 query latency | ≤ 1.8 s | server-side metric |
| p95 query latency | ≤ 4.0 s | server-side metric |
| Cold-start retrieval (warm-pool) | ≤ 8 s | Lightning AI metric |
| CI build time | ≤ 6 min | GitHub Actions |
| Test coverage | ≥ 70% on `src/core/` | pytest-cov |

---

## Part III — SOTA Survey: Visual RAG, April 2026

This part summarizes the state of the art so the architecture decisions in Part IV are grounded. All claims are dated. Where a metric is given, the source is cited in Appendix F.

### 3.1 Visual document retrieval — the paradigm shift

Visual document retrieval (VDR) embeds **rendered page images** rather than parsed text. The breakthrough paper is ColPali (July 2024), which feeds page screenshots into a vision-language model and produces ColBERT-style multi-vector embeddings — one vector per spatial patch. Retrieval uses **MaxSim late interaction**: each query-token vector is compared against every document patch vector, the maximum is taken per query token, and those maxima are summed. This preserves spatial layout information that OCR destroys.

As of April 2026:
- **Nemotron ColEmbed V2 8B** leads ViDoRe V3 leaderboard at NDCG@10 = 63.42 (Feb 2026)
- **ColQwen2.5** (built on Qwen2.5-VL-3B) is the de facto open-source default, balancing quality and inference cost
- **ColNomic-7B** offers 7B-class quality at competitive latency
- **Jina-v4** is the strong general-purpose alternative
- **AGREE / AGREEQwen2.5** is a fine-tuning recipe that yields +7% absolute nDCG@1 over ColQwen2.5 by adding attention-grounded supervision
- **Prune-then-Merge** (Feb 2026) reduces multi-vector storage by ~3–5× with minimal recall loss, by clustering redundant patches before indexing

### 3.2 Why visual retrieval beats text retrieval for planning docs

Indian planning regulations are ~70% tables (zoning matrices, FSI tables, parking ratios), ~20% diagrams (zoning maps, building envelope sketches, road cross-sections), and ~10% prose. OCR loses the table structure and the spatial relationships in the diagrams. Text retrieval over OCR'd content systematically misses queries about spatial regulations. Visual embeddings see the page as the planner sees it.

### 3.3 Multi-vector vs. single-vector storage

Multi-vector storage: ~50–100 vectors per page (one per patch). 320 dim × 100 patches = 32k floats per page = 128 KB at fp32 or 64 KB at fp16. For 100k pages: 6.4–12.8 GB. Manageable.

Single-vector dense models (CLIP, SigLIP, JINA-CLIP-v2) are 3–8% behind multi-vector on ViDoRe but storage is ~50× cheaper. We adopt **multi-vector primary + single-vector secondary** for the candidate-generation stage.

### 3.4 Vector databases for multi-vector retrieval

| DB | Multi-vector native | Server-side MaxSim | Hybrid (BM25+dense) | Filtering | Multimodal table | Verdict |
|---|---|---|---|---|---|---|
| Qdrant 1.13+ | ✅ | ✅ (server-side) | ✅ | ✅ rich | ❌ (use external) | **Primary** |
| LanceDB | ✅ | partial | ✅ | ✅ SQL | ✅ | Strong alt; great for storing image bytes + metadata + vectors in one table |
| Vespa | ✅ native | ✅ | ✅ | ✅ | ❌ | Best at scale, complex ops |
| Weaviate | ✅ | ✅ | ✅ | ✅ | partial | Good, less popular for pure retrieval |
| pgvector | partial (via expression) | ❌ | external | ✅ SQL | partial | Avoid for multi-vector at scale |
| ChromaDB | flat patch-level workaround | manual rerank | ❌ | partial | ❌ | What v3 used; we move off |
| Milvus | ✅ | ✅ | ✅ | ✅ | ❌ | Strong; we pick Qdrant for ergonomics |

**Decision: Qdrant primary, with LanceDB as the corpus-storage layer (page bytes + metadata) and the eval data store.** Multi-stage retrieval over Qdrant uses pooled single-vector ANN for candidate generation and full multi-vector MaxSim for reranking, both server-side in one call.

### 3.5 PDF parsing for the structured-text view

We need both rendered pages (for the visual encoder) and structured text (for hybrid retrieval and citation grounding). State of the art for PDF→Markdown as of April 2026:

| Tool | Strength | Weakness | Use case |
|---|---|---|---|
| **Docling** (IBM) | Preserves semantic hierarchy (sections, headings, tables); produces `DoclingDocument` JSON; built for RAG | Slower than Marker | **Primary** — hierarchy is gold for routing queries |
| **Marker** (with `--use_llm`) | Safest default; handles messy layouts; LLM cleanup option | Hierarchy info is shallower | **Fallback** when Docling fails |
| **MinerU** | Best for complex tables; CJK-aware (irrelevant for us); good academic-paper handling | Image handling weaker | Last resort |
| **Mistral OCR** | Cloud API; clean output | Cost + latency + dependency on third party | Avoid for batch ingestion |
| **olmOCR** | Open, decent | Behind Marker on most benchmarks | Skip |
| **PyMuPDF4LLM** | Fast pure-Python | Loses layout | Skip for ingestion |

**Decision: Docling primary; Marker fallback when Docling errors or returns empty hierarchy.** Both produce text + page-region metadata we can cross-reference against the visual patch index.

### 3.6 Late-interaction text models (for the parallel text path)

For the structured-text retrieval path (parallel to the visual path), the SOTA late-interaction text models in 2026:

- **GTE-ModernColBERT** (LightOn, 2026): ModernBERT backbone + PyLate training; SOTA on BEIR for late-interaction
- **ColBERT v2 / PyLate**: still strong, smaller
- **Jina-ColBERT v2**: multilingual, longer context

**Decision: GTE-ModernColBERT for text late interaction.** Indexed alongside the visual ColQwen patches in Qdrant, separate collection.

### 3.7 Reranking — VLM-as-cross-encoder

After candidate generation + MaxSim, the top 10–20 candidates can be reranked by sending the actual page images to a VLM with the query and asking for relevance scores. This is expensive but very accurate.

- **Gemini 2.5 Flash** as default reranker: 1M context, cheap, fast, sees images natively
- **Gemini 2.5 Pro** for evaluation-only reranking ground truth
- **Qwen2.5-VL-72B** as open-source alternative for cost optimization on Lightning GPUs
- **InternVL3** as an ablation candidate

The reranker prompt asks: "Score each of these N pages 1-5 for how well it answers `<query>`. Return JSON: `[{page_id, score, justification}]`."

### 3.8 VLM for answer generation

| Model | Context | Speed | Quality | Cost | Use case |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | 1M | medium | top-tier | $$$ | high-stakes / eval ground truth |
| **Gemini 2.5 Flash** | 1M | fast | very good | $ | **production default** |
| Gemini 3.0 Flash (preview) | 1M | fast | top-tier | $$ | watch — may replace 2.5 Flash |
| Qwen2.5-VL-72B | 32k | medium (on H100) | very good | self-host on Lightning | open-source fallback for cost shock |
| GPT-5 | 256k | medium | top-tier | $$$$ | not in v1 |
| Claude Opus 4.6 | 1M | fast (with extended output) | top-tier | $$$ | not in v1 (no image reasoning advantage over Gemini for cost) |

**Decision: Gemini 2.5 Flash production; Gemini 2.5 Pro eval; Qwen2.5-VL-72B as self-host fallback with a feature flag.**

### 3.9 RAG evaluation in 2026

The recommended production stack for RAG eval:
- **RAGAS** for retrieval and generation metrics: context_precision, context_recall, faithfulness, answer_relevance, answer_correctness
- **DeepEval** for CI/CD quality gates: pytest-style assertions on RAG outputs, fails the build if metrics drop
- **TruLens or Langfuse** for production monitoring: trace every query, log scores, alert on drift
- **ARES** for synthetic Q&A generation when a golden set isn't available — generate, judge, train custom judges

We use all four.

### 3.10 Hierarchical and agentic RAG patterns

For documents that span 1,000+ pages (NBC, master plans), simple flat retrieval struggles. Two patterns to adopt:

1. **Hierarchical chunking** — index at three levels: document → section → page. Top-level summaries route the query to the right section, then patch-level retrieval pinpoints pages.
2. **Agentic refinement** — when initial retrieval confidence is low, an agent reformulates the query (HyDE: write a hypothetical answer then embed *that*), or iteratively explores adjacent pages.

Both are v1.5 features. v1 ships with two-stage flat retrieval + VLM rerank.

### 3.11 Hybrid retrieval (sparse + dense)

BM25 + dense fusion via Reciprocal Rank Fusion (RRF) catches rare-term queries (specific section numbers, named regulations like "Annexure III"). Qdrant supports sparse vectors natively. We add BM25 as a third signal at MaxSim-rerank time using RRF.

### 3.12 Closing the loop: HyDE / CRAG / Self-RAG

- **HyDE** (Hypothetical Document Embeddings): use the LLM to write a fake answer, embed that, retrieve — works because the fake answer is in answer-space, not question-space.
- **CRAG** (Corrective RAG): a small classifier judges retrieval quality; if low, falls back to web search or query rewriting.
- **Self-RAG**: the LLM emits special tokens to decide if it should retrieve again.

We adopt **HyDE as a query-rewriting option** in the agentic refinement loop (v1.5), not v1. CRAG-style fallback **is** in v1: if retrieval confidence (top-1 MaxSim score) is below a threshold, the system surfaces "low-confidence" warning to the user and shows top-10 candidates instead of generating a confident answer.

---

## Part IV — Architectural Decisions

This section is a series of one-page Architecture Decision Records (ADRs). Each captures a binding choice.

### ADR-001 — Visual RAG primary, text-RAG secondary

- **Context**: Indian planning docs are layout-heavy (tables, diagrams).
- **Decision**: Embed page images with ColQwen2.5 (upgrade target: ColNomic-7B) as the primary retrieval signal. Run a parallel text retrieval pipeline using Docling-extracted markdown embedded with GTE-ModernColBERT.
- **Consequences**: Storage is ~10× larger than text-only RAG. We accept this for the retrieval-quality gain. At 100k pages × 64 KB/page (fp16 multi-vector) = 6.4 GB. Trivially fits in a single Qdrant node.
- **Alternatives considered**: Pure text-RAG (rejected: kills tables and diagrams). Pure visual (rejected: misses BM25 keyword matches like specific clause numbers).

### ADR-002 — Embedding model: ColQwen2.5 v1, ColNomic-7B v1.5

- **Context**: ColQwen3-embed-4B (v3 of this repo) is fine but ColQwen2.5 is the mainstream default with the best ecosystem support (illuin-tech/colpali); ColNomic-7B is the upgrade path.
- **Decision**: ColQwen2.5 for v1 (320-dim multi-vector, ~50–80 patches/page typical). Plan for an A/B against ColNomic-7B at v1.5.
- **Consequences**: 320-dim aligns with existing 573 MB embeddings *if* we keep ColQwen3-4B; switching to ColQwen2.5 means re-embedding. We re-embed everything anyway because the corpus expands beyond the original 738 pages.
- **Open question**: should we evaluate Nemotron ColEmbed V2 8B (current ViDoRe leader)? Yes, but as a v1.5 ablation; not v1 because ecosystem tooling is thinner.

### ADR-003 — Vector DB: Qdrant primary, LanceDB for corpus storage

- **Context**: Need server-side MaxSim, rich filtering, sparse+dense hybrid.
- **Decision**: Qdrant 1.13+ as the multi-vector index. LanceDB as the document/page metadata store and rendered-image cache, plus the eval-data store.
- **Consequences**: Two systems to operate, but each plays a clean role. Qdrant scales to billions of vectors; LanceDB gives us versioned datasets for evals.
- **Operational form**: Qdrant runs as a containerized service on the Lightning Studio (or cloud Qdrant). LanceDB is a directory-on-disk we sync between Studios via the project's persistent volume.

### ADR-004 — Reranker: VLM cross-encoder (Gemini 2.5 Flash)

- **Context**: Top-20 from MaxSim still has noise. A VLM reranker that *sees the page* delivers quality.
- **Decision**: After two-stage retrieval, send top-20 page images + query to Gemini 2.5 Flash with a structured-output prompt asking for relevance scores. Take top-5.
- **Consequences**: Adds ~1–2 s latency and ~$0.0001–0.0005 per query. Worth it.
- **Knob**: feature flag `RERANK_VLM=on|off`; when off, take top-5 directly from MaxSim. Default `on`.

### ADR-005 — Generator: Gemini 2.5 Flash default, Pro for high-stakes

- **Context**: Need a VLM that takes images + text, outputs grounded answers.
- **Decision**: Gemini 2.5 Flash as production default. Gemini 2.5 Pro as an opt-in for "deep mode" queries flagged by the user (UI toggle).
- **Consequences**: Free-tier Flash gives us 1500 RPD. Beyond that, paid tier — budget controlled.
- **Open-source escape hatch**: Qwen2.5-VL-72B self-hosted on a Lightning H100; activates via env var `GEN_BACKEND=qwen-vl`. Tested in CI but not active by default.

### ADR-006 — Document parser: Docling primary, Marker fallback

- **Context**: We need structured markdown alongside rendered images.
- **Decision**: Docling first (preserves hierarchy). If it fails or returns empty section trees, fall back to Marker with `--use_llm`. Final fallback: PyMuPDF text extraction.
- **Consequences**: Both tools are GPU-friendly on Lightning. Docling-Marker double-pass is acceptable cost at ingest (one-time per doc).

### ADR-007 — API: FastAPI + LitServe; UI: Next.js minimal

- **Context**: Need a queryable HTTP endpoint and a public web UI.
- **Decision**: FastAPI for the API surface (REST + SSE for streaming). LitServe wraps it for Lightning AI deployment with multi-GPU autoscaling. UI is Next.js 14 App Router with one page (`/`) and a streamed answer view. No state, no database, no auth.
- **Consequences**: Minimal moving parts. The UI calls the API directly. No login flow.

### ADR-008 — Evaluation: RAGAS + DeepEval + TruLens

- **Context**: Need offline metrics and CI gates.
- **Decision**: RAGAS for metric computation. DeepEval test cases that block PRs if metrics regress. TruLens (or Langfuse — pick one in Part X) for production monitoring.
- **Consequences**: A golden-set Q&A file lives in the repo at `evals/golden_set.jsonl`. CI runs DeepEval on every PR.

### ADR-009 — Code: Python 3.12, uv for envs, ruff for lint, pyright for types

- **Context**: Avoid the dependency hellscape of v3.
- **Decision**: Pin Python 3.12 (stable, matches Lightning Studios). Use `uv` for dep resolution and lockfile (`uv.lock`). Use `ruff` for lint+format (single tool). Use `pyright` for type-checking. Pre-commit hooks for all three.
- **Consequences**: Modern tooling, fast CI. Replaces the pip + `requirements*.txt` mess.

### ADR-010 — Repo structure: src layout, single package

- **Context**: Current repo mixes scripts and library code with `sys.path` hacks.
- **Decision**: src-layout. Single package `urban_rag/`. CLI entry `urban-rag` via `pyproject.toml`. No `sys.path.insert`.
- **Layout**:
```
urban-planning-rag/
├── pyproject.toml
├── uv.lock
├── README.md
├── PLAN.md (this file)
├── .github/workflows/   # CI
├── docs/                # additional design docs
├── src/
│   └── urban_rag/
│       ├── __init__.py
│       ├── ingest/      # PDF parse + render
│       ├── embed/       # ColQwen + GTE-ModernColBERT loaders
│       ├── index/       # Qdrant + LanceDB clients
│       ├── retrieve/    # query encoding, ANN + MaxSim + RRF + VLM rerank
│       ├── generate/    # Gemini / Qwen-VL generation
│       ├── eval/        # RAGAS + DeepEval + golden set
│       ├── api/         # FastAPI app
│       ├── cli/         # CLI commands (Typer)
│       ├── config.py    # pydantic settings
│       ├── logging.py   # structlog config
│       └── telemetry.py # OTel
├── web/                 # Next.js UI
├── data/                # gitignored; symlink target on Lightning
├── evals/
│   ├── golden_set.jsonl
│   ├── ablations/
│   └── reports/
├── infra/
│   ├── lightning/       # Studio bootstrap scripts
│   ├── docker/          # Qdrant compose, dev containers
│   └── litserve/        # LitServe wrappers
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

### ADR-011 — Single-source config via pydantic-settings + .env

- **Context**: Currently `.env` is read in three different places ad hoc.
- **Decision**: One `Settings` class. All config flows through it. Validation at startup.

### ADR-012 — Logging: structlog JSON

- All logs are structured JSON, one event per line, with correlation IDs propagated through the request context.

### ADR-013 — Telemetry: OpenTelemetry traces + Prometheus metrics

- Every request emits a trace span tree (parse → encode → ann → maxsim → rerank → generate). Metrics exported to Prometheus, scraped by Lightning's metrics stack or a small Grafana on the Studio.

### ADR-014 — Secrets in 1Password / env, never committed

- Rotate the existing Gemini and Lightning keys before any deploy.

### ADR-015 — Idempotent ingestion with content hashing

- Each document is identified by `sha256(file_bytes)`. Re-ingesting the same content is a no-op. Detected updates produce a new content-hash, both versions remain queryable, the latest is default.

---

## Part V — Document Ingestion Pipeline

This section is the recipe for taking a PDF on disk and producing all the artifacts the retrieval system needs.

### 5.1 Stages

```
PDF on disk
   ↓
[1] Validate & hash      → docs/<hash>/source.pdf
   ↓
[2] Render pages         → docs/<hash>/pages/p_0001.png ...
   ↓
[3] Layout-aware parse   → docs/<hash>/parsed.json (Docling)
   ↓
[4] Page classification  → docs/<hash>/pages.jsonl
   ↓
[5] Visual embed (GPU)   → Qdrant collection `pages_visual`
   ↓
[6] Text embed (GPU)     → Qdrant collection `pages_text`
   ↓
[7] BM25 sparse index    → Qdrant sparse vectors on `pages_text`
   ↓
[8] Manifest update      → manifest.parquet (LanceDB)
   ↓
[9] Eval-set check       → if doc has eval queries, validate retrieval
```

Each stage is idempotent and writes its output under a content-addressed directory. Stages can be re-run independently.

### 5.2 Stage 1 — Validate & hash

Inputs: a PDF file path.

```python
def validate_and_hash(pdf_path: Path) -> DocumentRecord:
    bytes_ = pdf_path.read_bytes()
    if len(bytes_) < 1024:
        raise InvalidDocument(f"too small: {pdf_path}")
    if not bytes_.startswith(b"%PDF"):
        raise InvalidDocument(f"not a PDF: {pdf_path}")
    h = hashlib.sha256(bytes_).hexdigest()
    dest = DOCS_DIR / h / "source.pdf"
    if dest.exists():
        return DocumentRecord.load(h)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(bytes_)
    record = DocumentRecord(
        hash=h,
        filename=pdf_path.name,
        size_bytes=len(bytes_),
        ingested_at=now_utc(),
        version=1,
    )
    record.save()
    return record
```

PDF integrity check via PyMuPDF — if `fitz.open` fails or the PDF is encrypted without password, reject and surface to a manual-review queue.

### 5.3 Stage 2 — Render pages

Adaptive DPI (kept from v3):

```python
def classify_page(page) -> PageType:
    has_images = len(page.get_images(full=True)) > 0
    num_drawings = len(page.get_drawings())
    if has_images or num_drawings > 40:
        return PageType.VISUAL  # 250 DPI
    return PageType.TEXT  # 100 DPI
```

Render with PyMuPDF `fitz.Matrix(dpi/72, dpi/72)`. Save PNG with `optimize=True`. For very large docs, consider lossy WebP at quality=85 — saves 60% storage with no measurable retrieval impact (test in eval).

Output: `docs/<hash>/pages/p_NNNN.png` and `docs/<hash>/pages/p_NNNN.meta.json` per page.

### 5.4 Stage 3 — Layout-aware parse (Docling primary)

```python
from docling.document_converter import DocumentConverter

def parse_doc(pdf_path: Path, hash_: str) -> DoclingDocument:
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    if not result.document:
        return parse_doc_marker(pdf_path, hash_)  # fallback
    save_path = DOCS_DIR / hash_ / "parsed.json"
    save_path.write_text(result.document.export_to_dict())
    return result.document
```

Docling output contains: section tree, table extractions (HTML), figure captions, page-region bounding boxes for each text element. We **cross-reference** these regions with the rendered page in stage 4, so a retrieved patch can be tied back to a section in the doc tree.

### 5.5 Stage 4 — Page classification & enrichment

For each page:
- `page_type` (text/visual)
- `dpi`
- `section_path` (from Docling tree, e.g. `["NBC 2016 Vol 1", "Part 4 Fire Safety", "4.2 Definitions"]`)
- `page_text_preview` (first 500 chars of extracted text)
- `tables_count`, `figures_count`
- `language` detected (langdetect)
- `is_appendix`, `is_index`, `is_toc` flags (regex + Docling heuristics)

Output: `docs/<hash>/pages.jsonl`, one JSON line per page.

### 5.6 Stage 5 — Visual embedding

Run on a Lightning AI Studio with an L4 / L40S / A100. The model is loaded once, batches of 8–32 pages are fed through depending on VRAM headroom.

```python
class VisualEmbedder:
    def __init__(self, model_id="vidore/colqwen2.5-v0.2"):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        ).eval()

    @torch.inference_mode()
    def embed_pages(self, page_paths: list[Path]) -> list[torch.Tensor]:
        images = [Image.open(p) for p in page_paths]
        feats = self.processor.process_images(images=images).to("cuda", torch.bfloat16)
        out = self.model(**feats)
        # out.embeddings: list of (num_patches, dim) tensors
        return [e.cpu() for e in out.embeddings]
```

Push to Qdrant `pages_visual` collection (one point per page, multi-vector), with payload `{doc_hash, page_num, section_path, page_type, dpi}`.

### 5.7 Stage 6 — Text embedding

Same Studio, runs after visual embedding finishes (or in parallel on a separate GPU). Uses GTE-ModernColBERT for multi-vector text or BGE-M3 for single-vector (we run both, primary is GTE-ModernColBERT multi-vector for parallelism with the visual path).

Input: chunked text from Docling — chunks are *section-aware*, never crossing section boundaries, with target 256 tokens and 32-token overlap. Each chunk has `{doc_hash, page_num, section_path, chunk_idx, text}`.

Push to Qdrant `pages_text` collection.

### 5.8 Stage 7 — BM25 sparse index

Either (a) Qdrant sparse vectors using `bm25-okapi` tokenizer per chunk, or (b) external Tantivy/Whoosh. We use Qdrant native sparse vectors — same collection as `pages_text`, separate vector field.

### 5.9 Stage 8 — Manifest update

The manifest is the source of truth for "what's in the corpus."

```
manifest.parquet (LanceDB)
columns: doc_hash, filename, num_pages, num_visual_patches, num_text_chunks,
         language, source_jurisdiction, doc_type, ingested_at, indexed_at,
         eval_status, version, parser_used, parser_warnings
```

The CLI command `urban-rag corpus list` reads this and pretty-prints.

### 5.10 Stage 9 — Eval-set validation

If the document is associated with eval queries (e.g., "Q: What is FSI for residential in Mumbai DP 2034?" tied to `mumbai_dp_2034.pdf`), the ingestion harness runs those queries through the just-indexed doc and asserts `Recall@5 >= 0.8`. If it fails, the ingestion is rolled back and a "needs review" record is created.

### 5.11 Source corpus

The target corpus for v1 (~500 docs is aspirational; we ship with these as the floor):

**National**:
- NBC 2016 Vol 1 (~2,200 pages)
- NBC 2016 Vol 2 (~600 pages)
- URDPFI 2014 Vol 1 (~450 pages) [already have]
- URDPFI 2014 Vol 2 (~250 pages) [already have]
- SWM Rules 2016 (~91 pages) [already have]
- IRC codes (selected, ~30 docs)
- BIS structural codes (selected, ~20 docs)
- MoHUA guidance documents (~50 docs)
- AMRUT, Smart Cities Mission, PMAY documents

**State / metro**:
- Delhi MPD-2041
- Mumbai DP-2034
- Bangalore RMP-2031
- Hyderabad MP-2031
- Chennai SMP-2027
- Kolkata MP-2025
- Pune DP, Ahmedabad DP
- + state DCRs / GDCRs

**Total target**: 75,000–120,000 pages, 200–500 docs.

Sourcing: documents come from official ministry / development authority websites. We script polite scrapers per source, with rate limits, robots.txt respect, and full provenance metadata.

### 5.12 Ingestion CLI

```bash
urban-rag ingest <pdf_or_dir> [--rebuild] [--skip-eval]
urban-rag corpus list
urban-rag corpus stats
urban-rag corpus rm <doc_hash>
urban-rag corpus reindex <doc_hash>
```

### 5.13 Data layout (final)

```
data/
├── docs/<sha256>/source.pdf
├── docs/<sha256>/pages/p_NNNN.png
├── docs/<sha256>/parsed.json (Docling output)
├── docs/<sha256>/pages.jsonl
├── manifest.parquet
└── qdrant/                          # local Qdrant data dir (or remote)
```

---

## Part VI — Embedding & Indexing Stack

### 6.1 Visual embedding details

**Model**: `vidore/colqwen2.5-v0.2` (open source, Apache 2.0).

**Inference recipe**:
- bfloat16 on H100 / A100 / L40S
- Flash Attention 2
- Batch size: 16 for L40S (24 GB VRAM), 32 for A100 (80 GB)
- Image preprocessing: keep aspect ratio, max 1280 visual tokens (matches the 250 DPI cap)

**Throughput** (rough, from ColPali inference benchmarks):
- L40S: ~3–5 pages/sec at 1280 visual tokens
- H100: ~10–15 pages/sec
- 100k pages on L40S: ~6–9 hours

**Output schema per page** (Qdrant point):
```json
{
  "id": "<sha256(doc_hash + ':' + page_num)>",
  "vector": {
    "patches": [[...320d...], [...], ...],   // multi-vector, ~30–80 vectors
    "pooled": [...320d...]                    // single vector for ANN candidate gen
  },
  "payload": {
    "doc_hash": "...",
    "doc_filename": "nbc_2016_vol1.pdf",
    "page_num": 142,
    "page_type": "VISUAL",
    "dpi": 250,
    "section_path": ["NBC 2016 Vol 1", "Part 4", "4.2.3"],
    "language": "en",
    "jurisdiction": "national",
    "doc_type": "code",
    "image_uri": "lance://docs/<hash>/pages/p_0142.png"
  }
}
```

The `pooled` single vector is computed as the **token-norm-weighted mean** of patch vectors, not a flat mean — high-information patches dominate. This is the candidate-generation key.

### 6.2 Text embedding details

**Model**: `lightonai/GTE-ModernColBERT-v1` (multi-vector text, ModernBERT 768-dim).

Same Qdrant point structure, separate collection `pages_text`. Each text chunk is an additional point referencing the parent page via payload.

### 6.3 Qdrant collection schemas

```python
client.create_collection(
    "pages_visual",
    vectors_config={
        "patches": models.VectorParams(
            size=320,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
        ),
        "pooled": models.VectorParams(size=320, distance=models.Distance.COSINE),
    },
    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    ),
)
```

**Quantization**: INT8 scalar. Cuts storage 4× with <1% recall loss on ColPali-class embeddings (per Qdrant benchmarks). Always-in-RAM keeps it fast.

### 6.4 Pre-aggregation for ANN candidate stage

The pooled vector is what ANN searches against. The MaxSim rerank is then performed on the multi-vector field for the candidate set Qdrant returns. Both happen in one server-side `query_points` call using Qdrant's `prefetch` syntax:

```python
client.query_points(
    collection_name="pages_visual",
    prefetch=[
        models.Prefetch(
            query=query_pooled.tolist(),
            using="pooled",
            limit=200,
        ),
    ],
    query=query_patches.tolist(),
    using="patches",
    limit=20,
    with_payload=True,
)
```

200 pooled candidates → reranked to top 20 by MaxSim, all server-side. This is the canonical Qdrant 1.13 multi-stage pattern.

### 6.5 Indexing throughput targets

- Visual: 100k pages → ~8 h on a single L40S, ~2 h on H100
- Text: 100k pages × ~10 chunks = 1M chunks → ~4 h on L40S
- Sparse: trivial CPU work

We embed the corpus once on a Lightning Studio with a temporary GPU upgrade, dump to Qdrant, then drop the GPU.

### 6.6 Re-embedding triggers

Re-embed when:
1. The embedding model changes (ColQwen2.5 → ColNomic-7B v1.5 upgrade)
2. The DPI strategy changes
3. A document's underlying file changes (new content_hash)

Re-embedding is a per-doc operation. The pipeline supports `--rebuild` flags at every stage.

---

## Part VII — Retrieval Pipeline

### 7.1 The full request flow

```
HTTP POST /query
  body: {"query": "FSI residential Mumbai", "top_k": 5, "deep": false}
  ↓
[1] query encode (visual)        ColQwen patches      ~200 ms (warm)
[2] query encode (text)          GTE-ModernColBERT    ~30 ms
[3] query encode (sparse)        BM25 tokenization    <1 ms
   ↓ (parallel)
[4] visual ANN+MaxSim            Qdrant prefetch      ~100 ms
[5] text MaxSim                  Qdrant               ~50 ms
[6] BM25                         Qdrant sparse        ~30 ms
   ↓
[7] RRF fusion                   in-process           <5 ms
[8] VLM rerank (top-20→top-5)    Gemini 2.5 Flash     ~1.0–1.5 s (network)
   ↓
[9] generate answer              Gemini 2.5 Flash     ~1.5–2.5 s (with images)
   ↓
HTTP 200 with streamed answer + citations
```

p50: ~2 s. p95: ~4 s. Acceptable.

### 7.2 Query encoding

Three encoders run in parallel:
- ColQwen2.5 on text (yes, ColQwen encodes text queries too — that's the whole "ColPali" approach)
- GTE-ModernColBERT on text
- BM25 tokenization (Indic-friendly, with stemming)

### 7.3 Stage 1 — Three-channel candidate generation

```python
async def candidate_gen(query: str) -> dict[str, list[Candidate]]:
    qv, qt, qs = await asyncio.gather(
        encode_visual(query),
        encode_text(query),
        tokenize_sparse(query),
    )
    visual, text, sparse = await asyncio.gather(
        qdrant_visual.search(qv, top_k=200),
        qdrant_text.search(qt, top_k=200),
        qdrant_sparse.search(qs, top_k=200),
    )
    return {"visual": visual, "text": text, "sparse": sparse}
```

### 7.4 Stage 2 — Reciprocal Rank Fusion

```python
def rrf_fuse(channels: dict[str, list[Candidate]], k: int = 60, top_n: int = 20):
    scores = defaultdict(float)
    payloads = {}
    for ch_name, ch_results in channels.items():
        for rank, cand in enumerate(ch_results):
            scores[cand.id] += 1.0 / (k + rank)
            payloads[cand.id] = cand.payload
    fused = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [Candidate(id=cid, score=s, payload=payloads[cid]) for cid, s in fused]
```

`k=60` is the canonical RRF constant. Channels are equal-weighted by default; we tune this on the eval set in Part X.

### 7.5 Stage 3 — VLM cross-encoder rerank

Send top-20 page images + query to Gemini 2.5 Flash:

```python
RERANK_PROMPT = """You are scoring which document pages most directly answer a query.

Query: {query}

For each numbered page below, give a score 0–10 where:
- 0 = irrelevant
- 5 = related but doesn't answer
- 10 = directly answers the query with specific evidence

Respond as JSON: {{"scores": [{{"page_id": "...", "score": ..., "reason": "..."}}, ...]}}

Pages:
"""

async def vlm_rerank(query: str, candidates: list[Candidate]) -> list[Candidate]:
    images = [load_image(c.payload["image_uri"]) for c in candidates]
    contents = [RERANK_PROMPT.format(query=query)]
    for c, img in zip(candidates, images):
        contents.extend([f"page_id={c.id}", img])
    resp = await gemini.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config={"response_mime_type": "application/json", "response_schema": RerankSchema},
    )
    scores = {s.page_id: s.score for s in resp.parsed.scores}
    for c in candidates:
        c.score = scores.get(c.id, 0)
    return sorted(candidates, key=lambda c: -c.score)[:5]
```

### 7.6 Confidence gating (CRAG-light)

After rerank, compute a confidence signal:
- `conf_high` if `top1_score >= 8` (out of 10) **and** there are at least 3 candidates with `score >= 6`
- `conf_med` if `top1_score >= 5`
- `conf_low` otherwise

In `conf_low` mode, the response includes `confidence: "low"` and the UI shows a "no high-confidence answer found, here are the closest matches" mode rather than a confident answer.

### 7.7 Filters

The query API supports filters:
- `jurisdiction`: `national | state:<name> | metro:<name>`
- `doc_type`: `code | master_plan | dcr | guidance | ...`
- `language`: `en | hi | ta | ...`

These flow through to Qdrant payload filters at stages 1, 4, 5, 6.

### 7.8 Caching

- Query embedding cache: hash query → ColQwen embedding (Redis / in-process LRU). 24 h TTL.
- Candidate cache: hash (query + filters) → fused candidates. 1 h TTL.
- VLM rerank cache: hash (query + candidate-set) → rerank order. 24 h TTL.

### 7.9 Streaming response

Use Server-Sent Events. Stream order:
1. `event: candidates` — top-20 from RRF, with snapshot images URIs (so UI can paint immediately)
2. `event: rerank` — top-5 after VLM rerank
3. `event: answer_token` — generation tokens as they stream
4. `event: citations` — final citation map
5. `event: done`

This lets the UI render quickly while the slow VLM calls finish.

### 7.10 Failure modes & fallbacks

| Failure | Fallback |
|---|---|
| Visual encoder unreachable | Continue with text + sparse; flag `degraded_mode: true` |
| Qdrant down | Return 503 with retry-after; alert |
| VLM rerank timeout | Use MaxSim top-5 directly; flag `vlm_rerank: skipped` |
| Generation timeout | Return retrieved candidates + "answer generation timed out" |
| All channels return zero results | Return empty + suggest broader query |

---

## Part VIII — Generation & Answer Synthesis

### 8.1 Prompt design

```
SYSTEM: You are an expert on Indian urban planning regulations.
You answer questions strictly from the provided document pages. If a page does
not directly support a claim, do not make that claim. Cite using [page_id].

USER:
Query: {query}

You have been given {n} document pages as images. Each page has an ID printed
in the metadata. Pages:
{page_metadata_summary}

Instructions:
1. Read each page carefully, especially tables and diagrams.
2. Compose an answer that directly addresses the query.
3. Use [page_id] inline citations after each factual claim.
4. If the pages don't contain the answer, say so plainly.
5. End with a "Sources" section listing the cited pages with their full names.

Format: Markdown.
```

### 8.2 Why send images, not text

Sending the rendered page image to the VLM is the whole reason we built a visual RAG. The VLM reads the layout, tables, and diagrams the same way a human planner does. Sending Docling-extracted text alongside is *optional*; we A/B test it. Hypothesis: images-only outperforms images+text for layout-heavy queries because the text duplicates information already in the image and steals context budget.

### 8.3 Citation grounding

The response is post-processed: every `[page_id]` citation is verified to exist in the candidate set. Hallucinated page_ids are stripped and an inline `[?]` is left in their place.

### 8.4 Deep mode

When `deep=true` in the request:
- Use Gemini 2.5 Pro instead of Flash
- Return top-10 pages instead of top-5
- Run a second-pass "self-critique" where the model is asked "is the above answer fully supported by the cited pages? Output JSON `{supported: bool, missing_citations: [...]}`. If `supported=false`, the answer is regenerated with a stronger grounding instruction.

This roughly doubles latency and cost. Used sparingly.

### 8.5 Refusal cases

- Query is non-planning ("what's the weather"): "I only answer questions about Indian urban planning regulations."
- Query is hostile / out-of-scope ("write a fake building permit"): refuse, with a non-judgmental explanation.
- Confidence is low: explain that no page strongly answers the query, show top candidates.

### 8.6 Answer schema

```typescript
type AnswerResponse = {
  answer_markdown: string;
  citations: Array<{
    page_id: string;
    doc_hash: string;
    doc_filename: string;
    page_num: number;
    section_path: string[];
    image_uri: string;
    rerank_score: number;
  }>;
  confidence: "high" | "medium" | "low";
  diagnostics: {
    latency_ms: { encode: number; retrieve: number; rerank: number; generate: number; total: number };
    backends: { encoder: string; reranker: string; generator: string };
    candidate_count: { visual: number; text: number; sparse: number; fused: number; reranked: number };
    flags: { vlm_rerank_skipped?: boolean; degraded_mode?: boolean };
  };
  query_id: string;  // for telemetry / feedback
};
```

### 8.7 Hallucination guardrails

- Faithfulness check at evaluation time using RAGAS
- A daily cron sample: pick 100 random production queries, compute faithfulness, alert if median drops below 0.85
- Adversarial probe: a small adversarial test set of "this is not in the corpus" queries; the system must refuse rather than make up plausible answers

---

## Part IX — Web Interface

The web interface is intentionally minimal. The aim is a research-grade query surface that demonstrates the system end-to-end — not a productionised SaaS UI. Anything beyond the requirements below is out of scope until the retrieval and generation stack hits the v1 quality bar.

### 9.1 Tech stack

- **Framework**: Next.js 14 (App Router), TypeScript, React 18 server components where possible
- **Styling**: Tailwind CSS + shadcn/ui primitives (Button, Input, Card, ScrollArea, Tabs, Tooltip)
- **State**: Server actions for the question submit; React Query (TanStack) for client-side polling and cache; no Redux, no Zustand
- **Streaming**: native fetch + ReadableStream consuming the FastAPI SSE/NDJSON stream
- **PDF / image rendering**: `react-pdf` for citation previews; the API returns presigned image URLs to S3-compatible storage so the browser fetches PNGs directly
- **Analytics**: Plausible (self-hosted) for page views; PostHog for product events (query_submitted, answer_received, citation_clicked, feedback_submitted)
- **Auth (v2 only)**: Clerk or NextAuth with magic-link; v1 is unauthenticated behind a Cloudflare-enforced allowlist

### 9.2 Routes & screens

| Route                  | Purpose                                                                          |
|------------------------|----------------------------------------------------------------------------------|
| `/`                    | Landing + query bar; top-3 example questions; corpus stat strip                  |
| `/q/[query_id]`        | Streamed answer with citation chips; permalink-able                              |
| `/q/[query_id]/page/[idx]` | Modal lightbox for a single citation page (PNG + extracted text + filename) |
| `/corpus`              | Browse indexed documents grouped by act/jurisdiction; counts and last-updated    |
| `/about`               | One-paragraph project description, license, links to GitHub & paper             |
| `/admin`               | (Behind feature flag) eval dashboard — RAGAS scores trend, last 7 days           |

There are no chat threads, no persistent conversations, no user accounts in v1. Every query is a fresh request. This avoids hidden state and makes evaluation reproducible.

### 9.3 Query flow (client-side)

1. User types question in `<QueryBar />` and submits (Enter or button)
2. Client POSTs `/v1/ask` with `{question, mode}`; server returns `{query_id}` and a stream URL
3. Client opens `EventSource('/v1/ask/{query_id}/stream')`
4. Server emits SSE events in this exact order: `retrieval_started`, `retrieval_completed` (with citations), `generation_started`, `token` (repeated), `generation_completed`, `done`
5. UI shows skeleton loader during retrieval (≤2s), then materialises citation chips at the top of the answer panel as soon as `retrieval_completed` arrives, then begins streaming tokens into the answer body
6. On `done`, the UI persists the full payload to localStorage keyed by `query_id` so reload works without re-querying

### 9.4 Citation UX

Citations are first-class. Every answer paragraph that contains a `[1]`, `[2]`, etc. inline marker hyperlinks to the corresponding citation chip. Each chip shows: thumbnail of the PDF page (lazy-loaded), document name, page number, and a section title if the section detector resolved one. Clicking the chip opens a side-panel lightbox showing the full-resolution rendered page. There is a "Copy citation" button which copies a structured Markdown reference: `[NBC 2016, Vol II, p.214](https://...)`.

### 9.5 Feedback widget

After every answer, a thumbs-up / thumbs-down with an optional 200-char free-text box. Submitting POSTs `/v1/feedback` with `{query_id, vote, comment}`. The server stores this in a Postgres `feedback` table for offline review and inclusion in the regression eval set.

### 9.6 Performance budget

- **TTFB on `/`** ≤ 200 ms (static; CDN-cached)
- **First citation chip rendered** ≤ 2.5 s after submit (P50)
- **First answer token** ≤ 4 s (P50), ≤ 8 s (P95)
- **Total page weight** ≤ 250 KB gzipped JS, ≤ 50 KB gzipped CSS

The build is checked in CI with `next build --profile` and Lighthouse-CI; regressions over 10% on any budget fail the PR.

### 9.7 Accessibility & i18n

- WCAG 2.2 AA: keyboard-only flow tested; ARIA labels on chips and feedback buttons; focus rings preserved
- Language: English-only in v1. URDPFI uses some Hindi/regional terms for typologies — the UI does **not** translate them; it preserves the source script. v2 may add `lang` switcher.

### 9.8 Out of scope (v1)

Conversational follow-ups, user accounts, saved searches, document upload, comparison mode, mobile-native app, dark mode (use system default), keyboard shortcuts beyond Enter, multi-tenant routing.

---

## Part X — Evaluation Framework

Evaluation is the spine of this project. Without continuous, automated, regression-gated evaluation we cannot tell whether the agent fleet is moving us forward or destroying quality. This section defines the eval methodology, datasets, metrics, gates, and dashboards.

### 10.1 Evaluation principles

1. **Reproducible**: every eval run is keyed by `(corpus_hash, model_versions, eval_set_hash)`. No floating "current" runs.
2. **Layered**: unit (component-level), retrieval-only, end-to-end, adversarial. A regression at any layer is investigated.
3. **Cheap and fast for CI**: < 5 minutes total wall-clock for the smoke set; the comprehensive set runs nightly.
4. **Human-validated**: every metric the system tracks has a human-labelled ground-truth set as anchor. LLM-as-judge is calibrated against the human anchor before being trusted.
5. **Drift-aware**: every model swap or prompt change is benchmarked before merge.

### 10.2 Datasets

Three nested datasets:

#### 10.2.1 Smoke (`eval/smoke.jsonl`)
- 25 hand-curated questions covering every document type (NBC, URDPFI v1, URDPFI v2, SWM Rules, IRC, MPD)
- Each entry: `{question, expected_documents: [...], expected_pages: [...], expected_keywords: [...], answer_rubric: "..."}`
- Used in CI on every PR. Must pass 100% retrieval recall@10 and 90% on the rubric.

#### 10.2.2 Regression (`eval/regression.jsonl`)
- 200 questions including all historic failures, edge cases, adversarial probes
- Built incrementally — every production failure triaged into the regression set
- Run nightly, on every release candidate, and after every model swap

#### 10.2.3 Comprehensive (`eval/comprehensive.jsonl`)
- 1,000 questions sampled across:
  - 500 from synthetic generation (LLM-generated from corpus passages with answer keys)
  - 300 from real user logs (anonymised, deduped)
  - 100 from urban planning postgrad question banks (CEPT, SPA Delhi exam papers — fair-use research)
  - 100 adversarial: out-of-corpus, ambiguous, multi-document, computational
- Run weekly; full RAGAS suite + DeepEval gates; human spot-check 50 random samples per run

### 10.3 Metrics

#### 10.3.1 Retrieval metrics (cheap, deterministic)

| Metric                   | Definition                                                          | Target |
|--------------------------|---------------------------------------------------------------------|--------|
| `recall@k` for k∈{5,10,20} | fraction of expected pages retrieved                              | ≥ 0.85 @ 10 |
| `mrr@10`                  | mean reciprocal rank of first correct page                         | ≥ 0.65 |
| `ndcg@10`                 | normalised discounted cumulative gain                              | ≥ 0.70 |
| `coverage@10`             | fraction of expected documents represented in top-10               | ≥ 0.95 |
| `latency_p50`, `latency_p95` | retrieval-only latency                                          | ≤ 600 ms / ≤ 1500 ms |

#### 10.3.2 Generation metrics (LLM-as-judge, expensive)

| Metric             | Definition                                                                | Target |
|--------------------|---------------------------------------------------------------------------|--------|
| `faithfulness`     | RAGAS — fraction of claims supported by retrieved context                | ≥ 0.90 |
| `answer_relevance` | RAGAS — semantic relevance of answer to question                         | ≥ 0.85 |
| `context_precision`| RAGAS — fraction of retrieved context that is actually useful            | ≥ 0.70 |
| `context_recall`   | RAGAS — fraction of ground-truth answer covered by retrieved context     | ≥ 0.85 |
| `answer_correctness`| RAGAS — composite vs ground-truth answer                                | ≥ 0.80 |
| `citation_accuracy`| custom — fraction of `[k]` markers that point to a correct page         | ≥ 0.95 |
| `refusal_correctness`| custom — on adversarial out-of-corpus, fraction correctly refusing    | ≥ 0.90 |

#### 10.3.3 End-to-end UX metrics

- `time_to_first_token_p50/p95`
- `time_to_completion_p50/p95`
- `cost_per_query_usd` (sum of embedding, retrieval infra amortised, VLM rerank, generation)

### 10.4 Tooling

- **RAGAS** (≥ 0.2.0) for the standard metrics. Wrap with our own runner that pins `model='gemini-2.5-flash'` for the judge to avoid cost drift.
- **DeepEval** for CI gates; defines `assert_test` per metric so a PR fails on regression. Integrated with pytest.
- **TruLens** for production telemetry; samples 5% of live queries and computes faithfulness in the background.
- **Langfuse** for trace storage; every query gets a trace with retrieval candidates, rerank scores, prompts, and outputs. Searchable dashboard for triage.
- **promptfoo** for prompt regression tests (when we change prompts only, not models — avoids dragging the full eval suite for a copy edit).

### 10.5 Eval runner architecture

```
src/eval/
  __init__.py
  runner.py           # orchestrates a run against a dataset
  metrics/
    retrieval.py      # recall, mrr, ndcg, coverage
    ragas_wrapper.py  # pinned RAGAS calls
    citation.py       # structured citation accuracy check
    refusal.py        # out-of-corpus refusal check
  datasets/
    loader.py         # JSONL → typed records
    synth.py          # synthetic question generation from corpus
  reports/
    html.py           # static HTML dashboard
    json.py           # machine-readable for CI gates
    diff.py           # compare two runs side-by-side
  cli.py              # `python -m src.eval run --dataset smoke --tag pr-123`
```

Runs write to `eval/runs/<tag>/<timestamp>/` with all raw outputs preserved for post-hoc analysis.

### 10.6 Gating policy

- **PR gate**: smoke set must pass; recall@10 ≥ 0.85, faithfulness ≥ 0.85 on smoke
- **Merge gate**: regression set must not drop > 2% on any metric vs main
- **Release gate**: comprehensive set faithfulness ≥ 0.90, answer_correctness ≥ 0.80; human spot-check approval
- **Production canary**: new release rolled to 5% of traffic; if `faithfulness_5pct - faithfulness_main > -0.03` for 24 hours, ramp to 100%

### 10.7 Human-in-the-loop calibration

Every Friday, an agent samples 50 production queries and produces:
- LLM-judge scores
- A side-by-side comparison panel for human review (Tanmay or a delegate)
- Disagreements where human and judge differ by > 1 point on a 5-point rubric

These disagreements seed prompt fixes for the judge; the judge is re-anchored monthly.

### 10.8 Adversarial test set

Examples of probes:
- "What is the FSI for Mars colonies?" → must refuse or say not in corpus
- "Compare NBC 2016 with NBC 2024" → must say NBC 2024 is not indexed if it isn't
- "Give me the exact text of section 5.3" → must quote verbatim from the page or refuse
- Numeric: "If FAR is 2.5 and plot is 500m², what's max built-up?" → must compute correctly using retrieved FAR rules
- Multi-doc: "Compare URDPFI v1 and v2 on parking standards" → must cite both versions distinctly

### 10.9 Eval dashboard

A static HTML dashboard generated per run:
- Top: aggregate metrics with traffic-light icons vs target
- Middle: per-question table with question, retrieval results, answer, scores, judge rationale
- Bottom: regression delta vs previous run; new failures highlighted
- Persisted to `https://eval.urban-rag.internal/<run_tag>/`

---

## Part XI — Deployment on Lightning AI Studios

Compute happens on rented GPUs. The deployment plan below is opinionated about Lightning AI because it matches Tanmay's existing account (`LIGHTNING_USER_ID=4fda5561-...`) and Lightning Studios offer a clean Studio → Job → Deployment flow that suits this project.

### 11.1 Topology

```
                ┌─────────────────────┐
                │  Cloudflare (CDN)   │
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                │   Next.js (Vercel)  │
                └──────────┬──────────┘
                           │  HTTPS
                ┌──────────┴──────────┐
                │  FastAPI Gateway     │
                │  (Lightning AI Job)  │
                │  CPU, autoscale 0–4 │
                └──────────┬──────────┘
                           │  gRPC / HTTP
        ┌──────────────────┼──────────────────────┐
        │                  │                      │
┌───────┴────────┐ ┌──────┴───────┐    ┌──────────┴────────┐
│ Embedding svc  │ │ Qdrant Cloud │    │ VLM rerank + gen  │
│ ColQwen / Nem. │ │ (managed)    │    │ Gemini API +      │
│ A100 40GB,     │ │              │    │ self-host fallback│
│ scale-to-zero  │ │              │    │ Qwen2.5-VL on H100│
└────────────────┘ └──────────────┘    └───────────────────┘
                           │
                ┌──────────┴──────────┐
                │ S3 (page PNGs)      │
                │ Postgres (metadata, │
                │ feedback, runs)     │
                └─────────────────────┘
```

### 11.2 Why this split

- **Gateway is CPU-bound** and small; running it on a Studio CPU instance is ~$0.10/hr and autoscales fine.
- **Embedding service is GPU-bound** for query encoding (single forward pass, ColQwen ~150ms on A100). Scale-to-zero with cold-start budget of 30s on first query of the day; subsequent traffic stays warm.
- **Qdrant** is managed (Qdrant Cloud) — no reason to run our own when their pricing is reasonable for our scale (< 100k vectors v1, < 1M v2). Backups + multi-AZ included.
- **VLM rerank + generation** uses Gemini API by default (no infra to manage); the self-hosted fallback on H100 is **only spun up** when we need data residency or when Gemini is throttled.

### 11.3 Lightning AI Studio layout

We maintain three Studios:

1. **`urban-rag-dev`** — Tanmay's interactive Studio for code, notebooks, debugging. A100 40GB.
2. **`urban-rag-batch`** — for batch indexing jobs. H100 80GB, on-demand only. Driven by `lightning run job ...` from CI.
3. **`urban-rag-prod-embed`** — production embedding service. A100 40GB, autoscale 0→2, deployed via Lightning's `lightning serve` model serving.

### 11.4 Container stack

Single Dockerfile with build targets:

```dockerfile
# Base — shared deps
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base
RUN apt-get update && apt-get install -y python3.12 python3-pip git \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen

# Embedding service
FROM base AS embed
COPY src/embed/ src/embed/
COPY src/common/ src/common/
ENV CUDA_VISIBLE_DEVICES=0
CMD ["uv", "run", "python", "-m", "src.embed.serve"]

# Gateway
FROM base AS gateway
COPY src/api/ src/api/
COPY src/common/ src/common/
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Batch indexer
FROM base AS index
COPY src/ src/
CMD ["uv", "run", "python", "-m", "src.index.batch"]
```

### 11.5 Lightning serve config

```yaml
# .lightning/embed.yaml
name: urban-rag-embed
machine: A10G_40GB
replicas:
  min: 0
  max: 2
  scale_up_cooldown: 30s
  scale_down_cooldown: 300s
healthcheck: /health
ports:
  - 8001
env:
  - QDRANT_URL: ${{ secrets.QDRANT_URL }}
  - QDRANT_API_KEY: ${{ secrets.QDRANT_API_KEY }}
  - HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

### 11.6 Indexing job pipeline

Indexing is a one-shot per-corpus-version job; not a long-running service. Triggered manually or via CI when corpus changes:

```bash
lightning run job \
  --name "index-$(date +%Y%m%d-%H%M)" \
  --machine H100 \
  --num_devices 1 \
  --command "uv run python -m src.index.batch \
              --corpus-version v1.2.0 \
              --input s3://urban-rag-corpus/v1.2.0/ \
              --output qdrant://urban-rag-prod"
```

A full index of 50,000 pages takes ~6h on a single H100 with mixed-precision. Cost ≈ $25 per full index.

### 11.7 Secrets management

- All secrets in Lightning Secrets (`lightning secrets set HF_TOKEN ...`)
- Local dev uses `.env.local` (gitignored) sourced from `direnv`
- CI uses GitHub Actions secrets, mirrored to Lightning via a sync job
- The leaked keys in `.env` (Gemini, Lightning) are **rotated immediately** as the first agent task — see Part XIV §1.

### 11.8 CI/CD

- GitHub Actions on every push:
  1. `ruff format --check && ruff check`
  2. `pyright`
  3. `pytest -q` (unit tests, no GPU)
  4. `python -m src.eval run --dataset smoke --tag $GITHUB_SHA`
  5. Build Docker images, push to GHCR
  6. On `main` merge: `lightning deploy ...` to bump the prod replicas to the new image

- Nightly cron: full regression eval, post results to Slack/Discord.

### 11.9 Rollback & blast radius

- Every Lightning deployment is versioned; rollback = `lightning deploy rollback urban-rag-embed`
- Qdrant collections are immutably named by corpus version (e.g., `pages_v1_2_0`); the gateway routes via an alias collection. Switching to a new version is a single API call; rollback is the same call to the previous alias target.
- Page PNGs in S3 are versioned + lifecycle-rule-protected (90-day soft-delete retention)

### 11.10 Disaster recovery

- Qdrant snapshots: hourly automatic, retained 7 days; daily manual export to S3, retained 90 days
- Postgres: pgBackRest daily full + WAL streaming; PITR window of 14 days
- Page PNG corpus: S3 versioning + cross-region replication (`ap-south-1` → `eu-west-1`)
- Source PDFs: original copies in `s3://urban-rag-source/` (cold storage, Glacier IR)

### 11.11 Cost model summary

| Component         | Monthly cost (USD, ballpark) | Notes                                     |
|-------------------|------------------------------|-------------------------------------------|
| Lightning compute | $80–$160                     | Embed service + occasional batch indexing |
| Qdrant Cloud      | $50                          | 1 GB plan, sufficient through v1          |
| S3 + egress       | $15                          | ~30 GB images, low egress                 |
| Postgres (Neon)   | $20                          | 0.5 GB DB, good for telemetry             |
| Gemini API        | $20–$80                      | Depends on query volume                   |
| Vercel + Cloudflare| $0                          | Free tier covers v1 traffic               |
| **Total**         | **~$185–$325/month**         |                                           |

Detailed model in Appendix D.

---

## Part XII — Engineering Standards

The codebase has been incoherent. Five competing pipelines coexist; comments contradict implementations; types are mostly absent; tests are notional. This section defines the standards every contributor — human or agent — must follow. Treat this as the durable contract. Agents that drift from it are reverted.

### 12.1 Project layout (`src/` layout)

```
urban-planning-rag/
├── pyproject.toml          # uv-managed, single source of truth
├── uv.lock
├── ruff.toml
├── pyrightconfig.json
├── .pre-commit-config.yaml
├── README.md
├── PLAN.md                 # this document; living
├── docs/                   # narrative docs (architecture, runbooks)
├── src/
│   ├── common/             # shared types, settings, logging
│   ├── ingest/             # parsing, chunking, classification
│   ├── embed/              # embedding service + model loaders
│   ├── index/              # batch indexers, Qdrant client wrappers
│   ├── retrieve/           # multi-channel retrieval, fusion, rerank
│   ├── generate/           # VLM clients, prompt builders, streaming
│   ├── api/                # FastAPI app, schemas, middleware
│   ├── eval/               # eval harness, metrics, runners
│   └── ops/                # health, telemetry, scripts
├── web/                    # Next.js app
├── eval/                   # eval datasets + run artifacts (mostly gitignored)
├── scripts/                # CLI utilities (NOT executed in production)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── data/                   # gitignored, samples-only checked in
└── .github/workflows/
```

The old top-level `cli.py`, `query_simple.py`, `query_vision.py`, `requirements*.txt`, and `HYBRID_RAG_V4.md` are **removed**, not preserved. See Appendix A.

### 12.2 Python toolchain

- **Python 3.12** pinned via `.python-version`
- **uv** for env + dep resolution; `pip` is forbidden in CI and dev
- **ruff** for lint + format; line length 100; `select = ["E","F","I","UP","B","SIM","RUF"]`
- **pyright** in strict mode for `src/`; basic mode for `scripts/`
- **pytest** with `pytest-asyncio`, `pytest-cov` (≥ 80% target on `src/common`, `src/retrieve`, `src/eval`)
- **pydantic v2** for all data contracts; **pydantic-settings** for config
- **structlog** for logging; never use `print` in `src/`
- **httpx** for HTTP; never `requests`
- **anyio** for async helpers; avoid mixing `asyncio` + `trio` styles

### 12.3 Type discipline

Strict typing is non-negotiable in `src/`. The `Any` type is allowed only when interfacing with libraries that lack stubs, and must be ringfenced behind a typed wrapper. No untyped `dict`/`list` for return types of public functions. Pydantic models are the lingua franca for cross-module data flow.

### 12.4 Configuration

Every configurable value lives in `src/common/settings.py`:

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="URBAN_RAG_")

    # Embedding
    embed_model_id: str = "vidore/colqwen2.5-v0.2"
    embed_dim: int = 320
    embed_dtype: str = "bfloat16"

    # Qdrant
    qdrant_url: str = Field(...)
    qdrant_api_key: str = Field(...)
    qdrant_collection_visual: str = "pages_visual_v1"

    # Generation
    gemini_api_key: str = Field(..., validation_alias="GEMINI_API_KEY")
    vlm_default_model: str = "gemini-2.5-flash"
    vlm_deep_model: str = "gemini-2.5-pro"

    # Eval
    eval_judge_model: str = "gemini-2.5-flash"
    eval_smoke_path: str = "eval/smoke.jsonl"

    # Observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    otel_endpoint: str | None = None

settings = Settings()
```

There is exactly one `Settings` instance, instantiated at module import. No `os.environ.get` scattered across the code.

### 12.5 Logging conventions

- All logs through `structlog`
- Always include `query_id`, `corpus_version`, `model_version` in the log context
- Log levels: `debug` for dev, `info` for production happy-path, `warning` for retry-recoverable, `error` for user-visible failure, `critical` only for "page someone now"
- No PII in logs; user questions are hashed for analytics, the raw text is only stored in the `queries` table behind auth

### 12.6 Testing standards

- **Unit tests**: every public function in `src/common`, `src/retrieve.fusion`, `src/eval.metrics` has a unit test. Pure functions get property-based tests via `hypothesis` where useful.
- **Integration tests**: spin up Qdrant in Docker, run an index → query → assert flow on a 5-page synthetic corpus.
- **End-to-end tests**: hit the FastAPI gateway with a real query against the smoke corpus; assert structured response shape and citation correctness.
- **Snapshot tests**: prompt outputs are snapshot-tested with `syrupy`; PR diffs surface prompt-driven behavior changes.

CI does not skip on flake; flaky tests are quarantined with `@pytest.mark.flaky` and fixed within 7 days or removed.

### 12.7 Branching & commit hygiene

- Trunk-based: `main` is always deployable
- Short-lived feature branches: `agent/<task-id>` for agent work, `feat/<slug>` for human work
- Conventional commits: `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`, `eval:`, `infra:`
- Each commit must compile, type-check, and pass the smoke eval
- Squash-merge on PR; the squashed commit message becomes the changelog entry

### 12.8 Pull request rules

- One logical change per PR; under 600 LOC diff is the soft cap, agents must split larger changes
- Every PR must include:
  - A short "what" + "why" in the description
  - Eval delta vs main on smoke set
  - Updated docs if behavior changes
- Two passing checks required to merge:
  1. CI green (lint, type, tests, smoke eval)
  2. A human or higher-tier agent (Opus / Sonnet) approval for non-trivial changes

### 12.9 Dependency governance

- Every dep added must be justified in the PR description
- New dep must be on PyPI, MIT/Apache/BSD-licensed (no GPL in core), maintained (last release < 12 months)
- Pinned via `uv.lock`; Renovate bot opens weekly upgrade PRs
- We avoid:
  - LangChain (too churny; we want explicit pipelines)
  - LlamaIndex as a runtime (we use it for parsing utilities only)
  - Any package that bundles transitively pinned scientific libs at incompatible versions

### 12.10 Code-style preferences

- Imports: stdlib → third-party → first-party; sorted by `ruff`
- Function signatures: prefer keyword-only args for anything past the first 2 positional args
- Errors: define a small `src/common/errors.py` hierarchy (`UrbanRAGError`, `RetrievalError`, `GenerationError`, etc.); raise these, not bare `Exception`
- Async: use `async def` end-to-end through the request path; sync glue allowed in batch jobs
- Comments: only when explaining *why*; never *what*. (Aligns with this project's CLAUDE.md.)

### 12.11 Prompt engineering as code

All prompts live in `src/generate/prompts/` as `.md` files with YAML frontmatter:

```yaml
---
id: answer.default
version: 3
input_vars: [question, citations]
output_format: markdown_with_inline_citations
judge_anchor: eval/anchors/answer_default.jsonl
---
You are an expert on Indian urban planning regulations. Answer the question using ONLY the provided pages...
```

Loading is done by `src/generate/prompts/loader.py` which validates input vars + tracks the version in every trace. A prompt change is a code change is a PR.

---

## Part XIII — Observability & Telemetry

You can't fix what you can't see. The system emits enough signal to debug a bad answer 6 weeks after the fact.

### 13.1 Trace model (Langfuse)

Every query is a trace with these spans, in order:

```
trace: query                    (root, attrs: query_id, mode, user_hash, corpus_version)
  ├─ span: retrieval
  │   ├─ span: query_expansion  (prompt, expanded_queries)
  │   ├─ span: visual_search    (qdrant, candidates, latency)
  │   ├─ span: text_search      (qdrant, candidates, latency)
  │   ├─ span: sparse_search    (bm25, candidates, latency)
  │   ├─ span: fusion           (rrf inputs/output)
  │   └─ span: vlm_rerank       (model, scored_candidates, latency, cost)
  ├─ span: generation
  │   ├─ span: prompt_build     (template_id, template_version, tokens)
  │   └─ span: vlm_generate     (model, prompt_tokens, output_tokens, latency, cost)
  └─ span: post_process         (citation_validation, schema_check)
```

### 13.2 Metrics (OpenTelemetry → Prometheus)

- `urban_rag_query_total{mode,status}` — counter
- `urban_rag_query_latency_seconds_bucket{mode,phase}` — histogram (phase ∈ {retrieval, rerank, generation, total})
- `urban_rag_retrieval_recall_at_10` — gauge from sampled live evals
- `urban_rag_faithfulness_p50` — gauge from background TruLens
- `urban_rag_cost_usd_total{component}` — counter
- `urban_rag_qdrant_latency_seconds_bucket{op}` — histogram

Dashboards in Grafana, three views:
- **Operator**: latency, error rate, queue depth, cost
- **Quality**: faithfulness, recall, refusal correctness
- **Capacity**: GPU utilisation, embedding throughput, scaling events

### 13.3 Alerting

- P0: gateway 5xx rate > 1% over 5 min → page on-call
- P1: faithfulness_p50 < 0.80 over 1h → ticket + Slack
- P2: cost_per_query 24h-rolling > $0.05 → ticket
- P3: embed service cold-start > 60s → ticket

### 13.4 Audit log

A separate write-once audit log captures: every model swap, every prompt version bump, every Qdrant collection switch, every secret rotation. Stored append-only in `audit_log` Postgres table; backed up daily.

### 13.5 Replay capability

Given a `query_id`, an operator can re-run that exact query against any past corpus version + model version using the trace's stored config. This is essential for "did we actually regress?" debates.

### 13.6 Privacy posture

- User questions and answers are stored for 30 days then aggregated/discarded
- PII detection (names, phone numbers, emails) on inbound queries with masking before logging
- Discord/Hermes integration follows `mcp__hermes__permissions_*` flow; never logs raw tokens

---

## Part XIV — Agent Execution Roadmap

This roadmap is the executable contract for the autonomous agent fleet. Every task is atomic, has acceptance criteria, and has a clearly-named branch + PR template. Tasks are dependency-ordered and tagged with the agent tier expected to handle them.

Agent tier conventions:
- **A** (Opus 4.7 / GPT-5 class): architectural decisions, prompt design, eval calibration
- **B** (Sonnet 4.6 / Claude Code default): feature implementation, refactors, test writing
- **C** (Haiku 4.5 / Sonnet for cheap parallel tasks): mechanical edits, doc generation, dependency bumps

Every task includes: id, tier, deps, deliverable, acceptance.

### 14.1 Phase 0 — Hygiene (run before anything else)

#### Task 00-01 [tier C, no deps]
**Rotate leaked secrets.** The `.env` checked into the workspace contains `GEMINI_API_KEY`, `LIGHTNING_API_KEY`, `LIGHTNING_USER_ID`. Rotate all three. Push new values to Lightning Secrets and `.env.local`. Confirm none of the old values are anywhere in git history (run `git log -p -S 'AIzaSyAnrBWsA4'`). If found in history, document for rewrite consideration but don't rewrite without approval.
- **Acceptance**: old keys revoked at provider; new keys verified working via a smoke `curl`

#### Task 00-02 [tier C, deps: 00-01]
**Add `.env` to `.gitignore` and ensure `.env.example` exists** with placeholder keys but no real values. Verify `.env` is currently NOT tracked.
- **Acceptance**: `git check-ignore .env` returns true; `.env.example` is committed

#### Task 00-03 [tier C, no deps]
**Delete dead code per Appendix A.** One PR per logical group (don't bundle).
- **Acceptance**: every file in Appendix A removed; CI green

#### Task 00-04 [tier B, deps: 00-03]
**Initialise `pyproject.toml` with uv, ruff, pyright, pytest config.** Migrate from `requirements*.txt`. Pin Python 3.12.
- **Acceptance**: `uv sync` succeeds; `ruff check` passes; `pyright` passes on the (mostly empty) `src/`

#### Task 00-05 [tier B, deps: 00-04]
**Restructure to `src/` layout.** Move `src/rag.py` etc. into the new submodule layout per §12.1. Update imports. Keep behavior identical for the moment.
- **Acceptance**: `pytest` (existing tests, even if empty) passes; module paths resolve

### 14.2 Phase 1 — Foundations

#### Task 01-01 [tier A, deps: 00-04]
**Author `src/common/settings.py`** per §12.4. Wire all entry points to use it; remove every `os.environ` lookup outside this file.
- **Acceptance**: grep for `os.environ` in `src/` returns only `settings.py`

#### Task 01-02 [tier B, deps: 01-01]
**Author `src/common/logging.py`** with structlog config. Every entry point installs the config at startup.
- **Acceptance**: a sample log line in JSON has `query_id`, `corpus_version`, `service`

#### Task 01-03 [tier B, deps: 01-01]
**Author `src/common/errors.py`** with the error hierarchy from §12.10.
- **Acceptance**: types exist, are exported, and are documented in the module docstring

#### Task 01-04 [tier B, deps: 00-05]
**Author `src/common/types.py`** with pydantic models for `DocumentRecord`, `PageRecord`, `ChunkRecord`, `RetrievalCandidate`, `Citation`, `Answer`, `Trace`. These are the inter-module contract.
- **Acceptance**: every downstream module imports from here; no duplicate type definitions

### 14.3 Phase 2 — Ingestion

#### Task 02-01 [tier B, deps: 01-04]
**Implement `src/ingest/load.py`** — `validate_and_hash`, content addressing, manifest writing. Tests on three sample PDFs.
- **Acceptance**: idempotent — re-running on same PDF doesn't duplicate; manifest includes SHA256

#### Task 02-02 [tier B, deps: 02-01]
**Implement `src/ingest/parse.py`** with Docling primary + Marker fallback. Output: structured `PageRecord` + `LayoutBlock` list.
- **Acceptance**: parses NBC 2016 Vol I & II, URDPFI v1, URDPFI v2, SWM Rules without crashes; ≥ 95% page count match between PDF and parsed output

#### Task 02-03 [tier B, deps: 02-02]
**Implement `src/ingest/classify.py`** — page classifier (`TEXT` vs `VISUAL`) per Part V §5.3. Adaptive DPI rendering pipeline.
- **Acceptance**: on a 200-page mixed sample, classifier accuracy ≥ 95% vs hand-labeled set; rendering produces correct-DPI PNGs

#### Task 02-04 [tier B, deps: 02-03]
**Implement `src/ingest/chunk.py`** — hierarchical chunking (doc → section → block → page → patch). Persist to `chunks` table.
- **Acceptance**: every chunk has parent links navigable both directions; total chunk count within ±10% of expected for a known document

#### Task 02-05 [tier B, deps: 02-04]
**Implement `src/ingest/sections.py`** — section-title detection using TOC parsing + regex + LLM fallback for ambiguous cases. Outputs `(page_number, section_title, section_id)` triples.
- **Acceptance**: on URDPFI v2 chapter-titled pages, section detector correctly identifies the section for ≥ 90%

### 14.4 Phase 3 — Embedding & Index

#### Task 03-01 [tier A, deps: 01-04]
**Decide v1 model: ColQwen2.5 v0.2** (per ADR-002). Document model card, prompt template, expected dim, max sequence length.
- **Acceptance**: `docs/models.md` updated with the decision and benchmarks

#### Task 03-02 [tier B, deps: 03-01]
**Implement `src/embed/colqwen.py`** — model loader with bf16, batch encoding, multivector output. CPU fallback for tests.
- **Acceptance**: encoding speed ≥ 5 pages/s on A100; output shape matches contract `(num_pages, num_patches, 320)`

#### Task 03-03 [tier B, deps: 03-02, 02-04]
**Implement `src/embed/serve.py`** — FastAPI service exposing `/embed` and `/embed_query` endpoints. Used by both batch indexing and live queries.
- **Acceptance**: passes a contract test that asserts query and doc embeddings produce expected MaxSim scores on a synthetic pair

#### Task 03-04 [tier B, deps: 03-02]
**Implement `src/index/qdrant_client.py`** — collection bootstrapping with multivector + scalar quantization config from §6.4.
- **Acceptance**: idempotent collection creation; alias-based versioning works

#### Task 03-05 [tier B, deps: 03-04, 03-03]
**Implement `src/index/batch.py`** — full corpus indexing job. Reads chunks from disk, calls embed service, upserts to Qdrant in batches of 64.
- **Acceptance**: indexes the existing 738-page corpus to Qdrant in ≤ 30 min on H100; row counts match

#### Task 03-06 [tier B, deps: 03-05]
**Implement `src/index/text_index.py`** — GTE-ModernColBERT parallel text index for the same corpus.
- **Acceptance**: separate Qdrant collection populated; cross-channel diversity test (same query → different top hits) confirms it isn't just a copy of the visual index

#### Task 03-07 [tier B, deps: 03-05]
**Implement `src/index/sparse.py`** — BM25 (rank_bm25) over chunk text + section titles for keyword fallback.
- **Acceptance**: searchable; BM25 alone hits ≥ 0.50 recall@10 on smoke set

### 14.5 Phase 4 — Retrieval

#### Task 04-01 [tier B, deps: 03-05, 03-06, 03-07]
**Implement `src/retrieve/visual.py`** — visual channel with prefetch (pooled) → MaxSim (multivector). Returns `RetrievalCandidate` list.
- **Acceptance**: smoke test: query "FAR for residential" returns at least 3 URDPFI pages in top 10

#### Task 04-02 [tier B, deps: 04-01]
**Implement `src/retrieve/text.py`** — GTE-ModernColBERT channel.
- **Acceptance**: smoke test similar to 04-01 with text-heavy queries

#### Task 04-03 [tier B, deps: 04-02]
**Implement `src/retrieve/sparse.py`** — BM25 channel.
- **Acceptance**: keyword query like "Schedule II" hits the right page

#### Task 04-04 [tier B, deps: 04-01, 04-02, 04-03]
**Implement `src/retrieve/fusion.py`** — RRF (k=60) across channels; returns top-N with provenance.
- **Acceptance**: each candidate carries which channel(s) it came from and per-channel ranks

#### Task 04-05 [tier A, deps: 04-04]
**Design VLM rerank prompt** per Part VII §7.7. Anchor with 30 hand-labeled (query, page, relevance) triples.
- **Acceptance**: judge prompt + anchor set committed; rerank judge agreement with humans ≥ 0.75 Cohen's κ

#### Task 04-06 [tier B, deps: 04-05]
**Implement `src/retrieve/rerank.py`** — Gemini 2.5 Flash cross-encoder rerank with structured JSON output, retry, timeout.
- **Acceptance**: integration test reorders fusion output; recall@5 after rerank ≥ recall@10 before rerank on smoke set

#### Task 04-07 [tier B, deps: 04-04]
**Implement `src/retrieve/expand.py`** — query expansion (HyDE + synonym + acronym). Optional flag.
- **Acceptance**: A/B on 50 queries shows expansion improves recall@10 by ≥ 3 points

#### Task 04-08 [tier B, deps: 04-06]
**Implement `src/retrieve/orchestrator.py`** — single entrypoint that runs expansion → fan-out → fusion → rerank with timeouts and degraded-mode handling.
- **Acceptance**: returns a fully-typed `RetrievalResult`; degraded mode test passes (kill rerank → still returns fused candidates)

### 14.6 Phase 5 — Generation

#### Task 05-01 [tier A, deps: 04-08]
**Author the answer prompt** (`src/generate/prompts/answer.default.md`) per Part VIII §8.3. Include refusal + citation requirements.
- **Acceptance**: smoke run produces well-formed answers with `[k]` citations

#### Task 05-02 [tier B, deps: 05-01]
**Implement `src/generate/gemini.py`** — streaming Gemini 2.5 Flash client with image attachment.
- **Acceptance**: streams tokens; max-output cap enforced; cost recorded in trace

#### Task 05-03 [tier B, deps: 05-02]
**Implement `src/generate/orchestrator.py`** — `answer(query, retrieval_result, mode)` returning streamed `Answer` events.
- **Acceptance**: e2e test on smoke produces grounded answers with valid citations

#### Task 05-04 [tier B, deps: 05-03]
**Implement self-host fallback (`src/generate/qwen_vl.py`)** behind a feature flag. Used only when Gemini quota is exceeded.
- **Acceptance**: fallback can serve a query end-to-end on a self-hosted endpoint; integration test mocks Gemini failure

### 14.7 Phase 6 — API & Web

#### Task 06-01 [tier B, deps: 05-03]
**Implement `src/api/main.py`** — FastAPI app with `/v1/ask`, `/v1/ask/{id}/stream`, `/v1/feedback`, `/v1/healthz`, `/v1/corpus`.
- **Acceptance**: OpenAPI doc generated; conformance test against schema in Appendix C passes

#### Task 06-02 [tier B, deps: 06-01]
**Implement SSE streaming layer** in the gateway, mapping orchestrator events to wire events.
- **Acceptance**: client receives ordered events; reconnection works on flaky networks

#### Task 06-03 [tier B, deps: 06-01]
**Bootstrap Next.js 14 app** in `web/` with `/` page, query bar, citation chips, streaming consumer.
- **Acceptance**: end-to-end happy path: type query → receive streamed answer with at least 3 citations rendered

#### Task 06-04 [tier B, deps: 06-03]
**Implement citation lightbox** + feedback widget.
- **Acceptance**: click chip → lightbox shows page PNG + extracted text; feedback POSTs and persists

#### Task 06-05 [tier C, deps: 06-03]
**Implement `/corpus` and `/about` pages.** Static, server-rendered.
- **Acceptance**: lighthouse score ≥ 95 on both

### 14.8 Phase 7 — Eval

#### Task 07-01 [tier A, deps: 04-08, 05-03]
**Build the smoke set (`eval/smoke.jsonl`)** — 25 hand-curated questions per §10.2.1.
- **Acceptance**: each entry validates against the schema; manual sanity check pass

#### Task 07-02 [tier B, deps: 07-01]
**Implement `src/eval/runner.py`** — orchestrates a run; writes per-question JSON + aggregate summary.
- **Acceptance**: `python -m src.eval run --dataset smoke` produces a results dir with all expected files

#### Task 07-03 [tier B, deps: 07-02]
**Implement `src/eval/metrics/retrieval.py`** — recall@k, MRR, NDCG, coverage.
- **Acceptance**: unit-tested against synthetic toy data with known answers

#### Task 07-04 [tier B, deps: 07-02]
**Implement `src/eval/metrics/ragas_wrapper.py`** — wired to RAGAS with pinned judge model.
- **Acceptance**: scores reproducibly within ±0.02 across 3 reruns on the same data

#### Task 07-05 [tier B, deps: 07-04]
**Implement DeepEval CI gates** in `tests/eval/test_smoke_gates.py`. Wire into PR CI.
- **Acceptance**: a deliberately-bad PR (e.g., revert ColQwen → MiniLM) fails the gate

#### Task 07-06 [tier B, deps: 07-02]
**Implement `src/eval/datasets/synth.py`** — generate the 1000-question comprehensive set per §10.2.3.
- **Acceptance**: generated set passes a sanity LLM-judge gate (≥ 80% answerable from corpus); committed for reuse

#### Task 07-07 [tier B, deps: 07-06]
**Build the regression set incrementally** by ingesting the first batch of 50 production failures + 50 hand-authored cases.
- **Acceptance**: `eval/regression.jsonl` exists with 100 entries

### 14.9 Phase 8 — Observability

#### Task 08-01 [tier B, deps: 06-01]
**Wire Langfuse tracing** in the gateway and orchestrator.
- **Acceptance**: every query produces a trace visible in Langfuse with all spans listed in §13.1

#### Task 08-02 [tier B, deps: 08-01]
**Wire OpenTelemetry metrics** to Prometheus.
- **Acceptance**: dashboards in Grafana show non-zero values for the metrics in §13.2

#### Task 08-03 [tier C, deps: 08-02]
**Author Grafana dashboards** (Operator, Quality, Capacity).
- **Acceptance**: dashboards JSON committed to `infra/grafana/`; render correctly with sample data

#### Task 08-04 [tier B, deps: 08-02]
**Wire alerting rules** per §13.3 in Prometheus AlertManager.
- **Acceptance**: simulated 5xx spike triggers an alert in staging

### 14.10 Phase 9 — Deployment

#### Task 09-01 [tier B, deps: 03-03, 06-01]
**Author Dockerfiles** per §11.4 with the three build targets.
- **Acceptance**: `docker build --target embed` succeeds; image runs the embed service

#### Task 09-02 [tier B, deps: 09-01]
**Provision Lightning Studios** per §11.3.
- **Acceptance**: `lightning serve` deploys `urban-rag-embed` and it answers `/health`

#### Task 09-03 [tier B, deps: 09-02]
**Provision Qdrant Cloud** + create v1 collections.
- **Acceptance**: collections exist; smoke embed → upsert → query path works against cloud

#### Task 09-04 [tier B, deps: 06-03]
**Deploy `web/` to Vercel.** Configure Cloudflare in front.
- **Acceptance**: prod URL serves the app; TLS valid; analytics events flow

#### Task 09-05 [tier B, deps: 09-02, 09-03]
**Wire CI/CD** per §11.8: build, push, deploy on merge to `main`.
- **Acceptance**: a no-op PR merged to main triggers a successful prod deploy

#### Task 09-06 [tier B, deps: 09-05]
**Configure rollback playbook** per §11.9 in `docs/runbooks/rollback.md`.
- **Acceptance**: dry-run rollback in staging completes in ≤ 5 minutes

### 14.11 Phase 10 — Corpus Expansion

The v1 corpus has 738 pages from URDPFI + SWM. The product target is "hundreds of urban planning data" — Tanmay said hundreds of *documents*. Phase 10 grows it.

#### Task 10-01 [tier A, no deps]
**Compile the corpus expansion plan.** Source list to add:
- NBC 2016 Vol I & II (re-attempt; the v1 indexing failed)
- All state Master Plans for top-15 Indian cities (MPD 2021, Mumbai DCR, Bengaluru RMP, Chennai SMP, Hyderabad HMDA, Pune DP, Ahmedabad CDP, Kolkata DMP, Jaipur Master Plan 2025, Lucknow Master Plan 2031, etc.)
- IRC codes relevant to urban roads (IRC:106, IRC:103, IRC:99, IRC:86)
- ISI standards (IS 875 series — design loads; IS 1893 — seismic)
- TCPO model regulations
- MoHUA AMRUT, Smart Cities Mission documents
- State/UT building bye-laws (Haryana BBL, UP Bhawan Nirman Vidhi, etc.)
- Climate-resilient planning guidelines (NDMA, IMD reports)
- World Bank / ADB urban India reports (where licensing permits)
- Output: `docs/corpus_v2_plan.md` listing 200+ specific documents with source URLs and licensing notes

#### Task 10-02 [tier B, deps: 10-01]
**Source acquisition automation** in `scripts/corpus/fetch.py` — script that, given a manifest, downloads PDFs to `s3://urban-rag-source/`, dedupes by SHA256, logs licensing.
- **Acceptance**: 50 PDFs successfully fetched & stored in a dry run; manifest persisted

#### Task 10-03 [tier B, deps: 10-02, 02-05]
**Run ingestion on first 50 docs.** Inspect parser output, fix per-document parser issues.
- **Acceptance**: 50 docs through full ingest pipeline; ≥ 90% page-level success

#### Task 10-04 [tier B, deps: 10-03, 03-05]
**Index expanded corpus.** Bump corpus version. Publish.
- **Acceptance**: Qdrant v1.1.0 collections live; smoke set still passes; comprehensive set re-run with delta report

#### Task 10-05 [tier B, deps: 10-04]
**Iterate to 200+ docs.** Continue acquisition + ingestion in batches of 25 with eval-after-each-batch.
- **Acceptance**: corpus crosses 200 documents; ≥ 50,000 pages; recall@10 holds ≥ 0.85

### 14.12 Phase 11 — Quality & Iteration (continuous)

These are recurring tasks the agent fleet runs forever, not one-shot deliveries.

- **Weekly**: run comprehensive eval; publish dashboard; open tickets for any regression > 2 points
- **Weekly**: triage feedback; promote 5–10 negative-feedback cases into the regression set
- **Weekly**: cost report; flag any component over budget
- **Bi-weekly**: prompt review; A/B any candidate improvements against current production prompts
- **Monthly**: model swap evaluation — re-benchmark current ViDoRe leaderboard top-3 against our smoke + comprehensive sets; if a candidate wins by ≥ 2 points NDCG@10 with similar cost, schedule a swap
- **Monthly**: dependency upgrade sweep
- **Quarterly**: human calibration of the LLM judge

### 14.13 Phase 12 — Stretch / Research (post-v1)

Listed for completeness; not gating v1 launch.

- Hindi/regional language query support (translate inbound query, answer in source language)
- Multi-document comparison mode (URDPFI v1 vs v2 side-by-side)
- Computational answer mode (FAR×plot=built-up area calculator with grounded constants)
- Document upload for ad-hoc indexing (with abuse / licensing controls)
- Voice query (Whisper → text → existing pipeline)
- Mobile-native client
- Public API with rate limits + auth keys

### 14.14 Roadmap dependency graph (high-level)

```
00-01 ── 00-02 ─┐
                ├─→ 00-04 ─→ 00-05 ─→ 01-* ─→ 02-* ─→ 03-* ─→ 04-* ─→ 05-* ─→ 06-* ─┐
00-03 ──────────┘                                                                    │
                                                                                     ▼
                                                                          07-* (eval gates)
                                                                                     │
                                                                                     ▼
                                                                          08-* (observability)
                                                                                     │
                                                                                     ▼
                                                                          09-* (deployment)
                                                                                     │
                                                                                     ▼
                                                                          10-* (corpus growth)
                                                                                     │
                                                                                     ▼
                                                                          11-* (continuous)
```

### 14.15 Agent etiquette

When running this roadmap autonomously:

1. **One task per branch.** Never bundle.
2. **Read PLAN.md and the deps tasks before starting.** Don't re-derive.
3. **Update the task status** in `docs/roadmap_status.md` (a tracking file added in 00-04). Tracker is markdown checkboxes per phase.
4. **If blocked**, write to `docs/blocked.md` with the blocker, what was tried, and what's needed. Do not invent workarounds that contradict ADRs.
5. **Tests first** for `src/common`, `src/eval`, `src/retrieve.fusion` — these are correctness-critical. For UI / glue code, tests can come second but must come.
6. **Run smoke eval before merging** — don't trust CI alone if you can run it locally.
7. **Cite this PLAN.md** in PR descriptions: `Implements PLAN.md §14.7 Task 07-01`. This makes the audit trail trivial.

---

## Appendix A — Dead Code Removal Manifest

The following files and directories are removed as part of the reset (Task 00-03). The audit in Part I established why each is dead, broken, speculative, or duplicative. Agents executing this manifest should remove in independent PRs grouped by cluster.

### A.1 Cluster: Top-level scratch CLIs

- `cli.py` — v3 CLI calling `src/rag.py`; superseded by `src/api/main.py` + `web/`
- `query_simple.py` — CPU MiniLM + mean-pool path; lossy and dimension-mismatched
- `query_vision.py` — random-sample-to-Gemini "RAG"; not retrieval at all

### A.2 Cluster: Outdated requirements + speculative docs

- `requirements.txt` — frozen pip; replaced by `pyproject.toml` + `uv.lock`
- `requirements-hybrid.txt` — frozen pip for the hybrid spec that was never validated
- `requirements-lightning.txt` — frozen pip for a Lightning experiment that fizzled
- `HYBRID_RAG_V4.md` — speculative design doc; superseded by this PLAN

### A.3 Cluster: Broken `src/` shims

- `src/api.py` — broken; query vector hardcoded to `embs[0,0,:]`
- `src/query.py` — same broken pattern
- `src/query_encoder.py` — MiniLM 384d vs ColQwen 320d dimension mismatch
- `src/indexer_optimized.py` — Chroma-based; superseded by Qdrant indexer
- `src/rag.py` — v3 monolith; functionality fragmented into `src/retrieve/`, `src/generate/`

### A.4 Cluster: Hybrid scaffold (1500 lines of un-validated spec)

- `src/hybrid_rag/` — entire directory:
  - `__init__.py`, `qdrant_store.py`, `pageindex_integration.py`, `fusion_reranker.py`, `knowledge_graph.py`, and any sibling files
  - Re-implement only what we actually want, in the new layout, with tests
- Note: the Knowledge Graph idea may resurface in a v2 stretch; if so, it gets a fresh PR with proper design — the scaffold itself does not survive

### A.5 Cluster: Stale scripts

- `scripts/archive/` — all archived experiments
- `scripts/send_emails.py` — unrelated to RAG; appears to be cross-contaminated from a different project
- `scripts/track_leads.py` — same; unrelated
- `scripts/index_lancedb.py` — never wired; LanceDB is staged for v1.5, not v1
- `scripts/migrate_to_v4.py` — migration to a v4 that doesn't exist
- `scripts/build_index.py` — superseded by `src/index/batch.py`

### A.6 Kept (do not remove)

- `scripts/embed.py` — sound v3 embedding script; salvageable; will be ported to `src/embed/` and then this file removed
- `scripts/pipeline.py` — useful orchestrator shape for local dev; keep until `src/ops/` replaces it
- `data/page_images/` — golden eval set; keep
- `data/embeddings/` — useful for offline backtesting until v1 index exists

### A.7 Removal procedure per cluster

1. Open PR titled `chore: remove dead code — <cluster name>`
2. Body: link to this appendix; list files
3. Single commit; squash-merge
4. Confirm CI green (smoke eval gates may not exist yet — that's fine until Phase 7)

---

## Appendix B — Document & Data Schemas

This appendix is the canonical schema reference. Every type is a pydantic v2 model in `src/common/types.py`.

### B.1 Source documents

```python
class DocumentRecord(BaseModel):
    doc_id: str                       # SHA256 of file content, hex
    filename: str                     # original filename, sanitized
    title: str                        # human-friendly title
    family: Literal["NBC","URDPFI","SWM","IRC","IS","MASTER_PLAN","BBL","OTHER"]
    jurisdiction: str | None          # "IN", "IN-DL", "IN-MH-Mumbai", etc. (ISO 3166-2)
    publisher: str | None             # "BIS", "MoHUA", "ULB-Mumbai"
    year: int | None
    version: str | None               # "v1", "v2", "2016", "2024"
    license: Literal["public_domain","gov_open","fair_use_research","unknown"]
    page_count: int
    sha256: str
    storage_uri: str                  # s3://urban-rag-source/<sha>.pdf
    ingested_at: datetime
```

### B.2 Pages

```python
class PageRecord(BaseModel):
    page_id: str                      # f"{doc_id}#p{page_num:04d}"
    doc_id: str
    page_num: int                     # 1-indexed
    page_type: Literal["TEXT","VISUAL","BLANK"]
    dpi_used: int                     # 100 or 250
    image_uri: str                    # s3://urban-rag-pages/<doc_id>/<page>.png
    extracted_text: str               # full page text (OCR'd or parsed)
    layout: list[LayoutBlock]         # structured blocks
    section_id: str | None            # foreign key to SectionRecord
    section_title: str | None
```

### B.3 Layout blocks

```python
class LayoutBlock(BaseModel):
    block_id: str
    page_id: str
    type: Literal["heading","paragraph","table","figure","caption","list","footnote"]
    text: str | None                  # text content, if any
    bbox: tuple[float,float,float,float]  # (x0,y0,x1,y1) in image coords
    confidence: float                 # parser confidence 0-1
```

### B.4 Sections

```python
class SectionRecord(BaseModel):
    section_id: str                   # f"{doc_id}#s{idx:03d}"
    doc_id: str
    title: str
    level: int                        # 1=chapter, 2=section, 3=subsection
    parent_section_id: str | None
    start_page: int
    end_page: int
    title_path: list[str]             # ["Chapter 4","4.3 Land Use","4.3.1 Residential"]
```

### B.5 Chunks (text channel)

```python
class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    page_id: str
    section_id: str | None
    text: str                         # 256-512 token chunk
    token_count: int
    embedding_model: str              # "Alibaba-NLP/gte-modernbert-colbert"
    chunk_index_in_section: int
```

### B.6 Retrieval candidates

```python
class RetrievalCandidate(BaseModel):
    page_id: str
    score: float                      # final fused score
    channel_scores: dict[str, float]  # {"visual": 12.3, "text": 0.81, "sparse": 4.2}
    channel_ranks: dict[str, int]     # {"visual": 1, "text": 4, "sparse": 12}
    rerank_score: float | None        # populated after VLM rerank
    rerank_rationale: str | None
    page_image_uri: str
    extracted_text_excerpt: str       # first 500 chars of page text
    section_title: str | None
```

### B.7 Retrieval result

```python
class RetrievalResult(BaseModel):
    query: str
    expanded_queries: list[str]
    candidates: list[RetrievalCandidate]
    latency_ms: int
    flags: dict[str, bool]            # degraded_mode, vlm_rerank_skipped, etc.
    retrieval_strategy: str           # "visual_primary" | "text_primary" | "hybrid"
```

### B.8 Citations & answers

```python
class Citation(BaseModel):
    idx: int                          # [1], [2], ...
    doc_id: str
    page_id: str
    page_num: int
    doc_title: str
    section_title: str | None
    score: float

class Answer(BaseModel):
    query_id: str
    text: str                         # markdown, includes [k] markers
    citations: list[Citation]
    refused: bool                     # true if model declined to answer
    refusal_reason: str | None
    model: str                        # "gemini-2.5-flash"
    prompt_template_id: str
    prompt_template_version: int
    completed_at: datetime
    latency_ms: int
    cost_usd: float
```

### B.9 Trace (for replay)

```python
class Trace(BaseModel):
    query_id: str
    user_hash: str | None
    mode: Literal["fast","deep"]
    corpus_version: str
    embed_model: str
    rerank_model: str
    gen_model: str
    retrieval_result: RetrievalResult
    answer: Answer
    spans: list[TraceSpan]
    started_at: datetime
    finished_at: datetime
```

### B.10 Postgres tables

| Table         | Purpose                                                     |
|---------------|-------------------------------------------------------------|
| `documents`   | one row per `DocumentRecord`                                |
| `pages`       | one row per `PageRecord` (sans embedding — that's in Qdrant)|
| `sections`    | one row per `SectionRecord`                                 |
| `chunks`      | one row per `ChunkRecord`                                   |
| `queries`     | one row per inbound query (with hashed user)                |
| `traces`      | one row per `Trace` (most data in Langfuse; this is the lookup)|
| `feedback`    | one row per user feedback submission                        |
| `eval_runs`   | one row per eval run with aggregate metrics                 |
| `audit_log`   | append-only ops audit                                       |

Schemas are defined in `src/common/db/schema.sql` and migrated with `alembic`.

---

## Appendix C — API Spec

The HTTP API contract exposed by the FastAPI gateway. OpenAPI is auto-generated by FastAPI and published at `/openapi.json`. This appendix is the human-readable reference and the CI conformance source.

### C.1 Conventions

- All endpoints under `/v1/`. Breaking changes require `/v2/`.
- JSON request and response bodies; SSE for streaming.
- Auth: v1 — none, behind Cloudflare allowlist; v2 — bearer JWT.
- Rate limits: 60 queries / minute / IP; 1000 queries / day / IP. Returned via `X-RateLimit-*` headers.
- All times in UTC ISO 8601.
- Errors: `{"error": {"code": "...", "message": "...", "trace_id": "..."}}` with appropriate HTTP status.

### C.2 `POST /v1/ask`

Submit a question; returns a `query_id` for streaming retrieval.

**Request body:**
```json
{
  "question": "What is FAR for residential zones in URDPFI?",
  "mode": "fast"
}
```

- `question`: 1–1000 chars, required
- `mode`: `"fast" | "deep"`, default `"fast"`. `deep` switches generator to Gemini 2.5 Pro and ups rerank top-K.

**Response 202 Accepted:**
```json
{
  "query_id": "q_01HQ4...",
  "stream_url": "/v1/ask/q_01HQ4.../stream",
  "expires_at": "2026-04-26T10:15:00Z"
}
```

### C.3 `GET /v1/ask/{query_id}/stream`

Server-Sent Events stream of retrieval + generation events.

**Event types (in order):**

```
event: retrieval_started
data: {"query_id":"q_01HQ4...","ts":"..."}

event: retrieval_completed
data: {"candidates":[...10 items...],"latency_ms":480}

event: generation_started
data: {"model":"gemini-2.5-flash","ts":"..."}

event: token
data: {"text":"FAR"}

event: token
data: {"text":" stands"}
... many ...

event: generation_completed
data: {"answer":{...full Answer...},"latency_ms":2900,"cost_usd":0.0021}

event: done
data: {"query_id":"q_01HQ4..."}
```

**Failure events:**

```
event: error
data: {"code":"retrieval_timeout","message":"...","stage":"retrieval"}

event: refused
data: {"reason":"out_of_corpus","message":"This is not covered by the indexed documents."}
```

### C.4 `GET /v1/ask/{query_id}`

Fetch the completed answer (after stream finishes). Used for permalinks / reload.

**Response 200:**
```json
{
  "query_id": "q_01HQ4...",
  "question": "...",
  "answer": { ...Answer... },
  "retrieval": { ...RetrievalResult-without-embeddings... },
  "trace_url": "https://langfuse.../trace/..."
}
```

### C.5 `POST /v1/feedback`

Submit thumbs-up/down on an answer.

**Request:**
```json
{
  "query_id": "q_01HQ4...",
  "vote": "up",
  "comment": "Good but missed the v2 update"
}
```

- `vote`: `"up" | "down"`, required
- `comment`: ≤ 200 chars, optional

**Response 204 No Content.**

### C.6 `GET /v1/corpus`

List indexed documents.

**Response 200:**
```json
{
  "corpus_version": "v1.0.0",
  "indexed_at": "2026-04-25T12:00:00Z",
  "documents": [
    {
      "doc_id": "abc...",
      "title": "URDPFI Guidelines Volume I, 2014",
      "family": "URDPFI",
      "jurisdiction": "IN",
      "page_count": 447,
      "version": "v1"
    }
  ],
  "totals": {
    "documents": 47,
    "pages": 12834
  }
}
```

### C.7 `GET /v1/corpus/{doc_id}/pages/{page_num}/image`

Returns a presigned S3 URL for the page PNG (302 redirect) or the bytes directly. Used by the citation lightbox.

### C.8 `GET /v1/healthz`

```json
{"status":"ok","version":"1.0.0","corpus_version":"v1.0.0"}
```

### C.9 `GET /v1/readyz`

200 only when:
- Embedding service is reachable
- Qdrant is reachable
- Last 10 queries had < 50% error rate

### C.10 Error code reference

| Code                   | HTTP | Meaning                                                         |
|------------------------|------|-----------------------------------------------------------------|
| `validation_error`     | 422  | Request body schema violation                                   |
| `rate_limited`         | 429  | Too many queries                                                |
| `retrieval_timeout`    | 504  | Retrieval exceeded budget; degraded fallback also failed       |
| `generation_failed`    | 502  | VLM call failed and fallback unavailable                        |
| `out_of_corpus`        | 200  | Returned via `refused` SSE event, not HTTP error               |
| `corpus_unavailable`   | 503  | Qdrant or embed service down; retry-after header set            |
| `internal`             | 500  | Anything else; trace_id required                                |

### C.11 OpenAPI conformance test

`tests/api/test_openapi.py` validates that the running server's `/openapi.json` matches the committed `docs/openapi.yaml` (modulo schema additions). PR fails if the contract drifts unintentionally.

---

## Appendix D — Cost & Capacity Model

A back-of-envelope cost model. Numbers should be re-validated after Phase 9.

### D.1 Assumptions

- Corpus v1: 50,000 pages
- Corpus v2 (target): 200,000 pages
- Daily query volume v1: 200 (low; mostly Tanmay + a few invitees)
- Daily query volume v2: 2,000 (modest public release)
- Average pages per query (after retrieval+rerank): 5
- Average prompt tokens / query (Gemini 2.5 Flash): 4,500 (5 page images + question + system)
- Average output tokens / query: 350

### D.2 Compute costs

| Component               | v1 daily | v1 monthly | v2 daily | v2 monthly |
|-------------------------|----------|------------|----------|------------|
| Embed service (A10G, scale 0–2) | ~3h | $80 | ~12h | $320 |
| Batch indexing (H100, occasional) | — | $25/index | — | $50/index |
| Qdrant Cloud | — | $50 | — | $200 |
| Gemini 2.5 Flash (rerank+gen) | — | $20 | — | $200 |
| S3 storage (PNGs + source) | — | $15 | — | $50 |
| Postgres (Neon) | — | $20 | — | $50 |
| Vercel + Cloudflare | — | $0 | — | $40 |
| **Total**               |          | **~$210** |          | **~$910** |

### D.3 Per-query cost breakdown (v1, fast mode)

- Embedding (query encode, amortised on warm replica): $0.0002
- Qdrant query: $0.0001 (effectively free under flat-rate plan)
- VLM rerank (Gemini 2.5 Flash, 5 images, ~3000 tokens prompt, 200 output): $0.0011
- Generation (Gemini 2.5 Flash, 5 images, ~4500 tokens prompt, 350 output): $0.0021
- Logging + storage: negligible
- **Total: ~$0.0035 / query**

Deep mode (Gemini 2.5 Pro): ~$0.025 / query.

### D.4 Capacity

- Embed A10G handles ~80 query encodes/sec at bf16; one replica covers our v2 daily volume comfortably
- Qdrant Cloud free 1GB tier covers up to ~150k pages with quantization; v2 needs the $50 plan
- Gemini Flash quota: project-default 60 RPM is sufficient to v2; monitor and request raise if needed
- Gateway CPU: 2 vCPU / 2GB autoscale 0–4; CPU-bound only on JSON serialization

### D.5 Where cost can blow up

1. **Deep-mode usage**: a 10x cost multiplier vs fast-mode; cap deep mode to 20% of queries via UI affordance + server-side governor
2. **Reranking large candidate sets**: O(N) on N candidates; cap N at 30 in production
3. **Image attachments to Gemini**: each page image is ~1500 tokens at the inline-base64 level; if we re-attach 20 pages instead of 5, cost quadruples
4. **Re-indexing churn**: every full reindex is $25–$50; batch corpus updates rather than per-doc reindexing

### D.6 Cost alerting

- Daily cost report (cron at 09:00 IST) → Discord
- Alert if `cost_per_query 24h-rolling > $0.05` (P2 in §13.3)

---

## Appendix E — Risk Register

Risks ordered by expected impact × probability. Each has an owner (initial: Tanmay) and a mitigation plan.

### E.1 Technical risks

#### R-T1 — VLM hallucinated citations
- **Impact**: high (destroys trust)
- **Probability**: medium
- **Mitigation**: structured citation extraction with regex validation; refusal prompt on insufficient context; faithfulness gate at 0.90 in CI; human spot-check 50 queries / week
- **Detection**: faithfulness < 0.85 in any rolling 1h window

#### R-T2 — ColQwen 320d capacity ceiling at 200k pages
- **Impact**: medium
- **Probability**: medium
- **Mitigation**: monitor recall@10 vs corpus growth; switch to ColNomic 7B (768d) at v1.5 if needed; Qdrant scalar quantization absorbs the storage hit
- **Detection**: recall@10 drop > 3 points at any corpus expansion checkpoint

#### R-T3 — Gemini API throttling
- **Impact**: medium
- **Probability**: low–medium
- **Mitigation**: self-host Qwen2.5-VL-72B fallback path (Task 05-04); request quota raise proactively; circuit-break to text-only mode on 429
- **Detection**: 429 rate > 1% over 10 min

#### R-T4 — Parser failures on scanned/poor-quality PDFs
- **Impact**: medium (silent quality loss)
- **Probability**: high (inevitable with state docs)
- **Mitigation**: Docling primary + Marker fallback + manual OCR remediation queue; per-document parse-success metric tracked
- **Detection**: parse success rate < 90% on any new document batch

#### R-T5 — Index drift (corpus updated, eval set stale)
- **Impact**: medium (eval lies)
- **Probability**: high if not managed
- **Mitigation**: every corpus version bump triggers a regression eval re-validation; expected-pages in eval set use stable doc_id+section_id rather than absolute page numbers
- **Detection**: regression eval recall@10 plummets after corpus version bump

#### R-T6 — Streaming endpoint instability under load
- **Impact**: medium
- **Probability**: low
- **Mitigation**: load-test gateway with k6 to 50 RPS before public launch; SSE reconnect logic on client; idempotent stream consumption
- **Detection**: stream_disconnect rate > 5%

### E.2 Operational risks

#### R-O1 — Secrets re-leak
- **Impact**: high
- **Probability**: medium without tooling
- **Mitigation**: `gitleaks` pre-commit hook; CI secret scanning; weekly key rotation cron; checked-in `.env` is forbidden by `.gitignore` rule
- **Detection**: `gitleaks` finding in any PR

#### R-O2 — Lightning AI account-level outage
- **Impact**: high
- **Probability**: low
- **Mitigation**: documented failover playbook to Modal or RunPod; embedding service is the only critical Lightning-resident piece
- **Detection**: Lightning health-check failures > 5 min

#### R-O3 — Qdrant Cloud regional outage
- **Impact**: high
- **Probability**: low
- **Mitigation**: hourly snapshots + daily exports to S3; documented restore-to-Modal procedure; read-only fallback to last good snapshot via embedded Qdrant in gateway pod
- **Detection**: Qdrant API 5xx > 5 min

#### R-O4 — Cost overrun (Gemini, Lightning)
- **Impact**: medium
- **Probability**: medium
- **Mitigation**: daily cost cron; per-query cost cap; deep-mode rate-limit; spending alarms at provider level
- **Detection**: monthly burn > 1.5× plan

### E.3 Legal / licensing risks

#### R-L1 — Document license ambiguity
- **Impact**: medium
- **Probability**: medium
- **Mitigation**: every ingested document is tagged with one of `public_domain | gov_open | fair_use_research | unknown`; `unknown` documents are excluded from production until cleared; legal review of corpus expansion plan in Task 10-01
- **Detection**: any corpus inclusion of `unknown`-licensed doc

#### R-L2 — User-submitted query contains PII
- **Impact**: low–medium (privacy)
- **Probability**: medium
- **Mitigation**: PII detection + masking pre-log (§13.6); 30-day raw retention only
- **Detection**: PII hits in audit sample

#### R-L3 — Output reproduces large verbatim passages from copyrighted material
- **Impact**: medium
- **Probability**: low (we cite, but a long literal quote is risky)
- **Mitigation**: prompt instructs paraphrase + citation, not verbatim reproduction; max contiguous quote length cap of 200 chars enforced by post-processor
- **Detection**: post-processor flags quotes > 200 chars

### E.4 Quality risks

#### R-Q1 — Eval set leakage into training (we don't train, but prompts can leak)
- **Impact**: low (we don't fine-tune in v1)
- **Probability**: low
- **Mitigation**: eval set never sent to model providers in any "improvement" or "feedback" mechanism; document this in `docs/eval_isolation.md`
- **Detection**: any prompt that includes `eval/` content outside an eval run

#### R-Q2 — LLM-as-judge calibration drift
- **Impact**: medium
- **Probability**: medium
- **Mitigation**: monthly human re-anchoring; pinned judge model version; alert if judge agreement with human anchor drops below 0.75 κ
- **Detection**: agreement drop in monthly check

### E.5 Strategic risks

#### R-S1 — Field moves on (a new SOTA visual RAG model lands, ours becomes obsolete)
- **Impact**: medium
- **Probability**: high (this field moves fast)
- **Mitigation**: monthly model swap evaluation (§14.12); architecture is decoupled from any specific model so swap cost is bounded
- **Detection**: ViDoRe leaderboard delta vs ours > 5 points NDCG@10

#### R-S2 — Project loses focus (scope creep into chat/conversational/multi-tenant)
- **Impact**: medium
- **Probability**: medium
- **Mitigation**: this PLAN.md as the durable contract; out-of-scope sections list (§9.8) is treated as binding; new ideas go in Phase 12 (stretch)
- **Detection**: PRs landing features outside Phase 1–11

---

## Appendix F — Glossary & References

### F.1 Glossary

- **AGREE**: Adaptive GeneRation-Enhanced Embedding; ColQwen2.5-based retriever with +7% lift
- **ColPali / ColQwen / ColNomic**: late-interaction multivector visual embedders; "Col" = column-wise; the "Col" prefix denotes ColBERT-style multivector encoding
- **Docling**: IBM's open-source document parser; replaces older PyMuPDF-only flows
- **DPI**: dots per inch; PNG render resolution from PDF
- **FAR / FSI**: Floor Area Ratio / Floor Space Index; same concept, different abbreviation
- **HyDE**: Hypothetical Document Embeddings; retrieval-time query expansion
- **Late interaction**: token-level (or patch-level) similarity scoring at retrieval time, vs early interaction (cross-encoder) which is too slow at scale
- **MaxSim**: aggregation function for late-interaction multivector retrieval; for each query token, max cosine over doc tokens; sum across query tokens
- **NBC**: National Building Code of India (BIS); the building-design bible
- **NDCG**: Normalized Discounted Cumulative Gain; ranking quality metric
- **PageIndex**: tree-based reasoning retrieval — explore-as-you-go vs flat retrieve-then-read
- **RAGAS**: open-source RAG evaluation suite
- **RRF**: Reciprocal Rank Fusion; standard cross-channel rank combiner with formula `score = Σ 1/(k + rank_i)`, k typically 60
- **SOTA**: state-of-the-art
- **SSE**: Server-Sent Events; HTTP streaming primitive used here for token streaming
- **SWM Rules**: Solid Waste Management Rules 2016, India
- **TruLens**: production observability tool with built-in RAG evals
- **URDPFI**: Urban and Regional Development Plans Formulation and Implementation guidelines (MoHUA)
- **ViDoRe**: Visual Document Retrieval Benchmark; current public leaderboard for visual RAG
- **VLM**: Vision-Language Model; here Gemini 2.5 Flash / Pro and Qwen2.5-VL

### F.2 References

#### Models
- ColPali (Faysse et al., 2024): https://arxiv.org/abs/2407.01449
- ColQwen2 / ColQwen2.5 (Hugging Face vidore/colqwen2.5-v0.2): https://huggingface.co/vidore/colqwen2.5-v0.2
- ColNomic-7B-V1.5 (nomic-ai/colnomic-embed-7b-v1.5): https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b
- Nemotron Multimodal ColEmbed V2-8B (NVIDIA): documented on ViDoRe v3 leaderboard
- AGREEQwen2.5: published evaluation results referenced in ViDoRe; check the original repository for runtime
- GTE-ModernColBERT (Alibaba-NLP/gte-modernbert-colbert): https://huggingface.co/Alibaba-NLP/gte-modernbert-colbert
- Gemini 2.5 Flash / Pro: https://ai.google.dev/gemini-api/docs/models
- Qwen2.5-VL-72B-Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct

#### Tooling
- Docling: https://github.com/DS4SD/docling
- Marker: https://github.com/VikParuchuri/marker
- Qdrant: https://qdrant.tech ; multivector docs https://qdrant.tech/documentation/concepts/vectors/#multivectors
- LanceDB: https://lancedb.com
- RAGAS: https://github.com/explodinggradients/ragas
- DeepEval: https://github.com/confident-ai/deepeval
- TruLens: https://www.trulens.org
- Langfuse: https://langfuse.com
- promptfoo: https://github.com/promptfoo/promptfoo
- LitServe: https://github.com/Lightning-AI/LitServe
- Lightning AI: https://lightning.ai

#### Benchmarks & datasets
- ViDoRe v1, v2, v3 leaderboards: https://huggingface.co/spaces/vidore/vidore-leaderboard

#### Methodology papers (selected)
- Reciprocal Rank Fusion (Cormack et al., SIGIR 2009): canonical RRF
- HyDE (Gao et al., 2022): https://arxiv.org/abs/2212.10496
- CRAG (Yan et al., 2024): Corrective RAG
- Self-RAG (Asai et al., 2023): https://arxiv.org/abs/2310.11511
- RAGAS (Es et al., 2023): https://arxiv.org/abs/2309.15217

#### Indian urban planning sources
- URDPFI Guidelines Vol I & II (MoHUA, 2014; revised 2015): public
- National Building Code of India 2016 (BIS, two volumes): public
- Solid Waste Management Rules 2016 (MoEFCC): public
- Master Plans of Indian cities: licensing varies; gov_open in most cases
- IRC codes: Indian Roads Congress; some public, some restricted
- IS standards: Bureau of Indian Standards; mostly paid; restricted use

### F.3 Document conventions

- Code blocks are illustrative; the real implementation may diverge in non-essential ways (variable names, helper splits) but must preserve the externally-visible behavior described
- All "v1.x" references in this PLAN are corpus / system version, not RAG model version
- "Agent" without qualifier refers to the autonomous-execution agent fleet running this roadmap, not the user-facing product

---

## Closing

This is a working document. Sections will be edited as facts on the ground change — agents discover better tools, ViDoRe leaderboards shift, the corpus reveals quirks, users surface failure modes. The structure is durable; the contents are not sacred. Every change to this PLAN goes through the same PR process as code, with the same review bar.

Tanmay's brief was: revive urban-rag, reach SOTA visual RAG, support hundreds of documents, simple web interface, agent-executable. This document is the contract that makes that brief executable. The Phase 0 hygiene tasks unblock everything else. Start there.

— Drafted by the Claude Code lane on the secondary frontier seat, anchored to the Hermes shared memory, intended for hand-off to the autonomous agent fleet.
