# Urban Planning RAG 🏙️

**Visual RAG system for Indian urban planning regulations**

A production-grade multimodal retrieval system that indexes planning documents (NBC, URDPFI, SWM Rules) as page images, embeds them with ColQwen2.5 visual encoders, and answers questions via Gemini 2.5 Flash with precise page citations.

---

## What This Does

Ask questions like:
- *"What is the FSI for residential zones?"*
- *"What are the parking requirements for commercial buildings?"*
- *"What are the indicators of good governance?"*

The system:
1. **Retrieves** relevant pages using 3-channel parallel search (visual + text + sparse BM25)
2. **Fuses** results with Reciprocal Rank Fusion (RRF k=60)
3. **Reranks** with Gemini 2.5 Flash VLM cross-encoder
4. **Generates** streaming answers with confidence levels and inline page citations

---

## Architecture

### Embedding
- **Visual**: `vidore/colqwen2.5-v0.2` — 128-dim multi-vector per patch (ColPali-style late interaction)
- **Text**: `lightonai/GTE-ModernColBERT-v1` — 768-dim multi-vector per token (ModernBERT late interaction)

### Vector DB
- **Qdrant** with two collections:
  - `pages_visual` — ColQwen2.5 multi-vector with MAX_SIM comparator + INT8 scalar quantization
  - `pages_text` — GTE-ModernColBERT multi-vector (text + BM25 sparse)

### Retrieval Pipeline
```
Query → 3-channel parallel search (visual/text/sparse BM25)
      → RRF k=60 fusion (top-20 per channel → top-20 fused)
      → Gemini 2.5 Flash VLM rerank (top-20 → top-5)
```

- **Visual**: ColQwen2.5 query encoding → Qdrant ANN (200 pooled) → MaxSim rerank → top 20
- **Text**: GTE-ModernColBERT query encoding → Qdrant ANN (200 pooled) → MaxSim rerank → top 20
- **Sparse**: BM25 query scoring → Qdrant sparse vector search → top 20
- **Fusion**: RRF k=60 across all three channels
- **Rerank**: Gemini 2.5 Flash cross-encoder scores page images → top 5

### Generation
- **Streaming Gemini 2.5 Flash** client with SSE events
- Confidence levels per claim, inline `[k]` citation markers
- Grounded in retrieved page images sent alongside the query

### API
- **FastAPI** gateway on port 3100
- SSE streaming for query responses
- `/v1/query` — streaming query endpoint
- `/v1/healthz` — health check
- `/metrics` — Prometheus metrics

### Web
- **Next.js 14 App Router** (`web/` directory)
- Streaming query via Server-Sent Events
- Citation lightbox (click `[k]` to view the source page)
- Thumbs up/down feedback buttons

### Observability
- **Langfuse** tracing (per-query spans, OTel-compatible)
- **Prometheus** metrics (`qdrant_latency_seconds`, `gemini_cost_usd_total`, `faithfulness_p50`)
- **Grafana** dashboards (see `infra/grafana/`)

### Evaluation
- `eval/smoke.jsonl` — 25 hand-curated questions (CI gate on every PR)
- `eval/regression.jsonl` — 106 questions including adversarial probes
- **RAGAS** metrics: faithfulness, answer_relevance, context_precision, context_recall, answer_correctness
- CI gates: recall@10 ≥ 0.85, faithfulness ≥ 0.85

---

## Corpus Status

| Stat | Value |
|------|-------|
| Documents indexed | 8 |
| Total pages | 743 PNG renders |
| Corpus ready to scale | Yes — 200+ documents queued once source URLs are resolved |

---

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in required values:

```bash
# Google Gemini API (required)
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant vector database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here   # optional for local

# Langfuse tracing (optional)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

See `.env.example` for the full template.

### 3. Ingest documents

```bash
# Ingest a single PDF
python -m urban_rag.cli ingest ./path/to/document.pdf

# Ingest all PDFs in a directory
python -m urban_rag.cli ingest ./pdfs/

# Corpus management
python -m urban_rag.cli corpus list
python -m urban_rag.cli corpus stats
```

### 4. Query the corpus

```bash
# Streaming query via CLI
python -m urban_rag.cli query "What is FSI for residential zones?"

# Retrieve-only mode (no generation)
python -m urban_rag.cli query "parking requirements" --retrieve-only

# Control top-k candidates
python -m urban_rag.cli query "open space standards" --top-k 10
```

### 5. Run services

```bash
# Start the FastAPI gateway (port 3100)
uvicorn urban_rag.api.main:app --host 0.0.0.0 --port 3100

# Start the embed service (port 3102)
uvicorn urban_rag.embed.serve:app --host 0.0.0.0 --port 3102

# Start Qdrant (Docker)
docker run -d --name urban-rag-qdrant -p 3103:6333 qdrant/qdrant

# Start the Next.js web UI (port 3101)
cd web && npm run dev -- --port 3101
```

### 6. Run evaluation

```bash
# Smoke eval (25 questions, CI gate)
python -m src.eval run --dataset smoke

# Regression eval (106 questions, weekly)
python -m src.eval run --dataset regression

# Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### 7. Lint and typecheck

```bash
ruff check src/urban_rag/
ruff format src/urban_rag/
pyright src/urban_rag/
```

---

## Project Structure

```
urban-planning-rag/
├── src/urban_rag/
│   ├── api/              # FastAPI gateway (main.py, /v1/query streaming endpoint)
│   ├── embed/            # ColQwen2.5 + GTE-ModernColBERT encoder loaders
│   │   ├── colqwen.py    # Visual embedding model
│   │   ├── text_encoder.py  # Text embedding model
│   │   └── serve.py      # Embed service (uvicorn, port 3102)
│   ├── index/            # Qdrant batch indexers (visual, text, BM25 sparse)
│   │   ├── batch.py      # Visual index (ColQwen2.5 → Qdrant pages_visual)
│   │   ├── text_index.py # Text index (GTE-ModernColBERT → Qdrant pages_text)
│   │   └── sparse.py     # Sparse BM25 indexer
│   ├── retrieve/         # Query execution
│   │   ├── visual.py     # Visual channel (Qdrant ANN + MaxSim)
│   │   ├── text.py       # Text channel (GTE-ModernColBERT + MaxSim)
│   │   ├── sparse.py     # Sparse BM25 channel (Qdrant native sparse)
│   │   ├── rerank.py     # Gemini 2.5 Flash VLM cross-encoder rerank
│   │   └── orchestrator.py  # 3-channel RRF fusion + orchestrates retrieval
│   ├── generate/         # Gemini streaming generation
│   │   ├── gemini.py     # Streaming client with SSE parsing
│   │   ├── orchestrator.py  # Grounded generation with citations
│   │   └── prompts.py   # Prompt templates (fast/deep modes)
│   ├── ingest/           # PDF parse + render ingest pipeline
│   │   ├── load.py       # PDF validation and hashing
│   │   ├── parse.py      # Docling/Marker markdown extraction
│   │   ├── classify.py   # Per-page DPI classifier (text vs visual)
│   │   ├── render.py     # PDF → PNG at adaptive DPI (100/250)
│   │   ├── chunk.py      # Text chunking with overlap
│   │   └── sections.py   # Section boundary detection
│   ├── eval/             # RAGAS metrics + smoke/regression CI gates
│   │   └── metrics/
│   │       └── ragas_wrapper.py  # Pinned judge model, reproducible scores
│   ├── telemetry/        # Observability
│   │   ├── tracing.py    # Langfuse OTel spans
│   │   └── metrics.py    # Prometheus gauges and histograms
│   ├── cli/              # Typer CLI commands (ingest, corpus, query)
│   └── common/           # Settings, types, logging, errors
├── web/                  # Next.js 14 App Router frontend
│   ├── app/
│   │   └── page.tsx     # Main query UI
│   └── lib/
│       └── api.ts        # SSE streaming client
├── eval/
│   ├── smoke.jsonl       # 25 CI gate questions
│   └── regression.jsonl  # 106 regression questions
├── infra/
│   ├── lightning/        # Lightning AI GPU deployment scripts
│   └── grafana/         # Dashboard configs
└── services.yaml        # Service commands manifest (single source of truth)
```

---

## Deferred Items (Require Credentials)

The following are not yet deployed — pending infrastructure credentials:

- **GPU embed service** on Lightning AI Studios (LitServe) — deploy with `infra/lightning/deploy-embed.sh`
- **Qdrant Cloud** production cluster — update `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
- **Vercel web deployment** — see `web/VERCEL_DEPLOY.md`

---

## Why Visual RAG?

Planning documents contain tables, diagrams, flowcharts, and color-coded maps. Traditional OCR destroys spatial layout and visual context. This system embeds entire page images as multi-vector representations, preserving all visual information for retrieval.

---

## Citation

```bibtex
@software{choudhary2026urbanrag,
  author = {Choudhary, Tanmay},
  title = {Urban Planning RAG: Visual Retrieval for Indian Planning Documents},
  year = {2026},
  url = {https://github.com/tanmayxchoudhary/urban-planning-rag}
}
```
