# Changelog

All notable changes to Urban Planning RAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-01-09

**Major version bump with breaking changes. Re-embedding required.**

### Added
- **Two-stage retrieval pipeline**
  - Stage 1: Multi-Query Token Expansion (distinctive query tokens by L2 norm)
  - Stage 2: MaxSim late-interaction reranking
- **Adaptive DPI embedding** with PageClassifier
  - TEXT_ONLY pages: 100 DPI
  - HAS_VISUALS pages: 250 DPI
  - Automatic classification via edge density + drawing count
- **Retrieval metrics** system
  - `_display_retrieval_metrics()` - Shows rank improvements
  - `show_all_candidates()` - Full candidate analysis for debugging
  - Token coverage tracking (multi-token matches)
- **ChromaDB patch-level indexing** - Replaces FAISS document-level index
- **PyMuPDF backend** - No poppler system dependency required
- **CLI flags** for v2.0.0 features
  - `--candidates` - Stage 2 pool size (default 50)
  - `--query-tokens` - Multi-query expansion (default 3)
  - `--no-metrics` - Hide retrieval metrics
- **Metadata fields**: `dpi`, `page_type` in metadata.json

### Changed
- **Model:** ColQwen-8B (16GB VRAM) → ColQwen2-4B (8GB VRAM)
  - 50% VRAM reduction
  - 0.5% quality loss (acceptable trade-off)
  - 15% faster inference
- **Embeddings format:** Stacked tensor → List of variable-length tensors
  - Storage efficiency: Variable patch counts (min=973, max=1271)
- **Vector DB:** FAISS IndexFlatIP → ChromaDB PersistentClient
  - Document-level (738 docs) → Patch-level (~919k patches)
- **PDF processing:** pdf2image → PyMuPDF (fitz)
  - No poppler dependency
  - Pure Python solution
- **retrieve() API:** New parameters (n_candidates, num_query_tokens, show_metrics)
- **Dependencies:** See Breaking Changes section

### Removed
- **FAISS** integration (replaced by ChromaDB)
- **pdf2image** (replaced by PyMuPDF)
- **poppler-utils** system dependency (no longer needed)

### Breaking
- v1.0.0 embeddings incompatible with v2.0.0 (must re-embed)
- FAISS indexes abandoned (ChromaDB rebuilds automatically)
- retrieve() signature extended (backward compatible with defaults)
- metadata.json schema changed (new fields required)
- Model ID changed (automatic download on first run)

### Fixed
- None (this is initial stable v2.0.0 release)

### Performance
- **VRAM:** 16GB → 8GB (-50%)
- **Embedding speed:** ~6 min for 738 pages (L4/A10)
- **Retrieval quality:** -0.5% vs v1.0.0 (tested in notebooks)
- **Index size:** 738 docs → 919k patches (larger but more precise)

### Documentation
- Added `BREAKING_CHANGES.md` - Detailed breaking changes with code examples
- Added `MIGRATION_GUIDE.md` - Step-by-step upgrade guide (15-20 min)
- Updated `README.md` - v2.0.0 features, installation (no poppler), benchmarks

---

## [1.0.0] - 2025-01-07

**Initial stable release (tagged as v1.0.0-stable-legacy)**

### Added
- Visual RAG system for Indian urban planning documents
- ColQwen-8B multi-vector embeddings
- FAISS vector database with averaged embeddings
- Gemini VLM integration for answer generation
- CLI interface for querying
- Python API for programmatic usage
- Document embedding script (scripts/embed.py)
- PDF to image conversion (150 DPI fixed)
- Support for 3 documents (swm_2016, urdpfi_vol1, urdpfi_vol2)

### Features
- **Embeddings:** ColQwen-8B (256 patches × 320 dim per page)
- **Vector DB:** FAISS IndexFlatIP with cosine similarity
- **Retrieval:** Single-stage averaged embedding search
- **Generation:** Gemini 3.0 Flash / 2.5 Flash
- **CLI:** `python cli.py "query"` with --top-k, --model flags

### Known Limitations
- Fixed 150 DPI for all pages (no adaptive quality)
- Averaged embeddings lose spatial precision
- FAISS document-level search (no late interaction)
- Requires 16GB VRAM for embedding
- Requires poppler system dependency

---

## Release Notes

### v2.0.0 Highlights
- **50% VRAM reduction** enables broader GPU support (A10, L4, T4)
- **Adaptive DPI** optimizes quality vs storage (100 DPI text, 250 DPI visuals)
- **Two-stage retrieval** improves precision via MaxSim reranking
- **No system dependencies** (PyMuPDF replaces poppler)
- **Detailed metrics** for retrieval debugging and optimization

### Migration Path
See `MIGRATION_GUIDE.md` for complete upgrade instructions. Estimated time: 15-20 minutes.

### Benchmark: v1.0.0 vs v2.0.0
| Metric | v1.0.0 | v2.0.0 | Change |
|--------|--------|--------|--------|
| VRAM (embed) | 16GB | 8GB | **-50%** |
| VRAM (query) | 16GB | 8GB | **-50%** |
| Retrieval quality | Baseline | -0.5% | Acceptable |
| Speed (embed) | ~6 min | ~6 min | Same |
| Speed (query) | Baseline | +10% | Faster |
| Index size | 738 docs | 919k patches | Larger |
| Dependencies | 7 + poppler | 8 (no poppler) | Cleaner |

---

## Links
- **Repository:** https://github.com/tanmayxchoudhary/urban-planning-rag
- **Issues:** https://github.com/tanmayxchoudhary/urban-planning-rag/issues
- **v1.0.0 Tag:** `v1.0.0-stable-legacy` (maintained for 6 months)
- **v2.0.0 Tag:** `v2.0.0`

---

## Support Policy
- **v2.0.0:** Active development, new features
- **v1.0.0:** Security fixes only (6 months post-v2.0.0)
- **Older versions:** No support
