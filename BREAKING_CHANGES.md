# Breaking Changes in v2.0.0

**⚠️ IMPORTANT: v2.0.0 is NOT backward compatible with v1.0.0. Re-embedding required.**

---

## Overview

Urban Planning RAG v2.0.0 introduces major architectural changes for improved efficiency and retrieval quality. All users must re-embed documents and rebuild indexes.

---

## 1. Embeddings Format Incompatibility

### Breaking Change
Embeddings storage format changed from stacked tensor to list of variable-length tensors.

### v1.0.0 (OLD)
```python
embeddings = torch.stack(all_embeddings)  # Shape: [738, 256, 320]
# Fixed 256 patches per page
```

###v2.0.0 (NEW)
```python
embeddings = [tensor1, tensor2, ...]  # List of tensors
# Variable patch counts: min=973, max=1271, mean=1246
```

### Impact
- v1.0.0 `embeddings.pt` cannot be loaded in v2.0.0
- v2.0.0 embeddings include adaptive DPI metadata

### Migration
```bash
# Re-embed all documents
python scripts/embed.py --docs-dir ./pdfs --output-dir ./data
```

**Time:** ~6 minutes on L4/A10 GPU (738 pages)

---

## 2. Vector Database Change (FAISS → ChromaDB)

### Breaking Change
Complete database backend replacement for patch-level indexing.

### v1.0.0 (OLD)
```python
import faiss
index = faiss.IndexFlatIP(320)
# Document-level index (738 documents)
```

### v2.0.0 (NEW)
```python
import chromadb
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_or_create_collection("urban_planning_patches")
# Patch-level index (~919k patches)
```

### Impact
- FAISS indexes are abandoned
- ChromaDB creates `chroma_db/` directory (not tracked in git)
- Index size increases: 738 docs → 919k patches

### Migration
No manual action needed. RAG automatically builds ChromaDB index on first run.

---

## 3. retrieve() API Signature Change

### Breaking Change
New parameters added for two-stage retrieval control.

### v1.0.0 (OLD)
```python
results = rag.retrieve(
    query="What is FSI?",
    top_k=3
)
```

### v2.0.0 (NEW)
```python
results = rag.retrieve(
    query="What is FSI?",
    top_k=3,
    n_candidates=50,        # NEW: Stage 2 pool size
    num_query_tokens=3,     # NEW: Multi-query expansion
    show_metrics=True       # NEW: Display rank improvements
)
```

### Impact
- Old code still works (new params have defaults)
- But cannot access new v2.0.0 features without updating calls

### Migration
```python
# Minimal update (backward compatible)
results = rag.retrieve(query="FSI", top_k=3)

# Full v2.0.0 features
results = rag.retrieve(
    query="FSI",
    top_k=3,
    n_candidates=100,      # More candidates for reranking
    num_query_tokens=5,    # More query tokens
    show_metrics=True       # See rank improvements
)
```

---

## 4. Model Change (ColQwen-8B → ColQwen2-4B)

### Breaking Change
Smaller, more efficient model with minor quality trade-off.

### Comparison
| Metric | v1.0.0 (8B) | v2.0.0 (4B) | Change |
|--------|-------------|-------------|--------|
| VRAM | 16GB | 8GB | **-50%** |
| Quality | Baseline | -0.5% | Acceptable |
| Speed | Baseline | +15% | Faster |

### Impact
- Pre-trained v1.0.0 embeddings incompatible
- Model ID changed: `tomoro-colqwen3-embed-8b` → `tomoro-colqwen3-embed-4b`

### Migration
Re-embedding automatically downloads new model.

---

## 5. PDF Processing (pdf2image → PyMuPDF)

### Breaking Change
Backend change eliminates system dependency.

### v1.0.0 (OLD)
```bash
# Required system dependency
sudo apt-get install poppler-utils  # Ubuntu/Debian
brew install poppler                # macOS
```

### v2.0.0 (NEW)
```python
import fitz  # PyMuPDF - pure Python, no system deps
```

### Impact
- Poppler no longer needed
- Simpler installation on all platforms
- Slightly different image rendering (negligible)

### Migration
```bash
# Remove old dependency (optional)
pip uninstall pdf2image

# Install new dependency
pip install PyMuPDF>=1.23.0
```

---

## 6. Dependency Changes

### Removed
```
faiss-cpu>=1.7.4
pdf2image>=1.16.3
```

### Added
```
chromadb>=0.4.0
PyMuPDF>=1.23.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
colpali-engine>=0.2.0
einops>=0.7.0
```

### Migration
```bash
# Clean install recommended
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

---

## 7. Metadata Schema Change

### Breaking Change
New fields added to `metadata.json`.

### v1.0.0 (OLD)
```json
{
  "source": "urdpfi_vol1.pdf",
  "page": 1,
  "total_pages": 447
}
```

### v2.0.0 (NEW)
```json
{
  "source": "urdpfi_vol1.pdf",
  "page": 1,
  "total_pages": 447,
  "dpi": 250,
  "page_type": "HAS_VISUALS"
}
```

### Impact
- Old metadata files won't have `dpi` and `page_type`
- v2.0.0 code expects these fields

### Migration
Re-run embedding script to generate new metadata.

---

## 8. CLI Changes (Minor Breaking)

### Changed
- `--load-encoder` flag now loads 4B model (was 8B)
- VRAM comment updated: "requires 16GB" → "requires 8GB"

### New Flags
```bash
--candidates 50        # Stage 2 pool size
--query-tokens 3       # Multi-query expansion
--no-metrics           # Hide retrieval metrics
```

### Migration
No code changes needed. New flags are optional.

---

## Quick Migration Checklist

- [ ] **Backup v1.0.0 data** (optional, for comparison)
- [ ] **Update dependencies**: `pip install -r requirements.txt`
- [ ] **Re-embed documents**: `python scripts/embed.py --docs-dir ./pdfs`
- [ ] **Test retrieval**: `python cli.py "test query"`
- [ ] **Update code** (if using Python API): Add new `retrieve()` parameters
- [ ] **Remove poppler** (optional): No longer needed

---

## What's NOT Breaking

- CLI interface (existing commands work)
- Gemini API integration
- `answer_query()` signature (unchanged)
- Data directory structure (same paths)
- Environment variables (`.env` file unchanged)

---

## Support

- Migration issues: See `MIGRATION_GUIDE.md`
- Questions: Open GitHub issue
- Old version: Use `v1.0.0-stable-legacy` git tag
