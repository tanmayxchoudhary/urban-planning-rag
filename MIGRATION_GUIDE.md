# Migration Guide: v1.0.0 → v2.0.0

Complete step-by-step guide for upgrading to Urban Planning RAG v2.0.0.

**Estimated time:** 15-20 minutes (mostly automated)

---

## Prerequisites

- Python 3.10+
- GPU with 8GB+ VRAM (for embedding)
- Original PDF documents (for re-embedding)

---

## Step 1: Backup v1.0.0 Data (Optional)

If you want to compare v1 vs v2 performance:

```bash
# Backup old embeddings
mkdir -p backups/v1.0.0
cp -r data/embeddings backups/v1.0.0/
cp -r data/page_images backups/v1.0.0/

# Note: FAISS index not needed (v2 uses ChromaDB)
```

**Time:** 1-2 minutes

---

## Step 2: Update Git Repository

```bash
# Fetch latest changes
git fetch origin

# Checkout v2.0.0
git checkout v2.0.0

# Or pull main if v2.0.0 is merged
git pull origin main
```

**Time:** < 1 minute

---

## Step 3: Update Dependencies

### Clean Install (Recommended)

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Remove old packages
pip uninstall -r requirements.txt -y

# Install new packages
pip install -r requirements.txt
```

### Or Upgrade in Place

```bash
pip install --upgrade -r requirements.txt
```

**Time:** 2-3 minutes

**Verify installation:**
```bash
python -c "import chromadb, fitz, cv2; print('✅ All imports OK')"
```

---

## Step 4: Re-Embed Documents

**⚠️ CRITICAL STEP:** v1.0.0 embeddings are incompatible.

### Option A: Re-Embed from PDFs (Recommended)

```bash
# Ensure PDFs are in ./pdfs directory
ls pdfs/

# Run embedding with v2.0.0
python scripts/embed.py --docs-dir ./pdfs --output-dir ./data

# Expected output:
# 🔍 Classifying pages for Adaptive DPI...
#   📄 swm_2016.pdf
#      📝 TEXT_ONLY: 18 pages @ 100 DPI
#      📊 HAS_VISUALS: 23 pages @ 250 DPI
#   ... (continues for all PDFs)
# ✅ Embedded 738 total pages
```

**Time:** ~6 minutes on L4/A10 GPU

### Option B: Download Pre-Generated v2.0.0 Embeddings

If available from project maintainers:

```bash
# Download from Google Drive / project releases
wget <v2.0.0-embeddings-url> -O v2-embeddings.zip

# Extract
unzip v2-embeddings.zip -d data/

# Verify
ls data/embeddings/
# Should see: embeddings.pt, metadata.json
```

**Time:** 2-3 minutes (download dependent)

---

## Step 5: Verify Metadata Schema

Check that new fields exist:

```bash
# Inspect metadata
python -c "
import json
with open('data/embeddings/metadata.json') as f:
    meta = json.load(f)
    first = meta[0]
    assert 'dpi' in first, 'Missing dpi field!'
    assert 'page_type' in first, 'Missing page_type field!'
    print('✅ Metadata schema valid')
    print(f'   dpi: {first[\"dpi\"]}')
    print(f'   page_type: {first[\"page_type\"]}')
"
```

Expected output:
```
✅ Metadata schema valid
   dpi: 250
   page_type: HAS_VISUALS
```

---

## Step 6: Test Retrieval

### Basic Test

```bash
python cli.py "What is FSI for residential zones?"
```

Expected output:
```
🚀 Initializing Urban Planning RAG System v2.0.0
============================================================
📂 Loading metadata...
🗄️ Initializing ChromaDB...
  ✅ Found existing index: 919224 patches
💾 Loading embeddings for MaxSim reranking...
  ✅ Loaded 738 page embeddings (list of tensors)
...

🔍 Query: 'What is FSI for residential zones?'
⚡ Stage 1: Multi-Query Token Expansion (top-3 tokens)...
  Found 50 candidate pages
🔥 Stage 2: MaxSim reranking...

======================================================================
📊 RETRIEVAL METRICS
======================================================================
🎯 Top 3 Selected (after MaxSim):
...
```

### Advanced Test (v2.0.0 Features)

```bash
# Test multi-query expansion
python cli.py "parking requirements" \
    --candidates 100 \
    --query-tokens 5 \
    --top-k 5

# Test retrieve-only
python cli.py "FSI norms" --retrieve-only

# Hide metrics
python cli.py "open space standards" --no-metrics
```

**Time:** 1-2 minutes per test

---

## Step 7: Update Your Code (If Using Python API)

### Minimal Update (Backward Compatible)

```python
# v1.0.0 code still works!
from src.rag import UrbanPlanningRAG

rag = UrbanPlanningRAG(data_dir="./data", load_query_encoder=True)
results = rag.retrieve(query="What is FSI?", top_k=3)
# ✅ Works, but doesn't use v2.0.0 features
```

### Full v2.0.0 Update

```python
from src.rag import UrbanPlanningRAG

# Initialize (same as v1.0.0)
rag = UrbanPlanningRAG(data_dir="./data", load_query_encoder=True)

# Retrieve with v2.0.0 features
results = rag.retrieve(
    query="What is FSI?",
    top_k=3,
    n_candidates=100,      # NEW: More candidates for Stage 2
    num_query_tokens=5,    # NEW: More query tokens
    show_metrics=True       # NEW: Display retrieval metrics
)

# View all candidates (debugging)
rag.show_all_candidates(top_n=50)

# Full RAG pipeline (unchanged API)
answer = rag.answer_query("What is FSI?", top_k=3)
```

---

## Step 8: Remove Old System Dependencies (Optional)

```bash
# Poppler no longer needed
# Ubuntu/Debian:
# sudo apt-get remove poppler-utils

# macOS:
# brew uninstall poppler

# Note: Only remove if not used by other projects
```

---

## Step 9: Performance Comparison (Optional)

Compare v1.0.0 vs v2.0.0 on known queries:

```python
# test_migration.py
from src.rag import UrbanPlanningRAG

# Load v2.0.0 RAG
rag_v2 = UrbanPlanningRAG(data_dir="./data", load_query_encoder=True)

# Test queries
test_queries = [
    "What is FSI for residential zones?",
    "Parking requirements for commercial buildings?",
    "Open space standards for layouts?"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    results = rag_v2.retrieve(query, top_k=3)

    print("\nTop 3 Results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['source']} p.{r['page']} (Score: {r['similarity']:.1f})")
```

Compare against v1.0.0 results (if backed up).

---

## Troubleshooting

### Issue: "chromadb module not found"

```bash
pip install chromadb>=0.4.0
```

### Issue: "embeddings.pt is incompatible"

```bash
# Re-embed documents (Step 4)
python scripts/embed.py --docs-dir ./pdfs --output-dir ./data
```

### Issue: "PyMuPDF (fitz) import error"

```bash
pip install PyMuPDF>=1.23.0
```

### Issue: "ChromaDB index build hangs"

```bash
# Delete and rebuild
rm -rf data/chroma_db
# Restart CLI - will rebuild automatically
python cli.py "test"
```

### Issue: "Model download fails"

```bash
# Set HuggingFace cache
export HF_HOME=/path/to/cache

# Or download manually:
# python -c "from transformers import AutoModel; AutoModel.from_pretrained('TomoroAI/tomoro-colqwen3-embed-4b', trust_remote_code=True)"
```

### Issue: "VRAM OOM during embedding"

```bash
# Reduce batch size
python scripts/embed.py --docs-dir ./pdfs --batch-size 10

# Or use CPU (slow):
# CUDA_VISIBLE_DEVICES="" python scripts/embed.py ...
```

---

## Rollback to v1.0.0 (If Needed)

```bash
# Checkout v1.0.0 tag
git checkout v1.0.0-stable-legacy

# Restore v1.0.0 dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Restore v1.0.0 data (if backed up)
rm -rf data/embeddings data/page_images
cp -r backups/v1.0.0/* data/
```

---

## Post-Migration Checklist

- [ ] Dependencies installed without errors
- [ ] Documents re-embedded successfully
- [ ] CLI queries work with v2.0.0 output
- [ ] Retrieval metrics display correctly
- [ ] Python API updated (if applicable)
- [ ] Performance acceptable (0.5% quality loss documented)

---

## Next Steps

- Read `CHANGELOG.md` for full feature list
- Experiment with `--candidates` and `--query-tokens` flags
- Try `rag.show_all_candidates()` for debugging
- Report issues: https://github.com/tanmayxchoudhary/urban-planning-rag/issues

---

## Support

- **Documentation:** See `BREAKING_CHANGES.md`, `README.md`
- **Issues:** GitHub Issues page
- **Old version:** Use `v1.0.0-stable-legacy` tag for 6 months post-v2.0.0
