# Setup Guide

Complete step-by-step guide to set up the Urban Planning RAG system locally.

---

## Prerequisites

- Python 3.10 or higher
- 4GB+ RAM
- Internet connection (for Gemini API)
- ~2GB free disk space (for embeddings and images)

**Optional:**
- NVIDIA GPU with 16GB+ VRAM (for query encoding)
- Otherwise, use Lightning.ai for query encoding

---

## Step 1: Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/urban-planning-rag.git
cd urban-planning-rag
```

---

## Step 2: System Dependencies

> **v2.0.0**: No system dependencies required! PyMuPDF handles all PDF processing natively.

Skip to Step 3.

---

## Step 3: Python Virtual Environment

### Create virtual environment

```bash
# Using venv (built-in)
python -m venv .venv

# Or using virtualenv
virtualenv .venv

# Or using conda
conda create -n urban-rag python=3.10
```

### Activate environment

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

Your terminal prompt should now show `(.venv)`.

---

## Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import chromadb; print(f'ChromaDB: {chromadb.__version__}')"
python -c "from google import genai; print('Gemini SDK OK')"
```

---

## Step 5: Download Data Files

Embeddings and page images are not included in the repository due to size (>1GB).

### Option A: Google Drive (Recommended)

1. Download from: [ADD YOUR GOOGLE DRIVE LINK]
2. Extract the archive
3. You should have:
   - `embeddings.pt` (573 MB)
   - `metadata.json` (few KB)
   - `page_images/` folder (738 PNG files)

### Option B: Generate Embeddings Yourself

If you have access to a GPU (16GB+ VRAM):

1. Place your PDFs in `docs/` directory
2. Upload `notebooks/embed_docs.ipynb` to Lightning.ai
3. Run all cells
4. Download the outputs

### Directory Structure After Download

```
urban-planning-rag/
├── data/
│   ├── embeddings/
│   │   ├── embeddings.pt        ← Downloaded
│   │   └── metadata.json        ← Downloaded
│   └── page_images/             ← Downloaded
│       ├── swm_2016__page_0001.png
│       ├── swm_2016__page_0002.png
│       └── ... (738 files total)
```

**Place files exactly as shown above.**

---

## Step 6: Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with Google account
3. Click "Get API Key" → "Create API key"
4. Copy your key

**Create `.env` file in project root:**

```bash
echo "GEMINI_API_KEY=your-actual-key-here" > .env
```

Or manually create `.env`:
```
GEMINI_API_KEY=AIzaSy...your-key-here
```

**Security:** The `.env` file is gitignored and will not be pushed to GitHub.

---

## Step 7: Verify Setup

Run the test script:

```bash
python scripts/test_gemini.py
```

Expected output:
```
Available Gemini models:
  ✅ gemini-3-flash-preview
  ✅ gemini-2.5-flash

🧪 Testing gemini-3-flash-preview...

📝 Response:
Urban planning is the process of designing and organizing...

✅ Gemini API working with new SDK
```

---

## Step 8: Run Your First Query

```bash
python cli.py "What is FSI for residential zones?"
```

Expected output:
```
============================================================
🚀 Initializing Urban Planning RAG System v2.0.0
============================================================

📂 Loading metadata...
🗄️ Initializing ChromaDB...
  ✅ Found existing index: 919224 patches
💾 Loading embeddings for MaxSim reranking...
  ✅ Loaded 738 page embeddings (list of tensors)
🤖 Initializing Gemini VLM...
  ✅ Gemini client ready

✅ RAG system ready with 738 pages indexed
============================================================

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

---

## Troubleshooting

### "GEMINI_API_KEY not found"

Make sure `.env` file exists in project root and contains:
```
GEMINI_API_KEY=your-key-here
```

### "Embeddings not found"

Check that files are in the correct location:
```bash
ls -lh data/embeddings/embeddings.pt
ls data/page_images/ | wc -l  # Should show 738
```

### "poppler-utils not found" (pdf2image error)

Install poppler system dependency (see Step 2).

### CUDA/GPU errors

If you don't have a GPU, the system will automatically use CPU.

To force CPU:
```python
# In src/rag.py, line 100, change:
device = "cpu"  # Force CPU
```

### Query encoding requires GPU

The system works WITHOUT GPU for:
- Loading embeddings
- ChromaDB search
- Gemini generation

GPU is ONLY needed for:
- Encoding NEW queries (if you want to do it locally)

**Alternative:** Use Lightning.ai notebook for query encoding, or wait for our hosted API.

---

## Optional: Using Lightning.ai for Query Encoding

If you don't have a local GPU but want to encode queries:

1. Upload `notebooks/rag.ipynb` to Lightning.ai
2. Upload your `data/` directory
3. Run queries in the notebook
4. Gemini calls work from Lightning.ai

---

## Next Steps

- Read `docs/project-documentation.md` for technical deep-dive
- See `README.md` for usage examples
- Explore `notebooks/` to understand how embeddings were created
- Modify `src/rag.py` to customize behavior

---

## Getting Help

**Issues?** Open an issue on GitHub with:
- Error message
- Your environment (OS, Python version)
- Steps to reproduce

**Questions?** See project README or contact maintainer.
