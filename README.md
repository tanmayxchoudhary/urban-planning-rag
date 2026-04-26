# Urban Planning RAG 🏙️

**Visual document retrieval system for Indian urban planning regulations**

Query planning documents (NBC, URDPFI, SWM) using state-of-the-art multimodal AI. Get accurate answers with precise page citations.

---

## 🎯 What This Does

Ask questions like:
- *"What is the FSI for residential zones?"*
- *"What are the parking requirements for commercial buildings?"*
- *"What are the indicators of good governance?"*

The system:
1. **Retrieves** relevant pages from planning documents using visual embeddings
2. **Generates** accurate answers using Gemini VLM
3. **Cites** specific page numbers as sources

---

## 🏗️ Architecture

- **Embeddings**: TomoroAI/tomoro-colqwen3-embed-4b (multi-vector visual retrieval, 8GB VRAM)
- **Vector DB**: ChromaDB (patch-level indexing with MaxSim reranking)
- **Retrieval**: Two-stage pipeline (Multi-Query Expansion + MaxSim late interaction)
- **VLM**: Gemini 3.0 Flash / 2.5 Flash (Google AI Studio API)
- **Deployment**: Cloud GPU ready with API endpoints

**Why visual RAG?**  
Planning documents contain tables, diagrams, flowcharts, and color-coded maps. Traditional OCR destroys spatial layout and visual context. Our system embeds entire page images, preserving all visual information.

---

## 📚 Indexed Documents

- **SWM 2016** (Solid Waste Management) - 91 pages
- **URDPFI Vol 1** (Urban and Regional Development Plans) - 447 pages
- **URDPFI Vol 2** - 251 pages
- **NBC** (National Building Code) - 2258 pages

**Total**: 3,047 pages indexed

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/tanmayxchoudhary/urban-planning-rag.git
cd urban-planning-rag
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

> **v3.0.0**: Unified embedding script with comprehensive error handling. No system dependencies required!

### 3. Download Data Files

The embeddings and page images are too large for GitHub (>1GB total).

**Download from:** [Google Drive Link - https://drive.google.com/drive/folders/1cAXUc5Yk24spGDQOxczJgCYYWd0ORlPx]

Extract and place files in this structure:
```
data/
├── embeddings/
│   ├── embeddings.pt       (573 MB)
│   └── metadata.json
└── page_images/
    ├── swm_2016__page_0001.png
    ├── swm_2016__page_0002.png
    ├── ...
    └── urdpfi_vol2__page_0251.png
```

**Want to embed your own documents?** See `scripts/README.md` for complete guide.

### 4. Set Up Gemini API Key

Get a free API key from [Google AI Studio](https://aistudio.google.com/)

Create `.env` file in project root:
```bash
GEMINI_API_KEY=your-key-here
```

### 5. Run Your First Query

```bash
# Using the pipeline
python scripts/pipeline.py --query "What is FSI for residential zones?"

# Or using the CLI
python cli.py "What is FSI for residential zones?"
```

---

## 💻 Usage

### Pipeline Script (Recommended)

```bash
# Full pipeline: embed + index + query
python scripts/pipeline.py --docs-dir ./pdfs --query "What is FSI?"

# Check data status
python scripts/pipeline.py --check

# Individual steps
python scripts/pipeline.py --docs-dir ./pdfs --step embed
python scripts/pipeline.py --step index
python scripts/pipeline.py --query "Parking requirements" --step query
```

### Command Line Interface

**Basic query:**
```bash
python cli.py "What are parking requirements?"
```

**Retrieve more pages:**
```bash
python cli.py --query "open space standards" --top-k 5
```

**Use different Gemini model:**
```bash
python cli.py "building height regulations" --model gemini-2.5-flash
```

**Retrieve only (no answer generation):**
```bash
python cli.py "FSI regulations" --retrieve-only
```

### Python API

```python
from src.rag import UrbanPlanningRAG

# Initialize RAG system
rag = UrbanPlanningRAG(data_dir="./data")

# Get answer with citations
answer = rag.answer_query("What is FSI for residential zones?", top_k=3)
print(answer)

# Or just retrieve relevant pages
results = rag.retrieve(query="parking requirements", top_k=5)
for r in results:
    print(f"{r['source']} - Page {r['page']}")
```

---

## ⚙️ How It Works

### 1. Document Embedding (One-Time, GPU Required)

PDFs are converted to images and embedded using ColQwen.

**Using the unified script:**
```bash
# On Lightning.ai, Colab, or local GPU (8GB+ VRAM)
python scripts/embed.py --docs-dir ./pdfs --output-dir ./data

# With verbose logging
python scripts/embed.py --docs-dir ./pdfs --verbose

# Custom batch size for memory constraints
python scripts/embed.py --docs-dir ./pdfs --batch-size 10
```

**Process (v3.0.0):**
```
PDF → PageClassifier (adaptive DPI) → ColQwen3-4B → Variable-length embeddings
```

**Output:**
- `embeddings.pt` - Page embeddings (variable patch counts × 320 dimensions)
- `metadata.json` - Page metadata (source, page number, DPI, page type)
- `page_images/` - PNG files at adaptive DPI (100 or 250)

### 2. Query Pipeline

**Query Pipeline (v3.0.0):**
```
User Query → ColQwen Encoding → ChromaDB Multi-Query Expansion → MaxSim Reranking → Gemini VLM → Answer
```

**GPU Requirements:**
- **Document embedding:** Requires GPU (8GB+ VRAM) - one-time operation
- **Query encoding:** Requires GPU (same model) - per query
- **Retrieval + Generation:** Works on CPU

---

## 📂 Project Structure

```
urban-planning-rag/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── cli.py                       # Command-line interface
├── .env                         # API keys (create this, not tracked)
│
├── src/
│   ├── rag.py                   # Main RAG class
│   ├── indexer_optimized.py     # Optimized ChromaDB indexing
│   └── __init__.py
│
├── scripts/                     # Core scripts (see scripts/README.md)
│   ├── embed.py                 # Unified embedding script (v3.0.0)
│   ├── pipeline.py              # End-to-end pipeline
│   ├── check_docs.py            # PDF inspection utility
│   ├── test_gemini.py           # Test Gemini API connection
│   └── archive/                 # Old versioned scripts
│
├── notebooks/                   # Development notebooks
│   ├── embed_docs.ipynb         # Document embedding pipeline
│   └── rag.ipynb                # Complete RAG system
│
├── data/                        # Data files (gitignored)
│   ├── embeddings/
│   │   ├── embeddings.pt
│   │   └── metadata.json
│   └── page_images/
│
└── docs/                        # Documentation
    └── setup.md                 # Detailed setup guide
```

---

## 🔧 Technical Details

### Embeddings

- **Model**: TomoroAI/tomoro-colqwen3-embed-4b
- **Architecture**: ColPali-style multi-vector embeddings
- **Output**: Variable patch vectors per page (320-dim each)
- **Storage**: List of tensors with variable patch counts
- **VRAM**: 8GB (down from 16GB in v1.0.0)

### Retrieval (v3.0.0)

- **Index**: ChromaDB PersistentClient (patch-level)
- **Stage 1**: Multi-Query Token Expansion
- **Stage 2**: MaxSim late-interaction reranking
- **Speed**: ~50ms per query on CPU

### Generation

- **Model**: Gemini 3.0 Flash Preview / Gemini 2.5 Flash
- **Input**: Natural language query + top-k page images
- **Output**: Answer with page citations
- **Cost**: Free tier (1500 requests/day)

---

## 🆕 What's New in v3.0.0

### Major Improvements

- **✨ Unified Embedding Script**: Single `embed.py` replaces 29 versioned scripts
- **🔧 NumPy 2.x Compatibility**: Updated dependencies for Python 3.14+
- **🛡️ Comprehensive Error Handling**: Detailed error messages and graceful degradation
- **📊 Better Logging**: Structured logging with timestamps and progress bars
- **🚀 Pipeline Script**: End-to-end automation with `pipeline.py`

### Consolidated Scripts

Old scripts archived to `scripts/archive/`:
- `embed_v21.py` through `embed_v30_lightning.py`
- `lightning_embed.py` variants
- Multiple benchmark scripts

### New Features

- **Dependency checking**: Automatic verification of required packages
- **Exit codes**: Proper exit codes for scripting (0=success, 1=error, 2=partial, 130=interrupted)
- **Data status check**: `pipeline.py --check` shows what data exists
- **Verbose mode**: `-v` flag for detailed logging

---

## 🚧 Limitations & Future Work

### Current Limitations

- Query encoding requires GPU (8GB VRAM) or cloud service
- Limited to 5 documents (~3,000 pages)
- Gemini free tier rate limit (1500 requests/day)

### Roadmap

**Phase 2: Improved Retrieval**
- Proper ColPali MaxSim scoring (no averaging)
- Multi-vector index for better precision
- Re-ranking with cross-encoder

**Phase 3: Scale Up**
- 100+ documents (NBC, State Master Plans, textbooks)
- Hybrid search (visual + text)
- Document chunking for long pages

**Phase 4: Production Ready**
- Query encoding API (no local GPU needed)
- Web interface
- Multi-language support (Hindi, regional languages)

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Adding more documents (NBC, State Master Plans)
- Improving retrieval precision
- Building web interface
- Multi-language support

---

## 📄 License

This project is licensed under the **Apache License 2.0**.

---

## 🙏 Acknowledgments

- **ColQwen** by TomoroAI for visual document retrieval
- **Gemini** by Google for vision-language generation
- **Lightning.ai** for GPU compute
- **ChromaDB** for efficient patch-level vector search

---

## Citation

If you use this work in academic research, please cite:
```bibtex
@software{choudhary2026urbanrag,
  author = {Choudhary, Tanmay},
  title = {Urban Planning RAG: Visual Retrieval for Indian Planning Documents},
  year = {2026},
  url = {https://github.com/tanmayxchoudhary/urban-planning-rag}
}
```
