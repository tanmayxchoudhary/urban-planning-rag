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

- **Embeddings**: TomoroAI/tomoro-colqwen3-embed-8b (multi-vector visual document retrieval)
- **Vector DB**: FAISS (cosine similarity with averaged embeddings)
- **VLM**: Gemini 3.0 Flash / 2.5 Flash (Google AI Studio API)

**Why visual RAG?**  
Planning documents contain tables, diagrams, flowcharts, and color-coded maps. Traditional OCR destroys spatial layout and visual context. Our system embeds entire page images, preserving all visual information.

---

## 📚 Indexed Documents

- **SWM 2016** (Solid Waste Management) - 40 pages
- **URDPFI Vol 1** (Urban and Regional Development Plans) - 447 pages
- **URDPFI Vol 2** - 251 pages

**Total**: 738 pages indexed

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/urban-planning-rag.git
cd urban-planning-rag
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Install system dependency (poppler-utils for PDF processing)
# Ubuntu/Debian:
sudo apt-get install poppler-utils

# macOS:
brew install poppler
```

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

**Want to embed your own documents?** See [`docs/EMBEDDING.md`](docs/EMBEDDING.md) for complete guide.

### 4. Set Up Gemini API Key

Get a free API key from [Google AI Studio](https://aistudio.google.com/)

Create `.env` file in project root:
```bash
GEMINI_API_KEY=your-key-here
```

### 5. Run Your First Query

```bash
python cli.py "What is FSI for residential zones?"
```

---

## 💻 Usage

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

**Using the Python script:**
```bash
# On Lightning.ai, Colab, or local GPU (16GB+ VRAM)
python scripts/embed.py --docs-dir ./docs --output-dir ./data
```

**Or using the notebook** (same process, interactive):
- See `notebooks/embed_docs.ipynb`

**Process:**
```
PDF → Page Images (150 DPI) → ColQwen → Embeddings (256×320 per page)
```

**Output:**
- `embeddings.pt` - 738 pages × 256 patches × 320 dimensions
- `metadata.json` - Page metadata (source, page number)
- `page_images/` - 738 PNG files at 150 DPI

### 2. Query Pipeline (GPU Required for Query Encoding)

**Full pipeline:**
```
User Query → ColQwen Encoding → FAISS Search → Top-K Pages → Gemini VLM → Answer
```

**GPU Requirements:**
- **Document embedding:** Requires GPU (16GB+ VRAM) - one-time operation
- **Query encoding:** Requires GPU (same model) - per query
- **Retrieval + Generation:** Works on CPU

**Options for running queries:**
1. **On GPU** (Lightning.ai, Colab, local):
   ```python
   rag = UrbanPlanningRAG(load_query_encoder=True)  # Loads ColQwen
   answer = rag.answer_query("What is FSI?")
   ```

2. **Without GPU** (provide pre-computed query embeddings):
   ```python
   rag = UrbanPlanningRAG(load_query_encoder=False)
   results = rag.retrieve(query_embedding=precomputed_vector)
   ```

**Note:** Both embedding and query encoding use the same 16GB ColQwen model.

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
│   └── __init__.py
│
├── scripts/
│   ├── check_docs.py            # PDF inspection utility
│   └── test_gemini.py           # Test Gemini API connection
│   └── embed.py		 # Embed your pdfs (use --doc-dir "path/to/pdf/folder")	
├── notebooks/                   # Development notebooks (Lightning.ai)
│   ├── embed_docs.ipynb         # Document embedding pipeline
│   └── rag.ipynb                # Complete RAG system
│
├── data/                        # Data files (gitignored, download separately)
│   ├── embeddings/
│   │   ├── embeddings.pt
│   │   └── metadata.json
│   └── page_images/
│
└── docs/
    └── embedding.md		 # Embed your own pdfs
    |---setup.md 		# detailed setup guide 
```

---

## 🔧 Technical Details

### Embeddings

- **Model**: TomoroAI/tomoro-colqwen3-embed-8b
- **Architecture**: ColPali-style multi-vector embeddings
- **Output**: 256 patch vectors per page (320-dim each)
- **Storage**: Averaged to single vector for FAISS compatibility
- **Trade-off**: Lower similarity scores (0.3-0.5) but correct relative ranking

### Retrieval

- **Index**: FAISS IndexFlatIP (inner product for cosine similarity)
- **Normalization**: L2 normalization on embeddings
- **Similarity**: Cosine similarity via normalized inner product
- **Speed**: ~10ms per query on CPU

### Generation

- **Model**: Gemini 3.0 Flash Preview / Gemini 2.5 Flash
- **Input**: Natural language query + top-k page images
- **Output**: Answer with page citations
- **Cost**: Free tier (1500 requests/day)

---

### Citing This Work

If you use this system in your research, please cite:

```
Tanmay Choudhary.  "Visual RAG for Urban Planning Documents."
GitHub: https://github.com/tanmayxchoudhary/urban-planning-rag
```

### Development Documentation

See `docs/EMBEDDING.md` to embed new documents:
- How to use Lightning.ai for embedding
- Alternative GPU options (Colab, local, cloud)
- Troubleshooting embedding issues
- Adding new documents to the index

---

## 🚧 Limitations & Future Work

### Current Limitations

- Query encoding requires GPU (16GB VRAM) or cloud service
- Limited to 3 documents (738 pages) - scaling in progress
- Averaged embeddings lose some spatial precision
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

- This project is licensed under the **Apache License 2.0**.

---

## 🙏 Acknowledgments

- **ColQwen** by TomoroAI for visual document retrieval
- **Gemini** by Google for vision-language generation
- **Lightning.ai** for GPU compute
- **FAISS** by Meta for efficient similarity search

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
