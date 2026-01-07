# Embedding New Documents

Guide for adding new documents to the Urban Planning RAG system.

---

## Overview

The embedding pipeline converts PDF documents into searchable vector embeddings. This is a **one-time process** that requires GPU resources.

**What you need:**
- PDF documents to embed
- GPU with 16GB+ VRAM (L4, A10, A100)
- OR access to cloud GPU (Lightning.ai, Google Colab, Kaggle)

**Output:**
- `embeddings.pt` - Vector embeddings for all pages
- `metadata.json` - Page metadata (source, page numbers)
- `page_images/` - PNG images of each page (150 DPI)

**Time estimate:** ~30 minutes for 1000 pages on L4 GPU

---

## Option 1: Lightning.ai (Recommended)

Lightning.ai provides free GPU credits for experimentation. This is how the original embeddings were created.

### Step 1: Create Lightning.ai Account

1. Go to https://lightning.ai
2. Sign up (free tier available)
3. Create a new Studio

### Step 2: Upload Notebook

1. Download `notebooks/embed_docs.ipynb` from this repository
2. In Lightning.ai Studio, click "Upload"
3. Select the notebook

### Step 3: Upload Your PDFs

Create a folder structure in Lightning.ai:

```
/teamspace/studios/this_studio/
├── embed_docs.ipynb
├── swm_2016.pdf
├── urdpfi_vol1.pdf
└── urdpfi_vol2.pdf
```

Or modify the notebook to use a different path.

### Step 4: Run the Notebook

**Cell 1:** Install dependencies
```python
!pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
!pip install transformers pillow requests pdf2image poppler-utils tqdm accelerate pypdf
!apt-get update && apt-get install -y poppler-utils
```

**Cell 2:** Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Cell 3:** Run embedding pipeline
- This loads ColQwen model (16GB)
- Converts PDFs to images
- Embeds all pages
- Saves outputs to `embeddings_output/`

### Step 5: Download Outputs

After the notebook completes:

1. Navigate to `embeddings_output/` folder
2. Download:
   - `embeddings.pt`
   - `metadata.json`
3. Navigate to `page_images/` folder
4. Download entire folder (or zip it first)

### Step 6: Place Files Locally

```bash
# Copy to your local project
cp embeddings.pt ~/urban-planning-rag/data/embeddings/
cp metadata.json ~/urban-planning-rag/data/embeddings/
cp -r page_images/ ~/urban-planning-rag/data/
```

---

## Option 2: Google Colab

Google Colab provides free GPU access (T4, 15GB VRAM). Slightly less powerful than Lightning.ai but works.

### Setup Colab Notebook

1. Go to https://colab.research.google.com
2. Create new notebook
3. Enable GPU: Runtime → Change runtime type → GPU (T4)

### Install Dependencies

```python
!pip install torch torchvision transformers pillow pdf2image pypdf tqdm accelerate
!apt-get install -y poppler-utils
```

### Upload PDFs

```python
from google.colab import files
uploaded = files.upload()  # Select your PDFs
```

### Run Embedding Code

Copy the main embedding code from `notebooks/embed_docs.ipynb` cells into Colab.

### Download Results

```python
from google.colab import files
files.download('embeddings_output/embeddings.pt')
files.download('embeddings_output/metadata.json')

# Zip images before download
!zip -r page_images.zip page_images/
files.download('page_images.zip')
```

---

## Option 3: Local GPU

If you have a workstation with NVIDIA GPU (16GB+ VRAM):

### Install CUDA Toolkit

**Ubuntu:**
```bash
sudo apt-get install nvidia-cuda-toolkit
```

**Verify:**
```bash
nvidia-smi  # Should show your GPU
```

### Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Run Embedding Script

```bash
# Use the standalone script (if you create one)
python scripts/embed_documents.py --docs-dir ./docs --output-dir ./data
```

Or use the notebook locally:
```bash
jupyter notebook notebooks/embed_docs.ipynb
```

---

## Option 4: Cloud GPU Services

### Kaggle

- Free GPU access (P100, 16GB VRAM)
- 30 hours/week limit
- Upload notebook → Enable GPU → Run

### Paperspace Gradient

- Free tier available
- Similar to Colab
- Better for longer runs

### Vast.ai (Paid)

- Rent GPU by the hour
- ~$0.10-0.30/hour for L4 equivalent
- Good for large-scale embedding

---

## Modifying for Your Documents

### Change PDF List

In `embed_docs.ipynb`, modify this section:

```python
pdf_files = [
    "swm_2016.pdf",           # Replace with your PDFs
    "urdpfi_vol1.pdf",
    "urdpfi_vol2.pdf",
    "nbc_2016.pdf",           # Add new documents
    "delhi_master_plan.pdf",
]
```

### Adjust DPI

Higher DPI = better quality but larger files:

```python
DPI = 150  # Default (good for tables/diagrams)
DPI = 200  # Better for small text
DPI = 100  # Faster, smaller files
```

### Batch Size

If you run out of memory, reduce batch size:

```python
BATCH_SIZE = 2   # Safer for 16GB VRAM
BATCH_SIZE = 4   # Works on 24GB VRAM
BATCH_SIZE = 8   # For 40GB+ VRAM
```

---

## Understanding the Embedding Process

### What Happens Behind the Scenes

```
PDF → pdf2image → PIL Images (150 DPI)
         ↓
    ColQwen Processor → Tokenization + Vision Features
         ↓
    ColQwen Model → Multi-Vector Embeddings (256 patches × 320 dim)
         ↓
    Save as PyTorch Tensor → embeddings.pt (BFloat16)
```

### File Sizes

**embeddings.pt:**
- Formula: `num_pages × 256 patches × 320 dim × 2 bytes (BFloat16)`
- Example: 738 pages → ~573 MB

**page_images/:**
- Formula: `num_pages × ~500KB per PNG (150 DPI)`
- Example: 738 pages → ~369 MB

**Total:** ~1GB for 738 pages

---

## Troubleshooting

### "CUDA out of memory"

**Solution:** Reduce batch size
```python
BATCH_SIZE = 1  # Slowest but safest
```

**Or** clear cache between batches:
```python
torch.cuda.empty_cache()
gc.collect()
```

### "flash-attn installation failed"

**Solution:** Use SDPA instead
```python
attn_implementation="sdpa"  # PyTorch's native attention
```

### "poppler-utils not found"

**Solution:** Install system dependency
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler
```

### Notebook crashes on large PDFs

**Solution:** Embed in chunks
```python
# Process one PDF at a time
for pdf_name in pdf_files:
    embed_single_pdf(pdf_name)
    save_checkpoint()  # Save progress
    clear_memory()
```

---

## Embedding Time Estimates

| Pages | GPU | Time |
|-------|-----|------|
| 100 | T4 (15GB) | ~5 min |
| 500 | L4 (24GB) | ~20 min |
| 1000 | L4 (24GB) | ~40 min |
| 5000 | A100 (40GB) | ~2 hours |

**Factors:**
- DPI (higher = slower)
- Batch size (larger = faster, but needs more VRAM)
- GPU speed

---

## Best Practices

### 1. Test Small First

Before embedding 100 documents, test with 1-2 documents:

```python
pdf_files = ["test_doc.pdf"]  # Just one for testing
```

Verify:
- Embeddings load correctly
- Metadata is accurate
- Images are readable

### 2. Checkpoint Long Runs

For large jobs (1000+ pages), save progress periodically:

```python
# After each PDF
torch.save(current_embeddings, f"checkpoint_{pdf_name}.pt")
```

### 3. Monitor GPU Memory

```python
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

If memory usage grows over time → memory leak → add cleanup code.

### 4. Verify Outputs

After embedding, verify:

```python
# Check embeddings shape
data = torch.load('embeddings.pt')
print(f"Shape: {data.shape}")  # Should be [num_pages, 256, 320]

# Check metadata count
import json
with open('metadata.json') as f:
    meta = json.load(f)
print(f"Pages: {len(meta)}")  # Should match num_pages
```

---

## Adding New Documents to Existing Index

If you already have embeddings and want to add more:

### Option A: Re-embed Everything

Simplest approach - just add new PDFs to the list and re-run.

**Pros:** Clean, consistent
**Cons:** Slow if you have many existing docs

### Option B: Append New Embeddings

Embed only new documents, then concatenate:

```python
# Load existing
existing = torch.load('embeddings.pt')
existing_meta = json.load(open('metadata.json'))

# Embed new docs
new_embeddings = embed_documents(new_pdfs)
new_meta = generate_metadata(new_pdfs)

# Combine
combined = torch.cat([existing, new_embeddings])
combined_meta = existing_meta + new_meta

# Save
torch.save(combined, 'embeddings.pt')
json.dump(combined_meta, open('metadata.json', 'w'))
```

**Pros:** Faster for incremental additions
**Cons:** Need to track indices carefully

---

## Cost Estimates

### Free Options

- Lightning.ai: Free credits available
- Google Colab: Free T4 GPU (limited hours)
- Kaggle: 30 hours/week free GPU

### Paid Options

- Lightning.ai (beyond free): ~$0.50/hour (L4)
- Paperspace: ~$0.50/hour (A4000)
- Vast.ai: ~$0.10-0.30/hour (varies by GPU)

**Example:** Embedding 5000 pages on paid L4:
- Time: ~2 hours
- Cost: ~$1.00

**Negligible cost** for a one-time operation.

---

## Docker Option (Advanced)

For reproducible embedding environment:

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    poppler-utils

RUN pip install torch torchvision transformers \
    pdf2image pillow pypdf tqdm

COPY embed_docs.py /app/
WORKDIR /app
```

```bash
# Build and run
docker build -t urban-rag-embedder .
docker run --gpus all -v ./docs:/docs -v ./data:/data urban-rag-embedder
```

---

## Summary: Recommended Workflow

**For most users:**

1. Use Lightning.ai (free GPU)
2. Upload `embed_docs.ipynb` notebook
3. Upload your PDFs
4. Run all cells
5. Download outputs
6. Place in `data/` directory

**Total time:** ~30-60 minutes including setup

**For developers:**

Consider creating an embedding API service where users upload PDFs and get back embeddings. This eliminates GPU requirement for end users.

---

## Questions?

**"Do I need to re-embed if I update a document?"**  
Yes, you'll need to re-embed that specific document or re-embed everything.

**"Can I use CPU instead of GPU?"**  
Technically yes, but it's painfully slow (4-8 hours for 738 pages). Not recommended.

**"What if I have 100+ documents?"**  
Same process, just more time. Consider using A100 GPU or splitting into batches.

**"Can I embed documents in other languages?"**  
Yes, ColQwen is multilingual. Works with Hindi, regional languages, etc.

---

## Next Steps

1. Choose your GPU option (Lightning.ai recommended)
2. Follow the setup steps
3. Run embedding pipeline
4. Download and verify outputs
5. Test retrieval with your new embeddings

For questions or issues, open an issue on GitHub.
