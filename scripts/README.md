# Urban Planning RAG - Scripts

This directory contains the core scripts for the Urban Planning RAG system.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline
python scripts/pipeline.py --docs-dir ./pdfs --query "What is FSI?"

# Or run steps individually:
python scripts/pipeline.py --docs-dir ./pdfs --step embed
python scripts/pipeline.py --step index
python scripts/pipeline.py --query "Parking requirements" --step query
```

## Scripts Overview

### Main Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `embed.py` | Embed PDFs using ColQwen3-embed-4B | `python scripts/embed.py --docs-dir ./pdfs` |
| `pipeline.py` | Complete end-to-end pipeline | `python scripts/pipeline.py --query "..."` |
| `build_index.py` | Build ChromaDB index | Legacy, use pipeline.py instead |

### Supporting Scripts

| Script | Purpose |
|--------|---------|
| `check_docs.py` | Check PDF documents |
| `test_gemini.py` | Test Gemini API connection |
| `test_validation.py` | Validate embeddings |

### Archived Scripts

Old versioned scripts (embed_v21.py through embed_v30_lightning.py, etc.) are in the `archive/` directory.

## embed.py

Unified embedding script with comprehensive error handling.

```bash
# Basic usage
python scripts/embed.py --docs-dir ./pdfs --output-dir ./data

# Custom settings
python scripts/embed.py \
    --docs-dir ./pdfs \
    --batch-size 10 \
    --text-dpi 100 \
    --visual-dpi 250 \
    --verbose

# Skip saving images (faster)
python scripts/embed.py --docs-dir ./pdfs --no-images
```

### Features

- **Adaptive DPI**: Automatically detects text vs visual pages
  - Text-only pages: 100 DPI
  - Visual content: 250 DPI
- **PyMuPDF backend**: No poppler dependency
- **Batch processing**: Configurable batch size for memory management
- **Comprehensive logging**: Detailed progress and error messages

## pipeline.py

Complete end-to-end pipeline for embedding and querying.

```bash
# Full pipeline
python scripts/pipeline.py --docs-dir ./pdfs --query "What is FSI?"

# Check data status
python scripts/pipeline.py --check

# Individual steps
python scripts/pipeline.py --step embed
python scripts/pipeline.py --step index
python scripts/pipeline.py --query "..." --step query
```

## CLI Usage

After embedding, use the main CLI:

```bash
# Query the RAG system
python cli.py "What is FSI for residential zones?"

# With options
python cli.py --query "parking requirements" --top-k 5
python cli.py "open space standards" --model gemini-2.5-flash
```

## Error Handling

All scripts include comprehensive error handling:

- **Dependency checks**: Verify required packages are installed
- **File validation**: Check that PDFs and directories exist
- **Graceful degradation**: Continue processing if one PDF fails
- **Clear error messages**: Human-readable error descriptions

## Logging

Set log level with `--verbose` or `-v`:

```bash
python scripts/embed.py --docs-dir ./pdfs --verbose
python scripts/pipeline.py --query "..." --verbose
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (missing files, import errors, etc.) |
| 2 | Partial success (some PDFs failed) |
| 130 | Interrupted by user (Ctrl+C) |

## Version History

- **v3.0.0** (Current): Unified embed.py, consolidated 29 versions, improved error handling
- **v2.x**: Multiple versioned scripts (archived)
- **v1.x**: Original implementation (archived)
