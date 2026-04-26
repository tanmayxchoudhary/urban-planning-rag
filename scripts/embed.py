#!/usr/bin/env python3
"""
Urban Planning RAG - Unified Embedding Script
=============================================
Single, clean embedding solution for Indian urban planning documents.

Features:
- Adaptive DPI: 100 DPI for text, 250 DPI for visual content
- PyMuPDF backend (no poppler dependency)
- ColQwen3-embed-4B model (8GB VRAM)
- Variable-length patch embeddings for optimal storage
- Comprehensive error handling and logging

Usage:
    python scripts/embed.py --docs-dir ./pdfs --output-dir ./data
    python scripts/embed.py --docs-dir ./pdfs --batch-size 10
    python scripts/embed.py --docs-dir ./pdfs --text-dpi 100 --visual-dpi 250

Author: Urban Planning RAG Project
Version: 3.0.0
"""

import argparse
import gc
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, List, Dict, Tuple, Optional

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Version info
__version__ = "3.0.0"


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are installed."""
    missing = []
    optional_missing = []
    
    # Required dependencies
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'fitz': 'PyMuPDF',
        'tqdm': 'tqdm',
    }
    
    # Optional dependencies
    optional = {
        'numpy': 'NumPy',
        'chromadb': 'ChromaDB',
    }
    
    for module, name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    for module, name in optional.items():
        try:
            __import__(module)
        except ImportError:
            optional_missing.append(name)
    
    if optional_missing:
        logger.warning(f"Optional dependencies missing: {', '.join(optional_missing)}")
    
    return len(missing) == 0, missing


# Import dependencies after check
try:
    import torch
    from transformers import AutoModel, AutoProcessor
    from PIL import Image
    import fitz  # PyMuPDF
    from tqdm import tqdm
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    logger.error(f"Failed to import dependencies: {e}")


# Constants
MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"

# Only define DTYPE if torch is available
DTYPE = torch.bfloat16 if HAS_DEPS and torch.cuda.is_available() else None

PageType = Literal["TEXT_ONLY", "HAS_VISUALS"]


@dataclass
class PageInfo:
    """Page classification result for adaptive DPI routing."""
    page_num: int
    page_type: PageType
    dpi: int
    has_images: bool
    num_drawings: int


class PageClassifier:
    """
    Classify PDF pages for adaptive DPI embedding.
    
    Routes text-only pages (100 DPI) vs visual content (250 DPI)
    to optimize quality vs storage trade-off.
    """

    DRAWINGS_THRESHOLD = 40

    def __init__(self, text_dpi: int = 100, visual_dpi: int = 250):
        """
        Initialize classifier.
        
        Args:
            text_dpi: DPI for text-only pages (default 100)
            visual_dpi: DPI for pages with visuals (default 250)
        """
        self.text_dpi = text_dpi
        self.visual_dpi = visual_dpi
        logger.info(f"PageClassifier: text_dpi={text_dpi}, visual_dpi={visual_dpi}")

    def classify_pdf(self, pdf_path: Path) -> List[PageInfo]:
        """
        Classify all pages in a PDF (instant, no rendering).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PageInfo objects (one per page)
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            RuntimeError: If PDF is corrupted or unreadable
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF {pdf_path}: {e}")
        
        results = []
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                results.append(self._classify_page(page, page_num + 1))
        finally:
            doc.close()
        
        return results

    def _classify_page(self, page: fitz.Page, page_num: int) -> PageInfo:
        """Classify single page based on visual content (metadata only, no render)."""
        try:
            has_images = len(page.get_images(full=True)) > 0
        except Exception:
            has_images = False
        
        try:
            num_drawings = len(page.get_drawings())
        except Exception:
            num_drawings = 0

        # Decision logic: visual if has images or many drawings
        if has_images or num_drawings > self.DRAWINGS_THRESHOLD:
            page_type = "HAS_VISUALS"
            dpi = self.visual_dpi
        else:
            page_type = "TEXT_ONLY"
            dpi = self.text_dpi

        return PageInfo(
            page_num=page_num,
            page_type=page_type,
            dpi=dpi,
            has_images=has_images,
            num_drawings=num_drawings
        )


class DocumentEmbedder:
    """Embed PDF documents using ColQwen visual encoder."""

    def __init__(self, batch_size: int = 20, device: Optional[str] = None):
        """
        Initialize embedder.
        
        Args:
            batch_size: Number of pages to process at once (default 20)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU (this will be SLOW).")
            self.device = "cpu"
        
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load ColQwen3-embed-4B model with error handling."""
        logger.info(f"Loading model: {MODEL_ID}")
        logger.info(f"Device: {self.device}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                max_num_visual_tokens=1280
            )
            logger.info("✓ Processor loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load processor: {e}")
        
        try:
            dtype = DTYPE if self.device == 'cuda' else torch.float32
            self.model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                trust_remote_code=True,
                device_map=self.device,
            ).eval()
            logger.info("✓ Model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        if self.device == "cuda":
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {allocated:.2f} GB / {total_mem:.1f} GB")

    def embed_pdf(
        self, 
        pdf_path: Path, 
        page_infos: List[PageInfo], 
        images_dir: Optional[Path] = None
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        Embed single PDF file with adaptive DPI.
        
        Args:
            pdf_path: Path to PDF file
            page_infos: Page classification results from PageClassifier
            images_dir: Optional directory to save page images
            
        Returns:
            Tuple of (embeddings list, metadata list)
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            RuntimeError: If processing fails
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing: {pdf_path.name}")
        
        # Convert PDF to images using PyMuPDF with adaptive DPI
        save_images = images_dir is not None
        images = []
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF {pdf_path}: {e}")
        
        try:
            desc = "Converting pages" if not save_images else "Converting + saving"
            for info in tqdm(page_infos, desc=desc, unit="page"):
                page = doc[info.page_num - 1]  # 0-indexed
                mat = fitz.Matrix(info.dpi / 72, info.dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)
                
                # Save image during conversion (no re-render needed later)
                if save_images:
                    filename = f"{pdf_path.stem}__page_{info.page_num:04d}.png"
                    img.save(images_dir / filename, "PNG", optimize=True)
        finally:
            doc.close()
        
        logger.info(f"✓ {len(images)} pages converted" + (" + saved" if save_images else ""))
        
        # Clear memory
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        
        # Embed in batches
        logger.info(f"Embedding (batch_size={self.batch_size})...")
        embeddings = []
        
        try:
            for start in tqdm(range(0, len(images), self.batch_size), desc="Batches", unit="batch"):
                batch_imgs = images[start : start + self.batch_size]
                
                # Process batch
                features = self.processor.process_images(images=batch_imgs)
                features = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in features.items()
                }
                
                # Generate embeddings
                with torch.inference_mode():
                    out = self.model(**features)
                    vecs = out.embeddings.to(torch.bfloat16).cpu()
                
                embeddings.extend(vecs)
                
                # Clear memory after batch
                del features, out, batch_imgs
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        except Exception as e:
            raise RuntimeError(f"Failed during embedding: {e}")
        
        # Create metadata
        metadata = [
            {
                "source": pdf_path.name,
                "page": info.page_num,
                "total_pages": len(images),
                "dpi": info.dpi,
                "page_type": info.page_type
            }
            for info in page_infos
        ]
        
        # Clear images
        del images
        gc.collect()
        
        if self.device == "cuda":
            logger.debug(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        
        return embeddings, metadata


def embed_documents(
    docs_dir: Path,
    output_dir: Path,
    pdf_files: Optional[List[str]] = None,
    batch_size: int = 20,
    text_dpi: int = 100,
    visual_dpi: int = 250,
    save_images: bool = True
) -> Dict:
    """
    Main embedding pipeline with adaptive DPI.
    
    Args:
        docs_dir: Directory containing PDF files
        output_dir: Directory to save outputs
        pdf_files: List of PDF filenames to process (if None, process all PDFs)
        batch_size: Batch size for embedding (default 20)
        text_dpi: DPI for text-only pages (default 100)
        visual_dpi: DPI for pages with visuals (default 250)
        save_images: Whether to save page images
        
    Returns:
        Dictionary with statistics about the embedding process
        
    Raises:
        FileNotFoundError: If docs_dir doesn't exist
        ValueError: If no PDFs found
        RuntimeError: If embedding fails
    """
    logger.info("=" * 60)
    logger.info(f"🚀 Urban Planning RAG - Document Embedding v{__version__}")
    logger.info("=" * 60)
    
    # Validate docs_dir
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    # Find PDFs
    if pdf_files is None:
        pdf_files = sorted([f.name for f in docs_dir.glob("*.pdf")])
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {docs_dir}")
    
    logger.info(f"Found {len(pdf_files)} PDF(s):")
    for pdf in pdf_files:
        logger.info(f"  - {pdf}")
    
    # Initialize classifier
    logger.info("Classifying pages for Adaptive DPI...")
    classifier = PageClassifier(text_dpi=text_dpi, visual_dpi=visual_dpi)
    
    all_classifications = {}
    total_text = 0
    total_visual = 0
    
    for pdf_name in pdf_files:
        pdf_path = docs_dir / pdf_name
        if not pdf_path.exists():
            logger.warning(f"{pdf_name} not found, skipping")
            continue
        
        logger.info(f"Classifying: {pdf_name}")
        try:
            results = classifier.classify_pdf(pdf_path)
            all_classifications[pdf_name] = results
            
            text_count = sum(1 for r in results if r.page_type == "TEXT_ONLY")
            visual_count = sum(1 for r in results if r.page_type == "HAS_VISUALS")
            total_text += text_count
            total_visual += visual_count
            logger.info(f"  TEXT_ONLY: {text_count} pages @ {text_dpi} DPI")
            logger.info(f"  HAS_VISUALS: {visual_count} pages @ {visual_dpi} DPI")
        except Exception as e:
            logger.error(f"Failed to classify {pdf_name}: {e}")
            continue
    
    total_pages = sum(len(v) for v in all_classifications.values())
    logger.info(f"✓ Classified {total_pages} pages ({total_text} text, {total_visual} visual)")
    
    # Initialize embedder
    embedder = DocumentEmbedder(batch_size=batch_size)
    
    # Prepare images directory if saving
    images_dir = None
    if save_images:
        images_dir = output_dir / "page_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Page images will be saved to: {images_dir}")
    
    # Process each PDF
    all_embeddings = []
    all_metadata = []
    failed_pdfs = []
    
    for pdf_name in pdf_files:
        pdf_path = docs_dir / pdf_name
        
        if pdf_name not in all_classifications:
            logger.warning(f"Skipping {pdf_name} (not classified)")
            failed_pdfs.append(pdf_name)
            continue
        
        logger.info("=" * 60)
        try:
            embeddings, metadata = embedder.embed_pdf(
                pdf_path, 
                all_classifications[pdf_name], 
                images_dir
            )
            all_embeddings.extend(embeddings)
            all_metadata.extend(metadata)
            logger.info(f"✓ Embedded {len(embeddings)} pages from {pdf_name}")
        except Exception as e:
            logger.error(f"Failed to embed {pdf_name}: {e}")
            failed_pdfs.append(pdf_name)
            continue
    
    # Save embeddings
    logger.info("=" * 60)
    logger.info("Saving embeddings...")
    
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save(all_embeddings, embeddings_dir / "embeddings.pt")
        with open(embeddings_dir / "metadata.json", "w") as f:
            json.dump(all_metadata, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save embeddings: {e}")
    
    # Calculate statistics
    if all_embeddings:
        total_elements = sum(emb.nelement() for emb in all_embeddings)
        file_size_mb = all_embeddings[0].element_size() * total_elements / 1e6
        patch_counts = [emb.shape[0] for emb in all_embeddings]
        
        stats = {
            "version": __version__,
            "total_pages": len(all_embeddings),
            "text_pages": total_text,
            "visual_pages": total_visual,
            "patch_counts": {
                "min": min(patch_counts),
                "max": max(patch_counts),
                "mean": sum(patch_counts) / len(patch_counts)
            },
            "embedding_dim": all_embeddings[0].shape[1],
            "file_size_mb": file_size_mb,
            "failed_pdfs": failed_pdfs
        }
        
        logger.info("✓ Saved embeddings:")
        logger.info(f"  - {embeddings_dir / 'embeddings.pt'} ({file_size_mb:.1f} MB)")
        logger.info(f"  - {embeddings_dir / 'metadata.json'}")
        logger.info(f"\nStatistics:")
        logger.info(f"  Total pages: {stats['total_pages']}")
        logger.info(f"  Patch counts: min={stats['patch_counts']['min']}, "
                   f"max={stats['patch_counts']['max']}, "
                   f"mean={stats['patch_counts']['mean']:.0f}")
        logger.info(f"  Embedding dim: {stats['embedding_dim']}")
        
        if save_images:
            logger.info(f"✓ Page images saved to: {images_dir}")
        
        if failed_pdfs:
            logger.warning(f"Failed PDFs: {failed_pdfs}")
    else:
        stats = {"error": "No embeddings generated"}
        logger.error("No embeddings were generated!")
    
    logger.info("=" * 60)
    logger.info("✅ EMBEDDING COMPLETE")
    logger.info("=" * 60)
    
    return stats


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description=f"Embed PDF documents using ColQwen3-embed-4B visual encoder (v{__version__})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Embed all PDFs in docs/ directory
  python scripts/embed.py --docs-dir ./pdfs --output-dir ./data

  # Embed specific PDFs
  python scripts/embed.py --docs-dir ./pdfs --pdfs swm_2016.pdf urdpfi_vol1.pdf

  # Adjust batch size for memory constraints
  python scripts/embed.py --docs-dir ./pdfs --batch-size 10

  # Custom DPI settings
  python scripts/embed.py --docs-dir ./pdfs --text-dpi 80 --visual-dpi 300

  # Skip saving page images (faster)
  python scripts/embed.py --docs-dir ./pdfs --no-images

v{__version__} Features:
  - Adaptive DPI: 100 DPI text, 250 DPI visuals (automatic)
  - PyMuPDF backend: No poppler dependency
  - ColQwen3-embed-4B: 8GB VRAM (down from 16GB in v1.0.0)
  - Variable patch counts: Optimal storage efficiency
  - Comprehensive error handling and logging
        """
    )

    parser.add_argument(
        '--docs-dir',
        type=Path,
        required=True,
        help='Directory containing PDF files'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./data'),
        help='Directory to save outputs (default: ./data)'
    )

    parser.add_argument(
        '--pdfs',
        nargs='+',
        help='Specific PDF files to process (default: all PDFs in docs-dir)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Batch size for embedding (default: 20)'
    )

    parser.add_argument(
        '--text-dpi',
        type=int,
        default=100,
        help='DPI for text-only pages (default: 100)'
    )

    parser.add_argument(
        '--visual-dpi',
        type=int,
        default=250,
        help='DPI for pages with visuals (default: 250)'
    )

    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Skip saving page images (only save embeddings)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()
    
    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    ok, missing = check_dependencies()
    if not ok:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Validate
    if not args.docs_dir.exists():
        logger.error(f"Error: docs-dir not found: {args.docs_dir}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run embedding
    try:
        stats = embed_documents(
            docs_dir=args.docs_dir,
            output_dir=args.output_dir,
            pdf_files=args.pdfs,
            batch_size=args.batch_size,
            text_dpi=args.text_dpi,
            visual_dpi=args.visual_dpi,
            save_images=not args.no_images
        )
        
        # Exit with error code if there were failures
        if stats.get('failed_pdfs'):
            sys.exit(2)
            
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(130)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
