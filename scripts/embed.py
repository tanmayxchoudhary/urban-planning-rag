#!/usr/bin/env python3
"""
Embed urban planning documents using TomoroAI/tomoro-colqwen3-embed-4b

v2.0.0 Features:
- Adaptive DPI: 100 DPI for text-only pages, 250 DPI for visual content
- PyMuPDF backend: No poppler dependency
- Efficient 4B model: 8GB VRAM (down from 16GB in v1.0.0)
- Variable-length embeddings: List of tensors for optimal storage

Usage:
    python scripts/embed.py --docs-dir ./docs --output-dir ./data

Output:
    - embeddings.pt: List of variable-length patch embeddings
    - metadata.json: Page metadata with DPI and page type
    - page_images/: PNG images at adaptive DPI
"""

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import fitz  # PyMuPDF
from pathlib import Path
import json
from tqdm import tqdm
import gc
import argparse
import sys
from dataclasses import dataclass
from typing import Literal, List


MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"
DTYPE = torch.bfloat16

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

    v2.0.0: Routes text-only pages (100 DPI) vs visual content (250 DPI)
    to optimize quality vs storage trade-off.
    
    Uses metadata-only detection (no rendering required):
    - has_images: Embedded images in page
    - num_drawings: Vector drawings (tables, flowcharts, diagrams)
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

    def classify_pdf(self, pdf_path: Path) -> List[PageInfo]:
        """
        Classify all pages in a PDF (instant, no rendering).

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PageInfo objects (one per page)
        """
        doc = fitz.open(pdf_path)
        results = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            results.append(self._classify_page(page, page_num + 1))
        doc.close()
        return results

    def _classify_page(self, page: fitz.Page, page_num: int) -> PageInfo:
        """Classify single page based on visual content (metadata only, no render)."""
        has_images = len(page.get_images(full=True)) > 0

        try:
            num_drawings = len(page.get_drawings())
        except:
            num_drawings = 0

        # Decision logic: visual if has images or many drawings
        if has_images or num_drawings > self.DRAWINGS_THRESHOLD:
            page_type = "HAS_VISUALS"
        else:
            page_type = "TEXT_ONLY"

        return PageInfo(
            page_num=page_num,
            page_type=page_type,
            dpi=self.visual_dpi if page_type == "HAS_VISUALS" else self.text_dpi,
            has_images=has_images,
            num_drawings=num_drawings
        )


class DocumentEmbedder:
    """Embed PDF documents using ColQwen visual encoder (v2.0.0)"""

    def __init__(self, batch_size: int = 20, device: str = "cuda"):
        """
        Initialize embedder.

        Args:
            batch_size: Number of pages to process at once (default 20 for 4B model)
            device: 'cuda' or 'cpu' (CPU is very slow, not recommended)
        """
        self.batch_size = batch_size
        self.device = device

        if device == "cuda" and not torch.cuda.is_available():
            print("❌ CUDA not available. Falling back to CPU (this will be SLOW).")
            self.device = "cpu"

        # Load model
        self._load_model()

    def _load_model(self):
        """Load ColQwen3-embed-4B model"""
        print(f"📦 Loading model: {MODEL_ID}")
        print(f"   Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            max_num_visual_tokens=1280
        )

        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE if self.device == 'cuda' else torch.float32,
            attn_implementation="sdpa",  # PyTorch native attention
            trust_remote_code=True,
            device_map=self.device,
        ).eval()

        print("✅ Model loaded")

        if self.device == "cuda":
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
            print(f"🔧 Total VRAM: {total_mem:.1f} GB")
            print(f"🔧 Model uses: {allocated:.2f} GB\n")

    def embed_pdf(self, pdf_path: Path, page_infos: List[PageInfo]):
        """
        Embed single PDF file with adaptive DPI.

        Args:
            pdf_path: Path to PDF file
            page_infos: Page classification results from PageClassifier

        Returns:
            embeddings: List of embedding tensors (variable patch counts)
            metadata: List of metadata dicts (one per page)
        """
        print(f"📄 Processing: {pdf_path.name}")

        # Convert PDF to images using PyMuPDF with adaptive DPI
        print("  🖼️  Converting with Adaptive DPI...")
        images = []

        doc = fitz.open(pdf_path)

        for info in tqdm(page_infos, desc="  Converting"):
            page = doc[info.page_num - 1]  # 0-indexed
            mat = fitz.Matrix(info.dpi / 72, info.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)

        doc.close()

        print(f"  ✅ {len(images)} pages converted")

        # Clear memory
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        # Embed in batches
        print(f"  🔮 Embedding (batch_size={self.batch_size})...")
        embeddings = []

        for start in tqdm(range(0, len(images), self.batch_size), desc="  Progress"):
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

        # Create metadata with v2.0.0 fields
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
            print(f"  🔧 GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated\n")

        return embeddings, metadata

    def save_page_images(self, pdf_path: Path, page_infos: List[PageInfo], output_dir: Path):
        """
        Save PDF pages as PNG images with adaptive DPI.

        Args:
            pdf_path: Path to PDF file
            page_infos: Page classification results
            output_dir: Directory to save images
        """
        doc = fitz.open(pdf_path)

        for info in page_infos:
            page = doc[info.page_num - 1]
            mat = fitz.Matrix(info.dpi / 72, info.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            filename = f"{pdf_path.stem}__page_{info.page_num:04d}.png"
            img.save(output_dir / filename, "PNG", optimize=True)

        doc.close()
        return len(page_infos)


def embed_documents(
    docs_dir: Path,
    output_dir: Path,
    pdf_files: list = None,
    batch_size: int = 20,
    text_dpi: int = 100,
    visual_dpi: int = 250,
    save_images: bool = True
):
    """
    Main embedding pipeline with adaptive DPI (v2.0.0).

    Args:
        docs_dir: Directory containing PDF files
        output_dir: Directory to save outputs
        pdf_files: List of PDF filenames to process (if None, process all PDFs)
        batch_size: Batch size for embedding (default 20 for 4B model)
        text_dpi: DPI for text-only pages (default 100)
        visual_dpi: DPI for pages with visuals (default 250)
        save_images: Whether to save page images
    """
    print("=" * 60)
    print("🚀 Urban Planning RAG - Document Embedding v2.0.0")
    print("=" * 60)

    # Find PDFs
    if pdf_files is None:
        pdf_files = sorted([f.name for f in docs_dir.glob("*.pdf")])

    if not pdf_files:
        print(f"❌ No PDF files found in {docs_dir}")
        sys.exit(1)

    print(f"📚 Found {len(pdf_files)} PDF(s):")
    for pdf in pdf_files:
        print(f"   - {pdf}")
    print()

    # Initialize classifier
    print("🔍 Classifying pages for Adaptive DPI...")
    classifier = PageClassifier(text_dpi=text_dpi, visual_dpi=visual_dpi)

    all_classifications = {}
    for pdf_name in pdf_files:
        pdf_path = docs_dir / pdf_name
        if not pdf_path.exists():
            print(f"⚠️  {pdf_name} not found, skipping")
            continue

        print(f"  📄 {pdf_name}")
        results = classifier.classify_pdf(pdf_path)
        all_classifications[pdf_name] = results

        text_count = sum(1 for r in results if r.page_type == "TEXT_ONLY")
        visual_count = sum(1 for r in results if r.page_type == "HAS_VISUALS")
        print(f"     📝 TEXT_ONLY: {text_count} pages @ {text_dpi} DPI")
        print(f"     📊 HAS_VISUALS: {visual_count} pages @ {visual_dpi} DPI")

    print(f"\n✅ Classified {sum(len(v) for v in all_classifications.values())} pages\n")

    # Initialize embedder
    embedder = DocumentEmbedder(batch_size=batch_size)

    # Process each PDF
    all_embeddings = []
    all_metadata = []

    for pdf_name in pdf_files:
        pdf_path = docs_dir / pdf_name

        if pdf_name not in all_classifications:
            print(f"⚠️  Skipping {pdf_name} (not classified)")
            continue

        print("=" * 60)
        embeddings, metadata = embedder.embed_pdf(pdf_path, all_classifications[pdf_name])

        all_embeddings.extend(embeddings)
        all_metadata.extend(metadata)

        print(f"  ✅ Embedded {len(embeddings)} pages from {pdf_name}")
        print(f"  📊 Total pages: {len(all_embeddings)}\n")

    # Save embeddings (v2.0.0: list of tensors, not stacked)
    print("=" * 60)
    print("💾 Saving embeddings...")

    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Save as list to preserve variable patch counts
    torch.save(all_embeddings, embeddings_dir / "embeddings.pt")

    with open(embeddings_dir / "metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Calculate statistics
    total_elements = sum(emb.nelement() for emb in all_embeddings)
    file_size = all_embeddings[0].element_size() * total_elements / 1e6
    patch_counts = [emb.shape[0] for emb in all_embeddings]

    print(f"✅ Saved embeddings:")
    print(f"   - {embeddings_dir / 'embeddings.pt'} ({file_size:.1f} MB)")
    print(f"   - {embeddings_dir / 'metadata.json'}")
    print(f"\n📊 Statistics:")
    print(f"   - Total pages: {len(all_embeddings)}")
    print(f"   - Patch counts: min={min(patch_counts)}, max={max(patch_counts)}, mean={sum(patch_counts)/len(patch_counts):.0f}")
    print(f"   - Embedding dim: {all_embeddings[0].shape[1]}")

    # Save page images
    if save_images:
        print("\n💾 Saving page images...")
        images_dir = output_dir / "page_images"
        images_dir.mkdir(exist_ok=True)

        total_images = 0
        for pdf_name in pdf_files:
            pdf_path = docs_dir / pdf_name
            if pdf_path.exists() and pdf_name in all_classifications:
                count = embedder.save_page_images(pdf_path, all_classifications[pdf_name], images_dir)
                total_images += count

        print(f"✅ Saved {total_images} page images to {images_dir}")

    print("\n" + "=" * 60)
    print("✅ EMBEDDING COMPLETE (v2.0.0)")
    print("=" * 60)
    print(f"📊 Total pages embedded: {len(all_embeddings)}")
    print(f"💾 Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Embed PDF documents using ColQwen3-embed-4B visual encoder (v2.0.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed all PDFs in docs/ directory
  python scripts/embed.py --docs-dir ./docs --output-dir ./data

  # Embed specific PDFs
  python scripts/embed.py --docs-dir ./docs --pdfs swm_2016.pdf urdpfi_vol1.pdf

  # Adjust batch size for memory constraints
  python scripts/embed.py --docs-dir ./docs --batch-size 10

  # Custom DPI settings
  python scripts/embed.py --docs-dir ./docs --text-dpi 80 --visual-dpi 300

  # Skip saving page images (faster)
  python scripts/embed.py --docs-dir ./docs --no-images

v2.0.0 Features:
  - Adaptive DPI: 100 DPI text, 250 DPI visuals (automatic)
  - PyMuPDF backend: No poppler dependency!         
  - ColQwen3-embed-4B: 8GB VRAM (down from 16GB in v1.0.0)
  - Variable patch counts: Optimal storage efficiency
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
        help='Batch size for embedding (default: 20 for 4B model, lower if OOM)'
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

    args = parser.parse_args()

    # Validate
    if not args.docs_dir.exists():
        print(f"❌ Error: docs-dir not found: {args.docs_dir}")
        sys.exit(1)

    # Run embedding
    try:
        embed_documents(
            docs_dir=args.docs_dir,
            output_dir=args.output_dir,
            pdf_files=args.pdfs,
            batch_size=args.batch_size,
            text_dpi=args.text_dpi,
            visual_dpi=args.visual_dpi,
            save_images=not args.no_images
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
