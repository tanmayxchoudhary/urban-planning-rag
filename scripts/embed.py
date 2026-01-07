#!/usr/bin/env python3
"""
Embed urban planning documents using TomoroAI/tomoro-colqwen3-embed-8b

This script converts PDFs to page images and generates multi-vector embeddings.
Requires GPU with 16GB+ VRAM.

Usage:
    python scripts/embed.py --docs-dir ./docs --output-dir ./data

Output:
    - embeddings.pt: Multi-vector embeddings (256 patches × 320 dim per page)
    - metadata.json: Page metadata (source, page number)
    - page_images/: PNG images of each page (150 DPI)
"""

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
from pdf2image import convert_from_path
from pathlib import Path
import json
from tqdm import tqdm
import gc
import argparse
import sys


MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-8b"
DTYPE = torch.bfloat16


class DocumentEmbedder:
    """Embed PDF documents using ColQwen visual encoder"""

    def __init__(self, batch_size: int = 2, dpi: int = 150, device: str = "cuda"):
        """
        Initialize embedder.

        Args:
            batch_size: Number of pages to process at once (lower = less VRAM)
            dpi: Resolution for PDF to image conversion (higher = better quality)
            device: 'cuda' or 'cpu' (CPU is very slow, not recommended)
        """
        self.batch_size = batch_size
        self.dpi = dpi
        self.device = device

        if device == "cuda" and not torch.cuda.is_available():
            print("❌ CUDA not available. Falling back to CPU (this will be SLOW).")
            self.device = "cpu"

        # Load model
        self._load_model()

    def _load_model(self):
        """Load ColQwen model and processor"""
        print(f"📦 Loading model: {MODEL_ID}")
        print(f"   Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            max_num_visual_tokens=1280
        )

        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            attn_implementation="sdpa",  # PyTorch native attention (no flash-attn)
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

    def embed_pdf(self, pdf_path: Path):
        """
        Embed single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            embeddings: List of embedding tensors (one per page)
            metadata: List of metadata dicts (one per page)
        """
        print(f"📄 Processing: {pdf_path.name}")

        # Convert PDF to images
        print("  🖼️  Converting to images...")
        images = convert_from_path(str(pdf_path), dpi=self.dpi)
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

        # Create metadata
        metadata = [
            {
                "source": pdf_path.name,
                "page": idx + 1,
                "total_pages": len(images)
            }
            for idx in range(len(images))
        ]

        # Clear images
        del images
        gc.collect()

        if self.device == "cuda":
            print(f"  🔧 GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated\n")

        return embeddings, metadata

    def save_page_images(self, pdf_path: Path, output_dir: Path):
        """
        Save PDF pages as PNG images.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
        """
        images = convert_from_path(str(pdf_path), dpi=self.dpi)

        for idx, img in enumerate(images):
            filename = f"{pdf_path.stem}__page_{idx+1:04d}.png"
            img.save(output_dir / filename, "PNG", optimize=True)

        return len(images)


def embed_documents(
    docs_dir: Path,
    output_dir: Path,
    pdf_files: list = None,
    batch_size: int = 2,
    dpi: int = 150,
    save_images: bool = True
):
    """
    Main embedding pipeline.

    Args:
        docs_dir: Directory containing PDF files
        output_dir: Directory to save outputs
        pdf_files: List of PDF filenames to process (if None, process all PDFs)
        batch_size: Batch size for embedding
        dpi: DPI for image conversion
        save_images: Whether to save page images
    """
    print("=" * 60)
    print("🚀 Urban Planning RAG - Document Embedding")
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

    # Initialize embedder
    embedder = DocumentEmbedder(batch_size=batch_size, dpi=dpi)

    # Process each PDF
    all_embeddings = []
    all_metadata = []

    for pdf_name in pdf_files:
        pdf_path = docs_dir / pdf_name

        if not pdf_path.exists():
            print(f"⚠️  Skipping {pdf_name} (not found)")
            continue

        print("=" * 60)
        embeddings, metadata = embedder.embed_pdf(pdf_path)

        all_embeddings.extend(embeddings)
        all_metadata.extend(metadata)

        print(f"  ✅ Embedded {len(embeddings)} pages from {pdf_name}")
        print(f"  📊 Total pages: {len(all_embeddings)}\n")

    # Save embeddings
    print("=" * 60)
    print("💾 Saving embeddings...")

    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    embeddings_tensor = torch.stack(all_embeddings)
    torch.save(embeddings_tensor, embeddings_dir / "embeddings.pt")

    with open(embeddings_dir / "metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    file_size = embeddings_tensor.element_size() * embeddings_tensor.nelement() / 1e6

    print(f"✅ Saved embeddings:")
    print(f"   - {embeddings_dir / 'embeddings.pt'} ({file_size:.1f} MB)")
    print(f"   - {embeddings_dir / 'metadata.json'}")

    # Save page images
    if save_images:
        print("\n💾 Saving page images...")
        images_dir = output_dir / "page_images"
        images_dir.mkdir(exist_ok=True)

        total_images = 0
        for pdf_name in pdf_files:
            pdf_path = docs_dir / pdf_name
            if pdf_path.exists():
                count = embedder.save_page_images(pdf_path, images_dir)
                total_images += count

        print(f"✅ Saved {total_images} page images to {images_dir}")

    print("\n" + "=" * 60)
    print("✅ EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"📊 Total pages embedded: {len(all_embeddings)}")
    print(f"💾 Output directory: {output_dir}")
    print(f"\nShape: {embeddings_tensor.shape}")
    print(f"  - Pages: {embeddings_tensor.shape[0]}")
    print(f"  - Patches per page: {embeddings_tensor.shape[1]}")
    print(f"  - Embedding dimension: {embeddings_tensor.shape[2]}")


def main():
    parser = argparse.ArgumentParser(
        description="Embed PDF documents using ColQwen visual encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed all PDFs in docs/ directory
  python scripts/embed.py --docs-dir ./docs --output-dir ./data

  # Embed specific PDFs
  python scripts/embed.py --docs-dir ./docs --pdfs swm_2016.pdf urdpfi_vol1.pdf

  # Adjust batch size for memory constraints
  python scripts/embed.py --docs-dir ./docs --batch-size 1

  # Higher quality images
  python scripts/embed.py --docs-dir ./docs --dpi 200

  # Skip saving page images (faster)
  python scripts/embed.py --docs-dir ./docs --no-images
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
        default=2,
        help='Batch size for embedding (default: 2, lower if OOM)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for PDF to image conversion (default: 150)'
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
            dpi=args.dpi,
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
