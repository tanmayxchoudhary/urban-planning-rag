#!/usr/bin/env python3
"""
Quick ChromaDB Index Builder

Fast standalone script to build ChromaDB index without loading full RAG system.
Use this on cloud GPU servers for quick indexing.

Usage:
    python scripts/build_index.py [--data-dir ./data] [--batch-size 50000]

Performance:
    - Sequential mode: ~8-12 minutes for 738 pages
    - Parallel mode: ~3-5 minutes for 738 pages (multi-core)

Requirements:
    - Pre-generated embeddings.pt and metadata.json
    - chromadb package installed
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexer_optimized import optimize_chroma_index


def main():
    parser = argparse.ArgumentParser(
        description="Build optimized ChromaDB index for Urban Planning RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index with defaults (recommended)
  python scripts/build_index.py

  # Custom data directory
  python scripts/build_index.py --data-dir ./my_data

  # Disable parallel processing (if multiprocessing issues)
  python scripts/build_index.py --no-parallel

  # Adjust batch size for memory constraints
  python scripts/build_index.py --batch-size 25000
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('./data'),
        help='Directory containing embeddings/ subdirectory (default: ./data)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50000,
        help='Batch size for ChromaDB insertion (default: 50000)'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (use sequential mode)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Max parallel workers (default: CPU count - 1)'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not args.data_dir.exists():
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    embeddings_path = args.data_dir / "embeddings" / "embeddings.pt"
    metadata_path = args.data_dir / "embeddings" / "metadata.json"
    
    if not embeddings_path.exists():
        print(f"❌ Error: Embeddings not found: {embeddings_path}")
        print("   Run: python scripts/embed.py --docs-dir ./pdfs --output-dir ./data")
        sys.exit(1)
    
    if not metadata_path.exists():
        print(f"❌ Error: Metadata not found: {metadata_path}")
        sys.exit(1)
    
    # Build index
    print("="*60)
    print("🏗️  ChromaDB Index Builder v2.1.0")
    print("="*60)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"⚙️  Batch size: {args.batch_size:,}")
    print(f"⚙️  Parallel mode: {not args.no_parallel}\n")
    
    try:
        collection = optimize_chroma_index(
            data_dir=str(args.data_dir),
            batch_size=args.batch_size,
            use_parallel=not args.no_parallel,
            max_workers=args.max_workers
        )
        
        print(f"\n✅ Index build complete!")
        print(f"📊 Total patches indexed: {collection.count():,}")
        print(f"💾 Collection name: {collection.name}")
        
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
