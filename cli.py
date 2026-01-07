#!/usr/bin/env python3
"""
Command-line interface for Urban Planning RAG system.

Usage:
    python cli.py "What is FSI for residential zones?"
    python cli.py --query "parking requirements" --top-k 5
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag import UrbanPlanningRAG


def main():
    parser = argparse.ArgumentParser(
        description="Urban Planning RAG - Query planning documents with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "What is FSI for residential zones?"
  python cli.py --query "parking requirements" --top-k 5
  python cli.py --query "open space standards" --model gemini-2.5-flash
        """
    )

    parser.add_argument(
        'query',
        nargs='?',
        help='Your question about urban planning regulations'
    )

    parser.add_argument(
        '--query', '-q',
        dest='query_alt',
        help='Alternative way to specify query'
    )

    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=3,
        help='Number of pages to retrieve (default: 3)'
    )

    parser.add_argument(
        '--model', '-m',
        default='gemini-3-flash-preview',
        choices=['gemini-3-flash-preview', 'gemini-2.5-flash'],
        help='Gemini model to use (default: gemini-3-flash-preview)'
    )

    parser.add_argument(
        '--data-dir',
        default='./data',
        help='Path to data directory (default: ./data)'
    )

    parser.add_argument(
        '--load-encoder',
        action='store_true',
        help='Force load ColQwen query encoder (requires 16GB RAM, will download model if not cached)'
    )

    parser.add_argument(
        '--retrieve-only',
        action='store_true',
        help='Only retrieve pages, don\'t generate answer'
    )

    args = parser.parse_args()

    # Get query from either positional or --query argument
    query = args.query or args.query_alt

    if not query:
        parser.print_help()
        print("\n❌ Error: Please provide a query")
        sys.exit(1)

    # Auto-detect if model is available (check cache, not GPU)
    import torch
    from pathlib import Path

    if args.load_encoder:
        # User explicitly wants encoder
        load_encoder = True
    else:
        # Check if ColQwen is in HuggingFace cache
        model_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--TomoroAI--tomoro-colqwen3-embed-8b"

        if model_cache.exists():
            # Model is cached, load it (will use GPU if available, CPU otherwise)
            device = "GPU" if torch.cuda.is_available() else "CPU"
            print(f"🔍 ColQwen model found in cache, loading on {device}")
            print("   (First time will be slower)\n")
            load_encoder = True
        else:
            # Model not cached
            print("⚠️  ColQwen model not found in cache")
            print("   Use --load-encoder to download and load (16GB model)\n")
            load_encoder = False

    try:
        # Initialize RAG system
        rag = UrbanPlanningRAG(
            data_dir=args.data_dir,
            load_query_encoder=load_encoder
        )

        if args.retrieve_only:
            # Just retrieve pages
            results = rag.retrieve(query=query, top_k=args.top_k)

            print("\n" + "=" * 60)
            print("📋 RETRIEVED PAGES")
            print("=" * 60)
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r['source']} - Page {r['page']}/{r['total_pages']}")
                print(f"   Similarity: {r['similarity']:.4f}")
                print(f"   Image: {r['image_path']}")

        else:
            # Full RAG pipeline
            answer = rag.answer_query(
                query=query,
                top_k=args.top_k,
                model=args.model
            )

            print("\n" + "=" * 60)
            print("📝 ANSWER")
            print("=" * 60)
            print(answer)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you have:")
        print("  1. Downloaded embeddings.pt and metadata.json")
        print("  2. Downloaded page_images/ directory")
        print("  3. Placed them in the correct data/ directory structure")
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
