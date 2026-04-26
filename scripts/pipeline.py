#!/usr/bin/env python3
"""
Urban Planning RAG - Complete Pipeline
======================================
End-to-end pipeline for embedding and querying urban planning documents.

This script orchestrates the full workflow:
1. Embed PDFs (if needed)
2. Build ChromaDB index (if needed)
3. Run queries against the RAG system

Usage:
    # Full pipeline
    python scripts/pipeline.py --docs-dir ./pdfs --query "What is FSI?"
    
    # Just embed
    python scripts/pipeline.py --docs-dir ./pdfs --step embed
    
    # Just query
    python scripts/pipeline.py --query "Parking requirements" --step query

Author: Urban Planning RAG Project
Version: 3.0.0
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

__version__ = "3.0.0"


def setup_environment() -> bool:
    """Check and setup the environment."""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available, using CPU")
        return True
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install -r requirements.txt")
        return False


def check_data_exists(data_dir: Path) -> Dict[str, bool]:
    """Check what data already exists."""
    embeddings_path = data_dir / "embeddings" / "embeddings.pt"
    metadata_path = data_dir / "embeddings" / "metadata.json"
    chroma_dir = data_dir / "chroma_db"
    images_dir = data_dir / "page_images"
    
    return {
        "embeddings": embeddings_path.exists(),
        "metadata": metadata_path.exists(),
        "chroma": chroma_dir.exists() and any(chroma_dir.iterdir()),
        "images": images_dir.exists() and any(images_dir.iterdir())
    }


def run_embedding(docs_dir: Path, output_dir: Path, batch_size: int = 20) -> bool:
    """Run the embedding step."""
    logger.info("=" * 60)
    logger.info("STEP 1: Embedding Documents")
    logger.info("=" * 60)
    
    try:
        # Import here to avoid circular imports
        sys.path.insert(0, str(Path(__file__).parent))
        from embed import embed_documents
        
        stats = embed_documents(
            docs_dir=docs_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            save_images=True
        )
        
        if stats.get("failed_pdfs"):
            logger.warning(f"Some PDFs failed: {stats['failed_pdfs']}")
        
        logger.info("✓ Embedding complete")
        return True
        
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return False


def run_indexing(data_dir: Path) -> bool:
    """Run the ChromaDB indexing step."""
    logger.info("=" * 60)
    logger.info("STEP 2: Building ChromaDB Index")
    logger.info("=" * 60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from indexer_optimized import optimize_chroma_index
        
        collection = optimize_chroma_index(
            data_dir=str(data_dir),
            batch_size=50000,
            use_parallel=True
        )
        
        logger.info(f"✓ Index built with {collection.count()} patches")
        return True
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return False


def run_query(
    data_dir: Path, 
    query: str, 
    top_k: int = 3,
    model: str = "gemini-3-flash-preview"
) -> Optional[str]:
    """Run a query against the RAG system."""
    logger.info("=" * 60)
    logger.info("STEP 3: Querying RAG System")
    logger.info("=" * 60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from rag import UrbanPlanningRAG
        
        rag = UrbanPlanningRAG(
            data_dir=str(data_dir),
            load_query_encoder=False  # Don't load heavy model for querying
        )
        
        # For now, just retrieve pages
        # Full answer generation would require query encoder + Gemini
        results = rag.retrieve(query=query, top_k=top_k)
        
        output = []
        output.append(f"\nQuery: '{query}'\n")
        output.append("Retrieved pages:\n")
        output.append("-" * 60)
        
        for i, r in enumerate(results, 1):
            output.append(f"\n{i}. {r['source']}")
            output.append(f"   Page: {r['page']}/{r['total_pages']}")
            output.append(f"   Score: {r['similarity']:.2f}")
            output.append(f"   Image: {r['image_path']}")
        
        result_str = "\n".join(output)
        logger.info(result_str)
        return result_str
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=f"Urban Planning RAG - Complete Pipeline (v{__version__})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: embed + index + query
  python scripts/pipeline.py --docs-dir ./pdfs --query "What is FSI?"
  
  # Just embed documents
  python scripts/pipeline.py --docs-dir ./pdfs --step embed
  
  # Just build index
  python scripts/pipeline.py --step index
  
  # Just query (requires existing data)
  python scripts/pipeline.py --query "Parking requirements" --step query
  
  # Check what data exists
  python scripts/pipeline.py --check
        """
    )
    
    parser.add_argument(
        '--docs-dir',
        type=Path,
        default=Path('./pdfs'),
        help='Directory containing PDF files (default: ./pdfs)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('./data'),
        help='Directory for data storage (default: ./data)'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Query to run against the RAG system'
    )
    
    parser.add_argument(
        '--step',
        choices=['all', 'embed', 'index', 'query'],
        default='all',
        help='Which step to run (default: all)'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check what data exists and exit'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Batch size for embedding (default: 20)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of results to retrieve (default: 3)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check data status first (doesn't require torch)
    data_status = check_data_exists(args.data_dir)
    
    if args.check:
        logger.info("Data status:")
        for key, exists in data_status.items():
            status = "✓" if exists else "✗"
            logger.info(f"  {status} {key}")
        sys.exit(0)
    
    # Check environment (only needed for actual operations)
    if not setup_environment():
        sys.exit(1)
    
    # Validate directories
    if args.step in ['all', 'embed'] and not args.docs_dir.exists():
        logger.error(f"Documents directory not found: {args.docs_dir}")
        sys.exit(1)
    
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Run requested steps
    success = True
    
    if args.step in ['all', 'embed']:
        if not data_status['embeddings'] or args.step == 'embed':
            success = run_embedding(args.docs_dir, args.data_dir, args.batch_size)
            if not success:
                logger.error("Embedding step failed")
                if args.step == 'embed':
                    sys.exit(1)
        else:
            logger.info("Embeddings already exist, skipping embedding step")
            logger.info("  (Use --step embed to force re-embedding)")
    
    if args.step in ['all', 'index']:
        if not data_status['chroma'] or args.step == 'index':
            success = run_indexing(args.data_dir)
            if not success:
                logger.error("Indexing step failed")
                if args.step == 'index':
                    sys.exit(1)
        else:
            logger.info("ChromaDB index already exists, skipping indexing")
            logger.info("  (Use --step index to force re-indexing)")
    
    if args.step in ['all', 'query']:
        if not args.query:
            logger.error("Query required for query step. Use --query")
            sys.exit(1)
        
        result = run_query(args.data_dir, args.query, args.top_k)
        if result is None:
            logger.error("Query step failed")
            sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("✅ Pipeline complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
