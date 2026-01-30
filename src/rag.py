"""
Urban Planning RAG System

Complete retrieval-augmented generation pipeline for Indian urban planning documents.

Architecture:
- Embeddings: TomoroAI/tomoro-colqwen3-embed-4b (multi-vector, 4B params)
- Vector DB: ChromaDB (patch-level indexing)
- Retrieval: Two-stage pipeline (Multi-Query Expansion + MaxSim reranking)
- VLM: Gemini 3.0 Flash / 2.5 Flash (Google AI Studio API)

Key Features:
- Lazy embedding loading for fast startup
- Optimized ChromaDB indexing with streaming batches
- Parallel patch extraction for faster processing
- Cloud API ready for remote GPU inference
"""

import torch
import chromadb
import numpy as np
import json
from pathlib import Path
from google import genai
from PIL import Image
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import gc


class UrbanPlanningRAG:
    """
    Complete RAG system for urban planning documents.

    Uses visual document retrieval (ColQwen embeddings) + Gemini VLM for generation.

    Features:
    - Two-stage retrieval: Multi-Query Token Expansion → MaxSim reranking
    - ChromaDB patch-level indexing for late interaction
    - Detailed retrieval metrics and rank improvement tracking
    - Cloud API compatible for remote GPU inference

    Attributes:
        data_dir: Directory containing embeddings and page images
        embeddings_data: Pre-computed page embeddings (list of tensors)
        metadata: Page metadata (list of dicts)
        chroma_client: ChromaDB client for patch-level search
        collection: ChromaDB collection
        processor: ColQwen processor (loaded on-demand for query encoding)
        model: ColQwen model (loaded on-demand for query encoding)
        gemini_client: Gemini API client
        _last_retrieval_metrics: Cached metrics from last retrieval
    """

    def __init__(self, data_dir: str = "./data", load_query_encoder: bool = False):
        """
        Initialize RAG system.

        Args:
            data_dir: Path to data directory containing embeddings/ and page_images/
            load_query_encoder: If True, loads ColQwen model for query encoding (requires GPU)
                               If False, assumes query embeddings are provided externally
        """
        self.data_dir = Path(data_dir)
        self.embeddings_path = self.data_dir / "embeddings" / "embeddings.pt"
        self.metadata_path = self.data_dir / "embeddings" / "metadata.json"
        self.images_dir = self.data_dir / "page_images"
        self._last_retrieval_metrics = None
        self._embeddings_data = None  # Lazy load - only load when needed

        # Load environment variables
        load_dotenv()

        print("=" * 60)
        print("🚀 Initializing Urban Planning RAG System ")
        print("=" * 60)

        # Validate data files
        self._validate_data_files()

        # Load metadata
        print("\n📂 Loading metadata...")
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Initialize ChromaDB
        print("🗄️ Initializing ChromaDB...")
        self._init_chroma()

        # Embeddings loaded lazily for MaxSim reranking (saves startup time)
        print("💾 Embeddings configured for lazy loading (MaxSim reranking)")

        # Initialize query encoder (optional)
        self.processor = None
        self.model = None
        if load_query_encoder:
            print("📦 Loading ColQwen 4B for query encoding...")
            self._load_query_encoder()

        # Initialize Gemini
        print("🤖 Initializing Gemini VLM...")
        self._init_gemini()

        print(f"\n✅ RAG system ready with {len(self.metadata)} pages indexed")
        print("=" * 60)

    @property
    def embeddings_data(self):
        """Lazy load embeddings only when needed for MaxSim reranking."""
        if self._embeddings_data is None:
            print("  📂 Lazy loading embeddings for MaxSim reranking...")
            self._embeddings_data = torch.load(self.embeddings_path, map_location='cpu')
            print(f"  ✅ Loaded {len(self._embeddings_data)} page embeddings")
        return self._embeddings_data

    def _validate_data_files(self):
        """Validate that required data files exist"""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {self.embeddings_path}\n"
                f"Please run: python scripts/embed.py --docs-dir ./pdfs --output-dir ./data"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {self.metadata_path}\n"
                f"Please run embedding script to generate metadata"
            )

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Page images directory not found at {self.images_dir}\n"
                f"Please run embedding script with --save-images"
            )

    def _init_chroma(self):
        """Initialize ChromaDB with patch-level index ()."""
        db_path = self.data_dir / "chroma_db"
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))

        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="urban_planning_patches",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"  ⚠️ ChromaDB initialization warning: {e}")
            # Create fresh collection
            try:
                self.chroma_client.delete_collection("urban_planning_patches")
            except:
                pass
            self.collection = self.chroma_client.create_collection(
                name="urban_planning_patches",
                metadata={"hnsw:space": "cosine"}
            )

        if self.collection.count() == 0:
            print("  ⚠️ Index empty. Building patch-level index...")
            self._build_chroma_index()
        else:
            print(f"  ✅ Found existing index: {self.collection.count()} patches")

    def _build_chroma_index(self):
        """
        Build ChromaDB patch-level index from embeddings.pt ( - Optimized).

        Uses OptimizedChromaIndexer for 5-10x faster indexing.
        
        Key optimizations:
        - Single-pass embedding load (eliminates double-load)
        - Streaming batch insertion (constant memory)
        - Parallel patch extraction (ProcessPoolExecutor)
        - Tuned HNSW parameters
        """
        from .indexer_optimized import OptimizedChromaIndexer
        
        print("  🚀 Using optimized indexer ()...")
        indexer = OptimizedChromaIndexer(
            data_dir=self.data_dir,
            batch_size=50000,
            use_parallel=True
        )
        
        # Build new collection with optimized settings
        optimized_collection = indexer.build_index()
        
        # Replace collection reference
        self.collection = optimized_collection
        
        # Clear any cached embeddings to free memory
        gc.collect()

    def _load_query_encoder(self):
        """Load ColQwen 4B model for query encoding ()"""
        from transformers import AutoModel, AutoProcessor

        MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"  Loading {MODEL_ID} on {device}...")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            max_num_visual_tokens=1280
        )

        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            attn_implementation="sdpa",
            trust_remote_code=True,
            device_map=device
        ).eval()

        print(f"  ✅ ColQwen 4B loaded on {device}")
        if device == "cuda":
            print(f"  🔧 VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def _init_gemini(self):
        """Initialize Gemini client"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("  ⚠️ GEMINI_API_KEY not found. Generation disabled.")
            self.gemini_client = None
            return

        self.gemini_client = genai.Client(api_key=api_key)
        print(f"  ✅ Gemini client ready")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode text query using ColQwen ().

        Args:
            query: Natural language query

        Returns:
            Query embedding tensor (num_tokens × 320 numpy array)

        Raises:
            RuntimeError: If query encoder not loaded (set load_query_encoder=True)
        """
        if self.processor is None or self.model is None:
            raise RuntimeError(
                "Query encoder not loaded. Initialize RAG with load_query_encoder=True "
                "or provide pre-computed query embeddings to retrieve()"
            )

        # Process query
        features = self.processor.process_texts([query])
        device = self.model.device
        features = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }

        # Generate embedding
        with torch.inference_mode():
            out = self.model(**features)
            query_tensor = out.embeddings[0].cpu()

        return query_tensor.float().numpy()

    def compute_maxsim(self, query_tensor: torch.Tensor, doc_tensor: torch.Tensor) -> float:
        """
        Compute MaxSim (late interaction) score ().

        MaxSim(Q, D) = Sum_q( Max_d( q · d ) )

        For each query token, find best matching document patch, then sum.

        Args:
            query_tensor: Query embeddings (num_query_tokens × 320)
            doc_tensor: Document embeddings (num_patches × 320)

        Returns:
            MaxSim score (higher is better)
        """
        if isinstance(query_tensor, np.ndarray):
            query_tensor = torch.from_numpy(query_tensor)
        if isinstance(doc_tensor, np.ndarray):
            doc_tensor = torch.from_numpy(doc_tensor)

        # Ensure same dtype
        query_tensor = query_tensor.float()
        doc_tensor = doc_tensor.float()

        # Similarity matrix: (num_q_tokens, num_d_patches)
        sim_matrix = torch.matmul(query_tensor, doc_tensor.T)

        # Max over document patches for each query token
        max_scores_per_token, _ = sim_matrix.max(dim=1)

        # Sum over query tokens
        return max_scores_per_token.sum().item()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        n_candidates: int = 50,
        num_query_tokens: int = 3,
        show_metrics: bool = True
    ) -> List[Dict]:
        """
        Two-stage retrieval with multi-query expansion ().

        Stage 1: Multi-Query Token Expansion
        - Select top-k most distinctive query tokens by L2 norm
        - Query ChromaDB with each token
        - Union results, rank by token coverage + best patch rank

        Stage 2: MaxSim Late-Interaction Reranking
        - Compute MaxSim score for top n_candidates
        - Re-rank by MaxSim score

        Args:
            query: Natural language query
            top_k: Number of final pages to return (default 3)
            n_candidates: Number of candidates for Stage 2 reranking (default 50)
            num_query_tokens: Number of distinctive query tokens to use (default 3)
            show_metrics: Display detailed retrieval metrics (default True)

        Returns:
            List of dicts with keys: source, page, total_pages, image_path, similarity
        """
        print(f"\n🔍 Query: '{query}'")
        print(f"⚡ Stage 1: Multi-Query Token Expansion (top-{num_query_tokens} tokens)...")

        query_tensor_np = self.encode_query(query)

        # Multi-query expansion: Use top-k distinctive query tokens by L2 norm
        token_norms = np.linalg.norm(query_tensor_np, axis=1)
        top_token_indices = np.argsort(token_norms)[-num_query_tokens:]

        # Query ChromaDB with each distinctive token and union results
        candidate_doc_scores = {}  # doc_id -> list of (patch_rank, token_idx)

        for i, tok_idx in enumerate(top_token_indices):
            query_vec = query_tensor_np[tok_idx].tolist()
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=n_candidates * 3,  # Get more patches per token
                include=["metadatas", "distances"]
            )

            for rank, meta in enumerate(results['metadatas'][0]):
                doc_id = meta['doc_id']
                if doc_id not in candidate_doc_scores:
                    candidate_doc_scores[doc_id] = []
                candidate_doc_scores[doc_id].append((rank, tok_idx))

        # Rank candidates by: (1) how many tokens matched, (2) best rank across tokens
        candidate_rankings = []
        for doc_id, matches in candidate_doc_scores.items():
            num_tokens_matched = len(set(m[1] for m in matches))
            best_rank = min(m[0] for m in matches)
            # Score: prioritize multi-token matches, then best rank
            expansion_score = (num_tokens_matched * 1000) - best_rank
            candidate_rankings.append((doc_id, expansion_score, num_tokens_matched, best_rank))

        # Sort by expansion score and take top n_candidates
        candidate_rankings.sort(key=lambda x: x[1], reverse=True)
        candidate_rankings = candidate_rankings[:n_candidates]

        # Build stage 1 results with initial ranks
        stage1_results = []
        for initial_rank, (doc_id, exp_score, num_tok, best_r) in enumerate(candidate_rankings):
            item = self.metadata[doc_id]
            stage1_results.append({
                'doc_id': doc_id,
                'source': item['source'],
                'page': item['page'],
                'initial_rank': initial_rank + 1,
                'tokens_matched': num_tok,
                'best_patch_rank': best_r + 1
            })

        print(f"  Found {len(stage1_results)} candidate pages")
        print(f"  Token coverage: {sum(1 for r in stage1_results if r['tokens_matched'] >= 2)}/{len(stage1_results)} matched 2+ tokens")

        # Stage 2: MaxSim reranking
        print(f"🔥 Stage 2: MaxSim reranking...")

        q_tensor = torch.from_numpy(query_tensor_np)

        for result in stage1_results:
            d_tensor = self.embeddings_data[result['doc_id']]
            result['maxsim_score'] = self.compute_maxsim(q_tensor, d_tensor)

        # Sort by MaxSim score
        stage1_results.sort(key=lambda x: x['maxsim_score'], reverse=True)

        # Assign final ranks
        for final_rank, result in enumerate(stage1_results):
            result['final_rank'] = final_rank + 1
            result['rank_improvement'] = result['initial_rank'] - result['final_rank']

        # Store metrics for display
        self._last_retrieval_metrics = {
            'query': query,
            'stage1_candidates': stage1_results.copy(),
            'top_k_selected': stage1_results[:top_k]
        }

        # Display metrics if requested
        if show_metrics:
            self._display_retrieval_metrics(top_k)

        # Build final results
        final_results = []
        for result in stage1_results[:top_k]:
            item = self.metadata[result['doc_id']]
            source_name = item['source'].replace('.pdf', '').replace(' ', '_').lower()
            image_filename = f"{source_name}__page_{item['page']:04d}.png"

            final_results.append({
                'source': item['source'],
                'page': item['page'],
                'total_pages': item['total_pages'],
                'image_path': str(self.images_dir / image_filename),
                'similarity': result['maxsim_score']
            })

        return final_results

    def _display_retrieval_metrics(self, top_k: int):
        """Display detailed retrieval metrics ()."""
        metrics = self._last_retrieval_metrics
        all_candidates = metrics['stage1_candidates']
        selected = metrics['top_k_selected']

        print(f"\n{'='*70}")
        print(f"📊 RETRIEVAL METRICS")
        print(f"{'='*70}")

        # Selected pages with rank improvement
        print(f"\n🎯 Top {top_k} Selected (after MaxSim):")
        print(f"{'Rank':<6}{'Source':<40}{'Page':<8}{'InitRank':<10}{'Δ Rank':<10}{'MaxSim':<10}")
        print(f"{'-'*70}")
        for r in selected:
            delta = r['rank_improvement']
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            src_short = r['source'][:38] + '...' if len(r['source']) > 38 else r['source']
            print(f"{r['final_rank']:<6}{src_short:<40}{r['page']:<8}{r['initial_rank']:<10}{delta_str:<10}{r['maxsim_score']:.1f}")

        # Summary stats
        improvements = [r['rank_improvement'] for r in selected if r['rank_improvement'] > 0]
        if improvements:
            print(f"\n📈 Rank Improvements: {len(improvements)}/{top_k} pages improved")
            print(f"   Average improvement: +{np.mean(improvements):.1f} ranks")
            print(f"   Best improvement: +{max(improvements)} ranks")

        # Show where reranking helped most
        big_movers = [r for r in selected if r['rank_improvement'] >= 10]
        if big_movers:
            print(f"\n🚀 Big movers (improved 10+ ranks):")
            for r in big_movers:
                print(f"   {r['source']} p.{r['page']}: #{r['initial_rank']} → #{r['final_rank']}")

        print(f"{'='*70}\n")

    def show_all_candidates(self, top_n: int = 50):
        """
        Show all Stage 1 candidates with rankings ().

        Useful for understanding retrieval behavior and debugging.

        Args:
            top_n: Number of candidates to display (default 50)
        """
        if self._last_retrieval_metrics is None:
            print("❌ No retrieval metrics available. Run retrieve() first.")
            return

        all_candidates = self._last_retrieval_metrics['stage1_candidates'][:top_n]

        print(f"\n{'='*80}")
        print(f"📋 ALL {len(all_candidates)} CANDIDATES (Stage 1 → Stage 2)")
        print(f"{'='*80}")
        print(f"{'Init#':<7}{'Final#':<8}{'Δ':<8}{'Tokens':<8}{'MaxSim':<10}{'Source':<35}{'Page':<6}")
        print(f"{'-'*80}")

        for r in all_candidates:
            delta = r['rank_improvement']
            delta_str = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "0"
            src_short = r['source'][:33] + '...' if len(r['source']) > 33 else r['source']
            print(f"{r['initial_rank']:<7}{r['final_rank']:<8}{delta_str:<8}{r['tokens_matched']:<8}{r['maxsim_score']:.1f}{'':>4}{src_short:<35}{r['page']:<6}")

        print(f"{'='*80}\n")

    def answer_query(
        self,
        query: str,
        top_k: int = 3,
        model: str = 'gemini-3-flash-preview'
    ) -> str:
        """
        Complete RAG pipeline: Retrieve relevant pages + Generate answer ().

        Args:
            query: Natural language question
            top_k: Number of pages to retrieve (default 3)
            model: Gemini model to use ('gemini-3-flash-preview' or 'gemini-2.5-flash')

        Returns:
            Generated answer with citations
        """
        if self.gemini_client is None:
            return "❌ Gemini API not initialized. Set GEMINI_API_KEY in .env file."

        # Retrieve relevant pages
        retrieved = self.retrieve(query=query, top_k=top_k)

        print(f"\n📋 Retrieved pages (MaxSim scored):")
        for i, r in enumerate(retrieved, 1):
            print(f"  {i}. {r['source']} - Page {r['page']} (Score: {r['similarity']:.1f})")

        # Load page images
        print(f"\n🖼️ Loading page images...")
        page_images = []
        for r in retrieved:
            img_path = Path(r['image_path'])
            if img_path.exists():
                page_images.append(Image.open(img_path))
            else:
                print(f"  ⚠️ Image not found: {img_path}")

        if not page_images:
            return "❌ No valid page images found. Please check that page_images/ directory exists."

        # Generate answer with Gemini
        print(f"🤖 Generating answer with {model}...")

        prompt = f"""You are an expert in Indian urban planning regulations.

Question: {query}

I've provided {len(page_images)} relevant pages from planning documents. Please:
1. Answer the question based on the provided pages
2. Cite which page number contains the information
3. If the information is not in the provided pages, say so

Be concise and specific."""

        try:
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=[prompt] + page_images
            )
            return response.text

        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"


# Convenience function for quick usage
def create_rag(data_dir: str = "./data", load_query_encoder: bool = False):
    """
    Create and return a RAG instance ().

    Args:
        data_dir: Path to data directory
        load_query_encoder: Whether to load ColQwen for query encoding

    Returns:
        UrbanPlanningRAG instance
    """
    return UrbanPlanningRAG(data_dir=data_dir, load_query_encoder=load_query_encoder)
