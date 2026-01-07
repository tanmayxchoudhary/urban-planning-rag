"""
Urban Planning RAG System
Complete retrieval-augmented generation pipeline for Indian urban planning documents.

Architecture:
- Embeddings: TomoroAI/tomoro-colqwen3-embed-8b (multi-vector)
- Vector DB: FAISS (cosine similarity with averaged embeddings)
- VLM: Gemini 3.0 Flash / 2.5 Flash (Google AI Studio API)
"""

import torch
import faiss
import numpy as np
import json
from pathlib import Path
from google import genai
from PIL import Image
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv


class UrbanPlanningRAG:
    """
    Complete RAG system for urban planning documents.
    
    Uses visual document retrieval (ColQwen embeddings) + Gemini VLM for generation.
    
    Attributes:
        data_dir: Directory containing embeddings and page images
        embeddings_data: Pre-computed page embeddings (torch.Tensor)
        metadata: Page metadata (list of dicts)
        index: FAISS index for similarity search
        processor: ColQwen processor (loaded on-demand for query encoding)
        model: ColQwen model (loaded on-demand for query encoding)
        gemini_client: Gemini API client
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
        
        # Load environment variables
        load_dotenv()
        
        print("=" * 60)
        print("🚀 Initializing Urban Planning RAG System")
        print("=" * 60)
        
        # Validate data files
        self._validate_data_files()
        
        # Load embeddings and metadata
        print("\n📂 Loading embeddings...")
        self.embeddings_data = torch.load(self.embeddings_path, map_location='cpu')
        
        print("📂 Loading metadata...")
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Build FAISS index
        print("🗄️ Building FAISS index...")
        self._build_faiss_index()
        
        # Initialize query encoder (optional)
        self.processor = None
        self.model = None
        if load_query_encoder:
            print("📦 Loading ColQwen for query encoding...")
            self._load_query_encoder()
        
        # Initialize Gemini
        print("🤖 Initializing Gemini VLM...")
        self._init_gemini()
        
        print(f"\n✅ RAG system ready with {len(self.metadata)} pages indexed")
        print("=" * 60)
    
    def _validate_data_files(self):
        """Validate that required data files exist"""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {self.embeddings_path}\n"
                f"Please download embeddings.pt and place it in {self.embeddings_path.parent}"
            )
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {self.metadata_path}\n"
                f"Please download metadata.json and place it in {self.metadata_path.parent}"
            )
        
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Page images directory not found at {self.images_dir}\n"
                f"Please download page_images/ and place it in {self.data_dir}"
            )
    
    def _build_faiss_index(self):
        """Build FAISS index from embeddings"""
        embeddings_list = []
        
        for idx in range(len(self.metadata)):
            # Get embedding for this page (shape: [256 patches, 320 dim])
            page_embedding = self.embeddings_data[idx].float()  # Convert BFloat16 to Float32
            
            # Average pool across patches to get single vector
            avg_embedding = page_embedding.mean(dim=0).numpy()
            embeddings_list.append(avg_embedding)
        
        # Stack into matrix
        embeddings_matrix = np.vstack(embeddings_list).astype('float32')
        self.embedding_dim = embeddings_matrix.shape[1]
        
        # Create FAISS index for cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Add to index
        self.index.add(embeddings_matrix)
        
        print(f"  ✅ Indexed {self.index.ntotal} pages (embedding_dim={self.embedding_dim})")
    
    def _load_query_encoder(self):
        """Load ColQwen model for query encoding (requires GPU)"""
        from transformers import AutoModel, AutoProcessor
        
        MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-8b"
        
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            max_num_visual_tokens=1280,
        )
        
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",  # Use PyTorch's native attention
            trust_remote_code=True,
            device_map=device,
        ).eval()
        
        print(f"  ✅ ColQwen loaded on {device}")
    
    def _init_gemini(self):
        """Initialize Gemini client"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment.\n"
                "Please create a .env file with: GEMINI_API_KEY=your-key-here"
            )
        
        self.gemini_client = genai.Client(api_key=api_key)
        print(f"  ✅ Gemini client ready")
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode text query using ColQwen.
        
        Args:
            query: Natural language query
            
        Returns:
            Query embedding vector (320-dim numpy array)
            
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        features = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }
        
        # Generate embedding
        with torch.inference_mode():
            out = self.model(**features)
            query_vec = out.embeddings[0].float().mean(dim=0).cpu().numpy()
        
        return query_vec
    
    def retrieve(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieve top-k relevant pages.
        
        Args:
            query: Natural language query (requires query encoder)
            query_embedding: Pre-computed query embedding (320-dim vector)
            top_k: Number of pages to retrieve
            
        Returns:
            List of dicts with keys: source, page, total_pages, image_path, similarity
            
        Raises:
            ValueError: If neither query nor query_embedding provided
        """
        if query is None and query_embedding is None:
            raise ValueError("Must provide either 'query' or 'query_embedding'")
        
        # Encode query if text provided
        if query is not None:
            print(f"\n🔍 Query: '{query}'")
            print(f"📊 Retrieving top {top_k} pages...")
            query_vec = self.encode_query(query)
        else:
            query_vec = query_embedding
        
        # Normalize query for cosine similarity
        query_norm = query_vec.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_norm)
        
        # Search FAISS index
        distances, indices = self.index.search(query_norm, top_k)
        
        # Format results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.metadata):
                item = self.metadata[idx]
                
                # Construct image filename
                source_name = item['source'].replace('.pdf', '').replace(' ', '_').lower()
                image_filename = f"{source_name}__page_{item['page']:04d}.png"
                
                results.append({
                    'source': item['source'],
                    'page': item['page'],
                    'total_pages': item['total_pages'],
                    'image_path': str(self.images_dir / image_filename),
                    'similarity': float(distances[0][i])
                })
        
        return results
    
    def answer_query(
        self,
        query: str,
        top_k: int = 3,
        model: str = 'gemini-3-flash-preview'
    ) -> str:
        """
        Complete RAG pipeline: Retrieve relevant pages + Generate answer.
        
        Args:
            query: Natural language question
            top_k: Number of pages to retrieve
            model: Gemini model to use ('gemini-3-flash-preview' or 'gemini-2.5-flash')
            
        Returns:
            Generated answer with citations
        """
        # Retrieve relevant pages
        retrieved = self.retrieve(query=query, top_k=top_k)
        
        print(f"\n📋 Retrieved pages:")
        for i, r in enumerate(retrieved, 1):
            print(f"  {i}. {r['source']} - Page {r['page']} (similarity: {r['similarity']:.3f})")
        
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
    Create and return a RAG instance.
    
    Args:
        data_dir: Path to data directory
        load_query_encoder: Whether to load ColQwen for query encoding
        
    Returns:
        UrbanPlanningRAG instance
    """
    return UrbanPlanningRAG(data_dir=data_dir, load_query_encoder=load_query_encoder)
