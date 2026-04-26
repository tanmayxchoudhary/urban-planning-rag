#!/usr/bin/env python3
"""
Qdrant Indexer for urban-planning-rag
Creates Qdrant collection from existing embeddings
"""
import json
import torch
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DATA_DIR = "/home/fedora/Projects/urban-planning-rag/data"
EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings/embeddings.pt"
METADATA_FILE = f"{DATA_DIR}/embeddings/metadata.json"
COLLECTION_NAME = "urban_planning"

def main():
    print("Loading embeddings...")
    embs = torch.load(EMBEDDINGS_FILE)
    # Shape: [738, 1271, 320] - take first token for doc-level
    if len(embs.shape) == 3:
        doc_embs = embs[:, 0, :].float().numpy()
    else:
        doc_embs = embs.float().numpy()
    
    print(f"Document embeddings shape: {doc_embs.shape}")
    
    print("Loading metadata...")
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} documents")
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    
    # Delete existing collection
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=doc_embs.shape[1],
            distance=Distance.COSINE
        )
    )
    print(f"Created collection: {COLLECTION_NAME}")
    
    # Prepare points
    points = []
    for i, (vec, meta) in enumerate(zip(doc_embs, metadata)):
        points.append(
            PointStruct(
                id=i,
                vector=vec.tolist(),
                payload={
                    "source": meta["source"],
                    "page": meta["page"],
                    "total_pages": meta["total_pages"],
                    "id": f"doc_{i}"
                }
            )
        )
    
    # Upload in batches
    print("Uploading vectors...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print(f"✓ Indexed {len(points)} documents in Qdrant")
    
    # Test query
    print("\nTesting search...")
    query_vector = doc_embs[0].tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    )
    print(f"Found {len(results.points)} results")
    for r in results.points:
        print(f"  - {r.payload['source']} p.{r.payload['page']} (score: {r.score:.3f})")

if __name__ == "__main__":
    main()
