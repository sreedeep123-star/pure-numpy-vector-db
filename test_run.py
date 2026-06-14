import numpy as np
from src.database import NumPyVectorDB
from src.hnsw import HNSWIndex

def run_metadata_test():
    DIMENSION = 4
    db = NumPyVectorDB(dimension=DIMENSION)
    
    # Define sample vectors with distinct metadata attributes
    items = [
        {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "meta": {"genre": "fantasy", "status": "active"}},
        {"id": 2, "vector": [0.9, 0.8, 0.7, 0.6], "meta": {"genre": "sci-fi", "status": "active"}},
        {"id": 3, "vector": [0.1, 0.15, 0.25, 0.35], "meta": {"genre": "fantasy", "status": "archived"}},
    ]
    
    print("📥 Ingesting items with custom metadata payloads...")
    for item in items:
        db.add_vectors(ids=[item["id"]], vectors=[item["vector"]], metadatas=[item["meta"]])
        
    query_vec = [0.1, 0.2, 0.3, 0.4]
    
    print("\n🔍 Query 1: Find closest vectors WITHOUT filter:")
    results_no_filter = db.query(query_vec, k=2)
    for v_id, score, meta in results_no_filter:
        print(f" -> ID: {v_id}, Distance: {score:.4f}, Meta: {meta}")
        
    print("\n🔍 Query 2: Find closest vectors filtered by {'genre': 'fantasy'}:")
    results_filtered = db.query(query_vec, k=2, where={"genre": "fantasy"})
    for v_id, score, meta in results_filtered:
        print(f" -> ID: {v_id}, Distance: {score:.4f}, Meta: {meta}")

if __name__ == "__main__":
    run_metadata_test()