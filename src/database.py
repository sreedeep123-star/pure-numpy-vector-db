import numpy as np
from typing import List, Tuple, Dict

class NumPyVectorDB:
    def __init__(self, dimension: int):
        """
        Initializes a pure NumPy exact vector database.
        
        :param dimension: The dimensionality of the vectors (e.g., 128, 768).
        """
        self.dimension = dimension
        # Pre-allocate an empty 2D float32 array for row-major cache contiguous layout
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        # Keeps track of external IDs matching the rows in self.vectors
        self.vector_ids: List[int] = []
        # Mapping for metadata payload storage
        self.metadata: Dict[int, dict] = {}

    def add_vectors(self, ids: List[int], vectors: List[List[float]], metadatas: List[dict] = None):
        """Adds a batch of vectors to the contiguous NumPy matrix storage."""
        np_vecs = np.array(vectors, dtype=np.float32)
        
        if np_vecs.ndim == 1:
            np_vecs = np_vecs.reshape(1, -1)
            
        if np_vecs.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {np_vecs.shape[1]}")
            
        if len(ids) != len(np_vecs):
            raise ValueError("The number of IDs must match the number of vectors.")

        # Vertically stack new vectors to maintain sequential memory layout
        self.vectors = np.vstack([self.vectors, np_vecs])
        self.vector_ids.extend(ids)
        
        if metadatas:
            for idx, meta in zip(ids, metadatas):
                self.metadata[idx] = meta

    def _compute_l2(self, query: np.ndarray) -> np.ndarray:
        """Vectorized row-wise L2 Euclidean Distance calculation."""
        # targets - query utilizes NumPy broadcasting
        return np.linalg.norm(self.vectors - query, axis=1)

    def _compute_cosine(self, query: np.ndarray) -> np.ndarray:
        """Vectorized row-wise Cosine Similarity calculation."""
        dot_products = np.dot(self.vectors, query)
        query_norm = np.linalg.norm(query)
        matrix_norms = np.linalg.norm(self.vectors, axis=1)
        
        # 1e-9 safety factor prevents division by zero for unnormalized vectors
        return dot_products / (matrix_norms * query_norm + 1e-9)

    def query(self, query_vector: List[float], k: int = 5, metric: str = "l2", where: dict = None) -> List[Tuple[int, float, dict]]:
        """
        Performs a brute-force exact linear scan with optional metadata pre-filtering.
        """
        if len(self.vectors) == 0:
            return []
            
        # 1. Apply Pre-Filtering based on metadata attributes
        valid_matrix_indices = []
        for matrix_idx, v_id in enumerate(self.vector_ids):
            if where:
                # Check if the stored metadata matches all conditions in the 'where' dict
                meta = self.metadata.get(v_id, {})
                match = all(meta.get(key) == val for key, val in where.items())
                if not match:
                    continue # Skip this vector if it doesn't match the filter
            valid_matrix_indices.append(matrix_idx)
            
        if not valid_matrix_indices:
            return [] # No vectors matched the metadata filter

        # Slice our matrix to only include valid, filtered vectors
        filtered_vectors = self.vectors[valid_matrix_indices]
        filtered_ids = [self.vector_ids[idx] for idx in valid_matrix_indices]

        query_np = np.array(query_vector, dtype=np.float32)
        
        # 2. Compute distances only on the filtered subsets
        if metric == "l2":
            # Vectorized row-wise calculation against filtered matrix
            distances = np.linalg.norm(filtered_vectors - query_np, axis=1)
            top_k_idx = np.argsort(distances)[:k]
            scores = distances[top_k_idx]
            
        elif metric == "cosine":
            dot_products = np.dot(filtered_vectors, query_np)
            query_norm = np.linalg.norm(query_np)
            matrix_norms = np.linalg.norm(filtered_vectors, axis=1)
            similarities = dot_products / (matrix_norms * query_norm + 1e-9)
            
            top_k_idx = np.argsort(similarities)[::-1][:k]
            scores = similarities[top_k_idx]

        # 3. Format and return results
        results = []
        for rank_idx, filtered_matrix_idx in enumerate(top_k_idx):
            v_id = filtered_ids[filtered_matrix_idx]
            results.append((v_id, float(scores[rank_idx]), self.metadata.get(v_id, {})))
            
        return results