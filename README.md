# Pure NumPy Vector Database & HNSW Search Index from Scratch

A high-performance, lightweight vector database and Hierarchical Navigable Small World (HNSW) approximate nearest neighbor (ANN) search index built entirely from scratch using **NumPy**. This engine implements optimized matrix mathematics, multi-layer graph skip-lists, and metadata pre-filtering without relying on external databases or high-level indexing libraries like Faiss or Scikit-Learn.

## 🚀 Key Features

* **Pure NumPy Execution:** Eliminates slow Python `for` loops by leveraging optimized contiguous C/BLAS matrix operations under the hood.
* **HNSW Graph Structure:** Achieves $O(\log N)$ search scaling by constructing a multi-layer graph network inspired by probabilistic skip-lists.
* **Vectorized Distance Engine:** Native support for high-dimensional Euclidean Distance (L2) and Cosine Similarity metrics.
* **Metadata Pre-Filtering:** Real-world attribute filtering constraints integrated directly into the vector calculation pipeline.

---

## 🧮 Mathematical Foundations

To maintain raw execution efficiency, all distance heuristics are calculated across entire vector matrices simultaneously utilizing raw linear algebra operations.

### 1. Euclidean Distance (L2)
Measures the true geometric distance between an arbitrary query vector $\mathbf{u}$ and target vector matrix $\mathbf{v}$:

$$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2} = \|\mathbf{u} - \mathbf{v}\|_2$$

### 2. Cosine Similarity
Measures the angular alignment between orientation paths irrespective of their overall magnitude scale:

$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

---

## 🏗️ Architecture Design

The engine divides performance responsibilities across two decoupled components:

1. **`NumPyVectorDB` (The Storage Core):** Manages row-major, C-contiguous matrix allocation arrays for sequential memory layout alignment. This maximizes CPU cache line utility when scanning multi-dimensional fields.
2. **`HNSWIndex` (The Routing Graph):** Handles multi-layered navigation graphs. Sparser top layers provide long-range routing jumps to find the vector neighborhood instantly, while dense lower layers isolate exact near neighbors via localized greedy searches.