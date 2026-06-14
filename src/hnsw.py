import numpy as np
import random
from typing import List, Tuple, Dict, Set
from src.database import NumPyVectorDB

class HNSWIndex:
    def __init__(self, db: NumPyVectorDB, M: int = 16, ef_construction: int = 64, ef_search: int = 32):
        """
        Initializes the HNSW Approximate Nearest Neighbor Index.
        
        :param db: An instance of the NumPyVectorDB containing the raw matrices.
        :param M: Max number of outgoing connections per node in a layer.
        :param ef_construction: Depth of dynamic candidate list evaluated during insertion.
        :param ef_search: Depth of dynamic candidate list evaluated during querying.
        """
        self.db = db
        self.M = M
        self.M0 = 2 * M  # Layer 0 can handle twice as many connections safely
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Multi-layer graph: List of dicts. 
        # Each layer index corresponds to a dict: { node_id: set(neighbor_ids) }
        self.graphs: List[Dict[int, Set[int]]] = []
        
        # Tracking the entry point into the entire multi-layer index
        self.enter_node: int = None
        self.max_level: int = -1
        
        # Normalization factor for exponential decay level assignment
        self.mL = 1.0 / np.log(M)

    def _get_random_level(self) -> int:
        """Determines the maximum layer a new vector will exist on using exponential decay."""
        r = random.uniform(1e-9, 1.0)
        return int(np.floor(-np.log(r) * self.mL))

    def _get_distance(self, node_id_1: int, node_id_2: int) -> float:
        """Helper to get L2 distance between two stored vectors via index IDs."""
        idx1 = self.db.vector_ids.index(node_id_1)
        idx2 = self.db.vector_ids.index(node_id_2)
        v1 = self.db.vectors[idx1]
        v2 = self.db.vectors[idx2]
        return float(np.linalg.norm(v1 - v2))

    def _get_distance_to_vector(self, query_vec: np.ndarray, target_node_id: int) -> float:
        """Helper to get L2 distance from an arbitrary query array to a stored node."""
        target_idx = self.db.vector_ids.index(target_node_id)
        target_vec = self.db.vectors[target_idx]
        return float(np.linalg.norm(query_vec - target_vec))

    def search_layer_greedy(self, query_vec: np.ndarray, enter_node: int, level: int) -> int:
        """
        Traverses a single graph layer greedily.
        Moves from node to neighbor only if the neighbor is closer to the query vector.
        """
        curr_node = enter_node
        curr_dist = self.get_distance_to_vector(query_vec, curr_node)
        
        layer_graph = self.graphs[level]
        changed = True
        
        while changed:
            changed = False
            # Check all bidirectional connections of the current node at this level
            neighbors = layer_graph.get(curr_node, set())
            if not neighbors:
                break
                
            for neighbor in neighbors:
                d = self._get_distance_to_vector(query_vec, neighbor)
                if d < curr_dist:
                    curr_dist = d
                    curr_node = neighbor
                    changed = True  # Hop to closer neighbor and continue scanning
                    
        return curr_node
    def search_layer(self, query_vec: np.ndarray, enter_nodes: List[int], ef: int, level: int) -> List[Tuple[float, int]]:
        """
        More advanced layer search tracking 'ef' candidates inside a priority queue structure.
        Returns the closest candidates found on this specific level.
        """
        # Visited set to prevent infinite cycles
        visited = set(enter_nodes)
        
        # Candidate pool storing items as (distance, node_id)
        candidates = []
        for node in enter_nodes:
            dist = self._get_distance_to_vector(query_vec, node)
            candidates.append((dist, node))
            
        # Sort candidates so the closest is first
        candidates.sort(key=lambda x: x[0])
        
        # Result set tracks the absolute closest nodes found so far
        v_results = list(candidates)
        
        layer_graph = self.graphs[level]
        
        while len(candidates) > 0:
            # Pop the closest unvisited candidate
            curr_dist, curr_node = candidates.pop(0)
            
            # If the closest candidate is further than the furthest item in our results, stop
            if curr_dist > v_results[-1][0]:
                break
                
            neighbors = layer_graph.get(curr_node, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    d = self._get_distance_to_vector(query_vec, neighbor)
                    furthest_result_dist = v_results[-1][0]
                    
                    if d < furthest_result_dist or len(v_results) < ef:
                        candidates.append((d, neighbor))
                        v_results.append((d, neighbor))
                        
                        # Re-sort and cap the results pool to the 'ef' limit
                        candidates.sort(key=lambda x: x[0])
                        v_results.sort(key=lambda x: x[0])
                        if len(v_results) > ef:
                            v_results = v_results[:ef]
                            
        return v_results

    def add_element(self, node_id: int):
        """
        Inserts a new vector node into the multi-layer HNSW graph structure.
        Assumes the vector has already been stored inside the database matrix.
        """
        idx = self.db.vector_ids.index(node_id)
        query_vec = self.db.vectors[idx]
        
        # 1. Determine maximum insertion level for this vector
        insert_level = self._get_random_level()
        
        # Grow the graph layers if this node climbed higher than our current maximum
        while len(self.graphs) <= insert_level:
            self.graphs.append({})
            
        curr_enter_node = self.enter_node
        
        # 2. Top-Down fast routing phase down to the entry level
        if curr_enter_node is not None:
            for lvl in range(self.max_level, insert_level, -1):
                curr_enter_node = self.search_layer_greedy(query_vec, curr_enter_node, lvl)
                
        # 3. Connection and scaling phase across all assigned layers
        start_lvl = min(self.max_level, insert_level)
        enter_nodes = [curr_enter_node] if curr_enter_node is not None else []
        
        for lvl in range(start_lvl, -1, -1):
            # Find the best neighborhood candidates at this layer
            candidates = self.search_layer(query_vec, enter_nodes, self.ef_construction, lvl)
            
            # Pick the top M closest neighbors to create bidirectional links
            layer_m = self.M0 if lvl == 0 else self.M
            neighbors_to_link = [node_id for _, node_id in candidates[:layer_m]]
            
            # Initialize adjacency map entries if needed
            if node_id not in self.graphs[lvl]:
                self.graphs[lvl][node_id] = set()
                
            for neighbor in neighbors_to_link:
                # Add forward link
                self.graphs[lvl][node_id].add(neighbor)
                
                # Add backward link
                if neighbor not in self.graphs[lvl]:
                    self.graphs[lvl][neighbor] = set()
                self.graphs[lvl][neighbor].add(node_id)
                
                # Prune neighbor connections if they exceed max configuration parameters
                if len(self.graphs[lvl][neighbor]) > layer_m:
                    # Sort neighbor's links by distance and drop the furthest one
                    neighbor_links = list(self.graphs[lvl][neighbor])
                    neighbor_links.sort(key=lambda n: self._get_distance(neighbor, n))
                    # Keep the closest ones, discard the rest
                    self.graphs[lvl][neighbor] = set(neighbor_links[:layer_m])
            
            # Prepare entry nodes for the next layer down
            enter_nodes = [node for _, node in candidates]
            
        # 4. Global tracking adjustments
        if insert_level > self.max_level or self.enter_node is None:
            self.max_level = insert_level
            self.enter_node = node_id