"""
GraphSAGE - Graph Sample and Aggregate
Simplified implementation for spatial-temporal prediction in math modeling
"""
import numpy as np
from typing import Dict, Any, List, Tuple


class GraphSAGENode:
    """
    Simplified GraphSAGE node representation
    
    Suitable for: 交通网络预测、社交网络分析、空间数据建模
    """
    
    def __init__(self, node_id: int, features: np.ndarray):
        self.node_id = node_id
        self.features = features
        self.embeddings = None
        
    def update_embedding(self, neighbor_embeddings: List[np.ndarray], 
                         aggregator: str = "mean"):
        """Update node embedding using aggregator"""
        if aggregator == "mean":
            neighbor_agg = np.mean(neighbor_embeddings, axis=0)
        elif aggregator == "lstm":
            neighbor_agg = self._lstm_aggregate(neighbor_embeddings)
        else:
            neighbor_agg = np.mean(neighbor_embeddings, axis=0)
        
        # Combine with own features
        combined = np.concatenate([self.features, neighbor_agg])
        # Simple projection
        self.embeddings = combined @ self.W_projection
        
    def _lstm_aggregate(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """LSTM-based aggregation"""
        if len(embeddings) == 0:
            return np.zeros(embeddings[0].shape if embeddings else (10,))
        # Simple LSTM-like aggregation
        h = embeddings[0]
        for e in embeddings[1:]:
            h = np.tanh(h + e)
        return h


class GraphSAGE:
    """
    GraphSAGE for node embedding and link prediction
    
    Architecture: Neighbor sampling -> Aggregation -> Projection
    """
    
    def __init__(self, n_features: int = 10, embedding_dim: int = 32, 
                 n_layers: int = 2, sampler_size: int = 10):
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.sampler_size = sampler_size
        self.nodes = {}
        self.edges = []
        
        # Weight matrices for each layer
        self.W_layers = []
        for _ in range(n_layers):
            W = np.random.randn(n_features if _ == 0 else embedding_dim, 
                              embedding_dim) * 0.01
            self.W_layers.append(W)
        
    def add_node(self, node_id: int, features: np.ndarray):
        """Add node to graph"""
        self.nodes[node_id] = GraphSAGENode(node_id, features)
        
    def add_edge(self, u: int, v: int):
        """Add edge to graph"""
        self.edges.append((u, v))
        
    def sample_neighbors(self, node_id: int, size: int = None) -> List[int]:
        """Sample neighbors of a node"""
        size = size or self.sampler_size
        neighbors = [v for u, v in self.edges if u == node_id] + \
                   [u for u, v in self.edges if v == node_id]
        if len(neighbors) > size:
            return np.random.choice(neighbors, size, replace=False).tolist()
        return neighbors
    
    def forward(self, node_ids: List[int]) -> np.ndarray:
        """Forward pass to compute embeddings"""
        embeddings = []
        
        for node_id in node_ids:
            if node_id not in self.nodes:
                continue
            
            node = self.nodes[node_id]
            current_features = node.features.copy()
            
            # Multi-layer aggregation
            for layer in range(self.n_layers):
                neighbors = self.sample_neighbors(node_id)
                neighbor_feats = [self.nodes[n].embeddings if hasattr(self.nodes[n], 'embeddings') and self.nodes[n].embeddings is not None 
                                 else self.nodes[n].features 
                                 for n in neighbors]
                
                # Aggregate
                if neighbor_feats:
                    agg = np.mean(neighbor_feats, axis=0)
                else:
                    agg = np.zeros(current_features.shape)
                
                # Combine and project
                combined = np.concatenate([current_features, agg])
                current_features = np.tanh(combined @ self.W_layers[layer])
            
            node.embeddings = current_features
            embeddings.append(current_features)
        
        return np.array(embeddings)
    
    def fit(self, node_ids: List[int], epochs: int = 10, 
            lr: float = 0.001) -> Dict[str, Any]:
        """Train GraphSAGE (simplified)"""
        self.history = []
        
        for epoch in range(epochs):
            embeddings = self.forward(node_ids)
            # Simplified loss: encourage connected nodes to have similar embeddings
            loss = 0
            for u, v in self.edges[:100]:  # Sample edges
                if u in self.nodes and v in self.nodes:
                    eu = self.nodes[u].embeddings
                    ev = self.nodes[v].embeddings
                    if eu is not None and ev is not None:
                        loss += np.linalg.norm(eu - ev) ** 2
            
            loss /= max(len(self.edges), 1)
            self.history.append(loss)
            
            # Update weights
            for i in range(len(self.W_layers)):
                self.W_layers[i] += np.random.randn(*self.W_layers[i].shape) * lr * loss
        
        return {"status": "success", "final_loss": float(loss), "epochs": epochs}
    
    def predict_link(self, u: int, v: int) -> float:
        """Predict link probability between two nodes"""
        if u not in self.nodes or v not in self.nodes:
            return 0.0
        eu = self.nodes[u].embeddings
        ev = self.nodes[v].embeddings
        if eu is None or ev is None:
            return 0.0
        # Dot product as similarity
        return float(np.dot(eu, ev) / (np.linalg.norm(eu) * np.linalg.norm(ev) + 1e-8))
    
    def get_node_similarity(self, u: int, v: int) -> float:
        """Get cosine similarity between two nodes"""
        return self.predict_link(u, v)
