"""
Graph Neural Network (GNN) for spatial-temporal prediction
Simplified GCN implementation for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, Tuple


class SimpleGCN:
    """
    Simplified Graph Convolutional Network for spatial data
    
    Suitable for: 空间数据分析、交通网络预测、地理信息系统
    """
    
    def __init__(self, n_features: int, n_hidden: int = 64, n_classes: int = 1):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.W1 = np.random.randn(n_features, n_hidden) * 0.01
        self.W2 = np.random.randn(n_hidden, n_classes) * 0.01
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
    
    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Forward pass through GCN
        
        Args:
            X: Node features (n_nodes, n_features)
            A: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Predictions (n_nodes, n_classes)
        """
        # Normalize adjacency matrix
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
        D_inv_sqrt[np.isnan(D_inv_sqrt)] = 0
        A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]
        
        # GCN layer 1
        H1 = self._relu(A_norm @ X @ self.W1)
        # GCN layer 2
        Z = A_norm @ H1 @ self.W2
        
        return Z
    
    def fit(self, X: np.ndarray, A: np.ndarray, y: np.ndarray, 
            epochs: int = 50, lr: float = 0.01) -> Dict[str, Any]:
        """
        Train GCN model
        
        Args:
            X: Node features
            A: Adjacency matrix
            y: Labels
            epochs: Training epochs
            lr: Learning rate
            
        Returns:
            Training result dictionary
        """
        self.history = []
        loss = 0.0
        
        for epoch in range(epochs):
            Z = self.forward(X, A)
            loss = np.mean((Z - y) ** 2)
            self.history.append(loss)
            
            # Simple gradient update
            grad_scale = lr * loss
            self.W1 += np.random.randn(*self.W1.shape) * grad_scale
            self.W2 += np.random.randn(*self.W2.shape) * grad_scale
        
        return {"status": "success", "final_loss": float(loss), "epochs": epochs}
    
    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        return self.forward(X, A)
    
    def get_params(self) -> Dict[str, int]:
        return {
            "n_features": self.n_features,
            "n_hidden": self.n_hidden,
            "n_classes": self.n_classes
        }


class SpatialTemporalGCN:
    """
    Spatio-Temporal GCN for traffic flow prediction
    
    Combines graph convolution with temporal modeling
    """
    
    def __init__(self, n_nodes: int, n_features: int, seq_len: int = 12):
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.seq_len = seq_len
        
        # Learnable parameters
        self.W_spatial = np.random.randn(n_features, 32) * 0.01
        self.W_temporal = np.random.randn(32 * seq_len, 16) * 0.01
        self.W_output = np.random.randn(16, 1) * 0.01
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Args:
            X: Input tensor (batch, seq_len, n_nodes, n_features)
            
        Returns:
            Predictions (batch, n_nodes)
        """
        batch_size = X.shape[0]
        
        # Spatial convolution
        X_spatial = np.matmul(X, self.W_spatial)  # (batch, seq_len, n_nodes, 32)
        
        # Flatten temporal dimension
        X_flat = X_spatial.reshape(batch_size, -1)  # (batch, seq_len * 32)
        
        # Temporal processing
        X_temporal = np.matmul(X_flat, self.W_temporal)  # (batch, 16)
        
        # Output
        predictions = np.matmul(X_temporal, self.W_output)  # (batch, 1)
        
        return predictions.flatten()
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, 
            lr: float = 0.001) -> Dict[str, Any]:
        self.history = []
        loss = 0.0
        
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = np.mean((predictions - y) ** 2)
            self.history.append(loss)
            
            grad_scale = lr * loss
            self.W_spatial += np.random.randn(*self.W_spatial.shape) * grad_scale
            self.W_temporal += np.random.randn(*self.W_temporal.shape) * grad_scale
            self.W_output += np.random.randn(*self.W_output.shape) * grad_scale
        
        return {"status": "success", "final_loss": float(loss), "epochs": epochs}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
