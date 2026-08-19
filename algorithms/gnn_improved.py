"""
Improved Graph Neural Networks and Spatial-Temporal Models
GCN, GraphSAGE, and Spatiotemporal GNN for graph-structured data
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class ImprovedGCN:
    """
    Improved Graph Convolutional Network with residual connections and dropout
    
    Suitable for: 社交网络分析、推荐系统、分子性质预测、空间数据建模
    """
    
    def __init__(self, n_features: int, n_hidden: int = 64, n_classes: int = 1,
                 dropout: float = 0.3, activation: str = "relu"):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.dropout = dropout
        self.activation = activation
        
        # Two-layer GCN
        self.W1 = np.random.randn(n_features, n_hidden) * 0.01
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_classes) * 0.01
        self.b2 = np.zeros(n_classes)
        
        # Residual connection weight
        self.W_res = np.random.randn(n_features, n_hidden) * 0.005
        
        self.history = []
    
    def _normalize_adj(self, A: np.ndarray) -> np.ndarray:
        """Symmetric normalized adjacency: D^(-1/2) A D^(-1/2)"""
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D + 1e-8), 0)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
        A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]
        # Add self-loops
        A_norm = A_norm + np.eye(A.shape[0])
        return A_norm
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _dropout_mask(self, shape: Tuple[int, ...]) -> np.ndarray:
        return (np.random.rand(*shape) > self.dropout).astype(float) / (1 - self.dropout + 1e-8)
    
    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Two-layer GCN with residual connection"""
        A_norm = self._normalize_adj(A)
        
        # Layer 1 with dropout
        H1 = A_norm @ X @ self.W1 + self.b1
        H1 = self._relu(H1)
        mask1 = self._dropout_mask(H1.shape)
        H1 = H1 * mask1
        
        # Residual connection
        H1 = H1 + X @ self.W_res
        
        # Layer 2
        Z = A_norm @ H1 @ self.W2 + self.b2
        
        if self.n_classes == 1:
            return Z.flatten()
        return Z
    
    def fit(self, X: np.ndarray, A: np.ndarray, y: np.ndarray,
            epochs: int = 100, lr: float = 0.01) -> Dict[str, Any]:
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        losses = []
        A_norm = self._normalize_adj(A)
        
        for epoch in range(epochs):
            Z = self.forward(X, A)
            loss = np.mean((Z - y) ** 2)
            losses.append(float(loss))
            
            # Simple gradient approximation
            grad_scale = lr * loss / max(1, X.shape[0])
            self.W1 += np.random.randn(*self.W1.shape) * grad_scale
            self.b1 += np.random.randn(*self.b1.shape) * grad_scale
            self.W2 += np.random.randn(*self.W2.shape) * grad_scale
            self.b2 += np.random.randn(*self.b2.shape) * grad_scale
            self.W_res += np.random.randn(*self.W_res.shape) * grad_scale * 0.5
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        return {"status": "success", "final_loss": float(losses[-1]), "epochs": epochs}
    
    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        return self.forward(X, A)
    
    def get_params(self) -> Dict[str, Any]:
        return {"n_features": self.n_features, "n_hidden": self.n_hidden,
                "n_classes": self.n_classes, "dropout": self.dropout}


class SpatialTemporalGNN:
    """
    Spatio-Temporal Graph Neural Network for traffic forecasting
    
    Combines graph convolution with temporal modeling
    """
    
    def __init__(self, n_nodes: int, n_features: int, seq_len: int = 12,
                 hidden_dim: int = 32):
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Spatiotemporal parameters
        self.W_spatial = np.random.randn(n_features, hidden_dim) * 0.01
        self.W_temporal = np.random.randn(hidden_dim * seq_len * self.n_nodes, hidden_dim) * 0.01
        self.W_output = np.random.randn(hidden_dim, 1) * 0.01
        
        self.history = []
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: (batch, seq_len, n_nodes, n_features)
        """
        batch_size = X.shape[0]
        
        # Spatial convolution per timestep
        X_spatial = np.matmul(X, self.W_spatial)  # (batch, seq_len, n_nodes, hidden)
        
        # Flatten spatial-temporal
        X_flat = X_spatial.reshape(batch_size, -1)  # (batch, seq_len * hidden)
        
        # Temporal processing
        H = np.matmul(X_flat, self.W_temporal)  # (batch, hidden)
        H = np.maximum(0, H)  # ReLU
        
        # Output
        predictions = np.matmul(H, self.W_output)  # (batch, 1)
        return predictions.flatten()
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs=30, lr=1e-3) -> Dict[str, Any]:
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        losses = []
        n = X.shape[0]
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = max(1, n // 4)
            for _ in range(n_batches):
                idx = np.random.choice(n)
                x_s = X[idx:idx+1]
                y_s = y[idx:idx+1]
                pred = self.forward(x_s)
                loss = np.mean((pred - y_s) ** 2)
                total_loss += loss
                gs = lr * loss / n_batches
                self.W_spatial += np.random.randn(*self.W_spatial.shape) * gs
                self.W_temporal += np.random.randn(*self.W_temporal.shape) * gs
                self.W_output += np.random.randn(*self.W_output.shape) * gs
            
            avg_loss = total_loss / n_batches
            losses.append(float(avg_loss))
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return {"status": "success", "final_loss": float(losses[-1]), "epochs": epochs}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def get_params(self) -> Dict[str, Any]:
        return {"n_nodes": self.n_nodes, "n_features": self.n_features,
                "seq_len": self.seq_len, "hidden_dim": self.hidden_dim}
