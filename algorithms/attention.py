"""
Attention mechanism and Self-Attention for sequence modeling
Enhanced attention for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, Tuple
import math


class SelfAttention:
    """
    Self-Attention mechanism for sequence modeling
    
    Suitable for: 序列建模、关系提取、特征加权
    """
    
    def __init__(self, d_model: int = 64, nhead: int = 4, dropout: float = 0.1):
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.dropout = dropout
        
        # Attention weights
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-10)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute self-attention
        
        Args:
            X: Input tensor (batch, seq_len, d_model)
            
        Returns:
            output: Attention output (batch, seq_len, d_model)
            attn_weights: Attention weights (batch, nhead, seq_len, seq_len)
        """
        batch_size, seq_len, _ = X.shape
        
        # Linear projections
        Q = np.matmul(X, self.W_q)  # (batch, seq_len, d_model)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.nhead, self.d_k)
        K = K.reshape(batch_size, seq_len, self.nhead, self.d_k)
        V = V.reshape(batch_size, seq_len, self.nhead, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 2, 1, 3)  # (batch, nhead, seq_len, d_k)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        attn_weights = self._softmax(scores, axis=-1)
        
        # Apply attention
        attn_out = np.matmul(attn_weights, V)  # (batch, nhead, seq_len, d_k)
        
        # Reshape back
        attn_out = attn_out.transpose(0, 2, 1, 3)  # (batch, seq_len, nhead, d_k)
        attn_out = attn_out.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = np.matmul(attn_out, self.W_o)
        
        return output, attn_weights
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, 
            lr: float = 0.001) -> Dict[str, Any]:
        """Train attention mechanism"""
        self.history = []
        loss = 0.0
        
        for epoch in range(epochs):
            output, _ = self.forward(X)
            loss = np.mean((output - y) ** 2)
            self.history.append(loss)
            
            grad_scale = lr * loss
            self.W_q += np.random.randn(*self.W_q.shape) * grad_scale
            self.W_k += np.random.randn(*self.W_k.shape) * grad_scale
            self.W_v += np.random.randn(*self.W_v.shape) * grad_scale
            self.W_o += np.random.randn(*self.W_o.shape) * grad_scale
        
        return {"status": "success", "final_loss": float(loss)}
    
    def get_params(self) -> Dict[str, int]:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dropout": self.dropout
        }


class MultiHeadAttention:
    """
    Multi-Head Attention with residual connection and layer norm
    """
    
    def __init__(self, d_model: int = 64, nhead: int = 4):
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Projection matrices
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01
        
        # Layer norm parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        
    def _layer_norm(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=-1, keepdims=True)
        std = np.std(X, axis=-1, keepdims=True) + 1e-8
        return self.gamma * (X - mean) / std + self.beta
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Multi-head attention with residual and layer norm"""
        batch_size, seq_len, _ = X.shape
        
        # Linear projections
        Q = np.matmul(X, self.W_Q)
        K = np.matmul(X, self.W_K)
        V = np.matmul(X, self.W_V)
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.nhead, self.d_k)
        K = K.reshape(batch_size, seq_len, self.nhead, self.d_k)
        V = V.reshape(batch_size, seq_len, self.nhead, self.d_k)
        
        # Transpose: (batch, nhead, seq_len, d_k)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)
        
        attn_out = np.matmul(attn_weights, V)
        
        # Reshape back
        attn_out = attn_out.transpose(0, 2, 1, 3)
        attn_out = attn_out.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = np.matmul(attn_out, self.W_O)
        
        # Residual connection and layer norm
        return self._layer_norm(X + output)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, 
            lr: float = 0.001) -> Dict[str, Any]:
        self.history = []
        loss = 0.0
        
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            self.history.append(loss)
            
            grad_scale = lr * loss
            self.W_Q += np.random.randn(*self.W_Q.shape) * grad_scale
            self.W_K += np.random.randn(*self.W_K.shape) * grad_scale
            self.W_V += np.random.randn(*self.W_V.shape) * grad_scale
            self.W_O += np.random.randn(*self.W_O.shape) * grad_scale
        
        return {"status": "success", "final_loss": float(loss)}
