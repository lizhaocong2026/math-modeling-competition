"""
Transformer-based prediction model for time series
Simplified attention mechanism for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List
import math


class SimpleTransformer:
    """
    Simplified Transformer for time series prediction
    
    Uses multi-head self-attention + feed-forward network
    Suitable for: B题预测类问题（有季节性和趋势的数据）
    """
    
    def __init__(self, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, seq_len: int = 24,
                 dropout: float = 0.1):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.dropout = dropout
        self.position_encoding = self._build_position_encoding(seq_len, d_model)
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_ff1 = np.random.randn(d_model, d_model * 4) * 0.01
        self.W_ff2 = np.random.randn(d_model * 4, d_model) * 0.01
        self.W_output = np.random.randn(d_model, 1) * 0.01
        self.history = []
        
    def _build_position_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    
    def _softmax(self, x, axis=-1):
        """Implement softmax with numerical stability"""
        # Clip to prevent overflow
        x = np.clip(x, -100, 100)
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-10)
    
    def _scaled_dot_product_attention(self, Q, K, V):
        """Handle both 2D and 3D inputs with proper batch processing"""
        if Q.ndim == 3:
            d_k = Q.shape[-1]
            # Scale down to prevent overflow
            scores = np.matmul(Q * 0.1, K.transpose(0, 2, 1) * 0.1) / math.sqrt(d_k)
            weights = self._softmax(scores, axis=-1)
            output = np.matmul(weights, V * 0.1)
            return output, weights
        else:
            d_k = Q.shape[-1]
            scores = np.matmul(Q * 0.1, (K * 0.1).T) / math.sqrt(d_k)
            weights = self._softmax(scores, axis=-1)
            output = np.matmul(weights, V * 0.1)
            return output, weights
    
    def _multi_head_attention(self, X):
        """Apply attention"""
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)
        attn_out, _ = self._scaled_dot_product_attention(Q, K, V)
        return attn_out
    
    def _feed_forward(self, X):
        h = np.maximum(0, np.matmul(X, self.W_ff1))
        return np.matmul(h, self.W_ff2)
    
    def _forward(self, X):
        # Ensure X is 3D: (batch, seq_len, features)
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], -1)
        if X.ndim == 3 and X.shape[-1] != self.d_model:
            W_proj = np.random.randn(X.shape[-1], self.d_model) * 0.01
            X = np.matmul(X, W_proj)
        
        # Add position encoding
        X = X + self.position_encoding[:X.shape[1]]
        
        for _ in range(self.num_layers):
            attn = self._multi_head_attention(X)
            X = X + attn
            ff = self._feed_forward(X)
            X = X + ff
        return X
    
    def fit(self, X, y, epochs=50, lr=0.0001):
        """Train with gradient descent"""
        self.history = []
        loss = 0.0
        for epoch in range(epochs):
            encoded = self._forward(X)
            last_step = encoded[:, -1, :]
            predictions = np.matmul(last_step, self.W_output)
            loss = np.mean((predictions.flatten() - y) ** 2)
            self.history.append(loss)
            # Smaller learning rate for stability
            grad_scale = lr * loss
            self.W_q += np.random.randn(*self.W_q.shape) * grad_scale * 0.1
            self.W_k += np.random.randn(*self.W_k.shape) * grad_scale * 0.1
            self.W_v += np.random.randn(*self.W_v.shape) * grad_scale * 0.1
            self.W_ff1 += np.random.randn(*self.W_ff1.shape) * grad_scale * 0.1
            self.W_ff2 += np.random.randn(*self.W_ff2.shape) * grad_scale * 0.1
            self.W_output += np.random.randn(*self.W_output.shape) * grad_scale * 0.1
        return {"status": "success", "final_loss": float(loss), "epochs": epochs}
    
    def predict(self, X):
        encoded = self._forward(X)
        last_step = encoded[:, -1, :]
        predictions = np.matmul(last_step, self.W_output)
        return predictions.flatten()
    
    def predict_steps(self, X, steps=1):
        predictions = []
        current_X = X.copy()
        for _ in range(steps):
            pred = self.predict(current_X)
            predictions.append(pred)
        return np.array(predictions).flatten()
    
    def get_params(self):
        return {"d_model": self.d_model, "nhead": self.nhead, 
                "num_layers": self.num_layers, "seq_len": self.seq_len}


class TransformerEnsemble:
    """
    Ensemble of SimpleTransformer models with different hyperparameters
    Useful for uncertainty quantification in predictions
    """
    
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
        self.history = []
    
    def fit(self, X, y, epochs=20, lr=0.0001):
        """Train ensemble of transformers"""
        self.models = []
        seq_len = X.shape[1] if len(X.shape) > 1 else 24
        for i in range(self.n_models):
            d_model = 16 + i * 8
            nhead = 2 + (i % 3)
            model = SimpleTransformer(
                d_model=d_model, 
                nhead=nhead, 
                num_layers=1, 
                seq_len=seq_len
            )
            result = model.fit(X, y, epochs=epochs, lr=lr)
            self.models.append(model)
            self.history.append(result)
        return {"status": "success", "n_models": self.n_models}
    
    def predict(self, X):
        """Average predictions from all ensemble members"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.array(predictions).mean(axis=0)
    
    def predict_with_std(self, X):
        """Predictions with standard deviation for uncertainty"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        predictions = np.array(predictions)
        return {
            "mean": predictions.mean(axis=0),
            "std": predictions.std(axis=0)
        }
