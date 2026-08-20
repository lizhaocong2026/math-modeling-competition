import numpy as np
from typing import Tuple
import math


class MultiHeadAttention:
    def __init__(self, d_model=64, nhead=4, dropout=0.1):
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.dropout = dropout
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

    def _softmax(self, x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (e.sum(axis=axis, keepdims=True) + 1e-10)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = X.shape[0]
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)
        Q = Q.reshape(batch_size, -1, self.nhead, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.nhead, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.nhead, self.d_k).transpose(0, 2, 1, 3)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        attn_weights = self._softmax(scores, axis=-1)
        out = np.matmul(attn_weights, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = np.matmul(out, self.W_o)
        self._attn_weights = attn_weights
        return output, attn_weights

    def predict(self, X):
        out, _ = self.forward(X)
        return out