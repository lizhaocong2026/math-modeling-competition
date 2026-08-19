import numpy as np
from typing import Tuple, Dict, Any
import math


class FlashAttention:
    def __init__(self, d_model=64, nhead=4, seq_len=16, dropout=0.1, lr=0.01):
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.seq_len = seq_len
        self.dropout = dropout
        self.lr = lr
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        self.pe = self._build_pe(seq_len, d_model)
        self.history = []

    def _build_pe(self, seq_len, d_model):
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def _softmax(self, x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (e.sum(axis=axis, keepdims=True) + 1e-10)

    def forward(self, X):
        batch_size = X.shape[0]
        X_pe = X + self.pe[np.newaxis, :, :]
        Q = np.matmul(X_pe, self.W_q)
        K = np.matmul(X_pe, self.W_k)
        V = np.matmul(X_pe, self.W_v)
        Q = Q.reshape(batch_size, self.seq_len, self.nhead, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, self.seq_len, self.nhead, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, self.seq_len, self.nhead, self.d_k).transpose(0, 2, 1, 3)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        attn_weights = self._softmax(scores, axis=-1)
        out = np.matmul(attn_weights, V)
        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(batch_size, self.seq_len, self.d_model)
        output = np.matmul(out, self.W_o)
        self._attn_weights = attn_weights
        self._Q, self._K, self._V = Q, K, V
        return output, attn_weights

    def fit(self, X, y, epochs=10, batch_size=32, verbose=False):
        n = X.shape[0]
        mse_hist = []
        for ep in range(epochs):
            idx = np.random.permutation(n)
            ep_loss, nb = 0.0, 0
            for st in range(0, n, batch_size):
                en = min(st + batch_size, n)
                bi = idx[st:en]
                Xb, yb = X[bi], y[bi]
                out, _ = self.forward(Xb)
                err = out - yb
                loss = np.mean(err ** 2)
                ep_loss += loss
                nb += 1
                self._backward(Xb, yb, err)
            mse_hist.append(ep_loss / max(nb, 1))
            if verbose and (ep % 3 == 0 or ep == epochs - 1):
                print(f"Epoch {ep}/{epochs} MSE={mse_hist[-1]:.6f}")
        self.history = mse_hist
        return {"status": "success", "final_mse": mse_hist[-1] if mse_hist else None}

    def _backward(self, X, y, error):
        batch = X.shape[0]
        d_out = 2.0 * error / (batch * self.d_model)
        out_pre = np.matmul(self._attn_weights, self._V)
        out_pre = out_pre.transpose(0, 2, 1, 3).reshape(batch, self.seq_len, self.d_model)
        d_W_o = np.einsum("bst,bmd->td", out_pre, d_out)
        self.W_o -= self.lr * d_W_o / batch
        d_out_pre = d_out @ self.W_o
        d_out_pre = d_out_pre.reshape(batch, self.seq_len, self.nhead, self.d_k)
        d_out_pre = d_out_pre.transpose(0, 2, 1, 3)
        d_scores = np.matmul(d_out_pre, self._V.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        d_attn = self._softmax_deriv(self._attn_weights) * d_scores
        d_V = np.matmul(d_attn.transpose(0, 1, 3, 2), self._V)
        d_K = np.matmul(d_attn.transpose(0, 1, 3, 2), self._Q)
        d_Q = np.matmul(d_attn, self._K)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(batch, self.seq_len, self.d_model)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(batch, self.seq_len, self.d_model)
        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(batch, self.seq_len, self.d_model)
        X_pe = X + self.pe[np.newaxis, :, :]
        d_W_v = np.einsum("bst,bmd->td", X_pe, d_V)
        d_W_k = np.einsum("bst,bmd->td", X_pe, d_K)
        d_W_q = np.einsum("bst,bmd->td", X_pe, d_Q)
        self.W_v -= self.lr * d_W_v / batch
        self.W_k -= self.lr * d_W_k / batch
        self.W_q -= self.lr * d_W_q / batch

    def _softmax_deriv(self, softmax_out):
        return softmax_out * (1 - softmax_out)

    def predict(self, X):
        out, _ = self.forward(X)
        return out