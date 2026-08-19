import numpy as np
from typing import Tuple

class ContrastiveLoss:
    def __init__(self, temperature=0.07, margin=1.0):
        self.temperature = temperature
        self.margin = margin
        self.history = []
    def _softmax(self, x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (e.sum(axis=axis, keepdims=True) + 1e-10)
    def forward(self, emb_a, emb_p, emb_n):
        pos_sim = np.sum(emb_a * emb_p, axis=1) / self.temperature
        neg_sim = np.sum(emb_a * emb_n, axis=1) / self.temperature
        logits = np.stack([pos_sim, neg_sim], axis=1)
        probs = self._softmax(logits, axis=1)
        loss = -np.mean(np.log(probs[:, 0] + 1e-10))
        self.history.append(loss)
        return loss, pos_sim
    def contrastive_loss(self, emb_a, emb_p, emb_n):
        dist_pos = np.sum((emb_a - emb_p)**2, axis=1)
        dist_neg = np.sum((emb_a - emb_n)**2, axis=1)
        loss = np.mean(np.minimum(dist_pos, self.margin**2) + self.margin * np.maximum(dist_neg - self.margin, 0))
        self.history.append(loss)
        return loss

class ContrastiveEncoder:
    def __init__(self, input_dim=64, embed_dim=32):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.W = np.random.randn(input_dim, embed_dim) * 0.01
        self.b = np.zeros((1, embed_dim))
    def encode(self, X):
        h = X @ self.W + self.b
        h = np.maximum(0, h)
        norm = np.linalg.norm(h, axis=1, keepdims=True) + 1e-10
        return h / norm
    def fit(self, X_pos, X_neg, epochs=50, lr=0.01):
        loss_fn = ContrastiveLoss()
        for ep in range(epochs):
            a = self.encode(X_pos)
            p = self.encode(X_pos + np.random.randn(*X_pos.shape)*0.1)
            n = self.encode(X_neg)
            loss, _ = loss_fn.forward(a, p, n)
            self.W -= lr * (a.T @ (a - p)) / X_pos.shape[0]
        return {"status": "success", "final_loss": loss}