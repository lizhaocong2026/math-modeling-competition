import numpy as np
from typing import Dict, Any

class DiffusionLM:
    def __init__(self, seq_len=24, latent_dim=64, num_steps=100, lr=1e-3):
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.lr = lr
        self.beta = np.linspace(1e-4, 0.02, num_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        self.W1 = np.random.randn(latent_dim, latent_dim*2) * 0.01
        self.W2 = np.random.randn(latent_dim*2, latent_dim) * 0.01
        self.b1 = np.zeros((1, latent_dim*2))
        self.b2 = np.zeros((1, latent_dim))
        self.history = []
    def _relu(self, x): return np.maximum(0, x)
    def _forward_noise(self, z, t):
        h = self._relu(z @ self.W1 + self.b1)
        return h @ self.W2 + self.b2
    def q_sample(self, x0, t):
        t_i = int(t)
        sab = np.sqrt(self.alpha_bar[t_i])
        somab = np.sqrt(1 - self.alpha_bar[t_i])
        noise = np.random.randn(*x0.shape)
        return sab * x0 + somab * noise, noise
    def p_sample(self, xt, t):
        t_i = int(t)
        at = self.alpha[t_i]
        sab = np.sqrt(self.alpha_bar[t_i])
        noise_pred = self._forward_noise(xt, t_i)
        mean = (xt - (1-at)/sab * noise_pred) / np.sqrt(at)
        variance = max(self.beta[t_i], 0)
        noise = np.random.randn(*xt.shape) * np.sqrt(variance)
        return mean + noise
    def fit(self, X, epochs=50, batch_size=32, verbose=False):
        n = X.shape[0]
        hist = []
        for ep in range(epochs):
            idx = np.random.permutation(n)
            ep_loss, nb = 0.0, 0
            for st in range(0, n, batch_size):
                en = min(st + batch_size, n)
                bi = idx[st:en]
                Xb = X[bi]
                t = np.random.randint(0, self.num_steps, len(Xb))
                xt, noise = self.q_sample(Xb, t[0])
                npred = self._forward_noise(xt, t[0])
                loss = np.mean((npred - noise)**2)
                ep_loss += loss; nb += 1
                self._update_weights(xt, noise)
            hist.append(ep_loss / max(nb, 1))
            if verbose and (ep % 10 == 0 or ep == epochs-1):
                print("Epoch %d/%d loss=%.6f", ep, epochs, hist[-1])
        self.history = hist
        return {"status": "success", "final_loss": hist[-1] if hist else None}
    def _update_weights(self, xt, noise):
        eps = 1e-4
        for W in [self.W1, self.W2]:
            for i in range(min(2, W.shape[0])):
                for j in range(min(2, W.shape[1])):
                    old = W[i, j]
                    W[i, j] = old + eps
                    pp = self._forward_noise(xt, 0)
                    lp = np.mean((pp - noise)**2)
                    W[i, j] = old - eps
                    pm = self._forward_noise(xt, 0)
                    lm = np.mean((pm - noise)**2)
                    W[i, j] = old - self.lr * (lp - lm) / (2*eps)
    def generate(self, shape=None):
        if shape is None: shape = (1, self.seq_len, self.latent_dim)
        xt = np.random.randn(*shape)
        for t in range(self.num_steps-1, -1, -1):
            xt = self.p_sample(xt, t)
        return xt