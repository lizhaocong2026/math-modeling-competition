import numpy as np
from typing import Dict, Any, Tuple


class VAE:
    def __init__(self, input_dim=64, latent_dim=16, hidden_dim=128, lr=1e-3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        # Encoder
        self.enc_w1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.enc_b1 = np.zeros((1, hidden_dim))
        self.enc_w_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.enc_b_mu = np.zeros((1, latent_dim))
        self.enc_w_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.enc_b_logvar = np.zeros((1, latent_dim))
        # Decoder
        self.dec_w1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.dec_b1 = np.zeros((1, hidden_dim))
        self.dec_w2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.dec_b2 = np.zeros((1, input_dim))
        self.history = []

    def _relu(self, x): return np.maximum(0, x)
    def _sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def encode(self, X):
        h = self._relu(X @ self.enc_w1 + self.enc_b1)
        mu = h @ self.enc_w_mu + self.enc_b_mu
        logvar = h @ self.enc_w_logvar + self.enc_b_logvar
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def decode(self, z):
        h = self._relu(z @ self.dec_w1 + self.dec_b1)
        return self._sigmoid(h @ self.dec_w2 + self.dec_b2)

    def forward(self, X):
        mu, logvar = self.encode(X)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, X, X_recon, mu, logvar):
        recon_loss = np.mean((X - X_recon) ** 2)
        kl_loss = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))
        return recon_loss + kl_loss

    def fit(self, X, epochs=50, batch_size=32, verbose=False):
        n = X.shape[0]
        hist = []
        for ep in range(epochs):
            idx = np.random.permutation(n)
            ep_loss = 0.0
            nb = 0
            for st in range(0, n, batch_size):
                en = min(st + batch_size, n)
                bi = idx[st:en]
                Xb = X[bi]
                recon, mu, logvar = self.forward(Xb)
                loss = self.loss(Xb, recon, mu, logvar)
                ep_loss += loss
                nb += 1
                self._grad_update(Xb, recon, mu, logvar)
            hist.append(ep_loss / max(nb, 1))
            if verbose and (ep % 10 == 0 or ep == epochs-1):
                print("Epoch %d/%d loss=%.6f", ep, epochs, hist[-1])
        self.history = hist
        return dict(status="success", final_loss=hist[-1] if hist else None)

    def _grad_update(self, X, recon, mu, logvar):
        eps = 1e-4
        for W in [self.enc_w1, self.enc_w_mu, self.enc_w_logvar, self.dec_w1, self.dec_w2]:
            for i in range(min(2, W.shape[0])):
                for j in range(min(2, W.shape[1])):
                    old = W[i, j]
                    W[i, j] = old + eps
                    r, m, lv = self.forward(X)
                    lp = self.loss(X, r, m, lv)
                    W[i, j] = old - eps
                    r, m, lv = self.forward(X)
                    lm = self.loss(X, r, m, lv)
                    W[i, j] = old - self.lr * (lp - lm) / (2*eps)

    def generate(self, shape=None):
        if shape is None: shape = (1, self.latent_dim)
        z = np.random.randn(*shape)
        return self.decode(z)