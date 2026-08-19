import numpy as np
from typing import Dict, Any


class NeuralODE:
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=1, dt=0.01, lr=1e-3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.lr = lr
        # Neural network for dxdot = f(x, theta)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b2 = np.zeros((1, input_dim))
        self.W_out = np.random.randn(input_dim, output_dim) * 0.01
        self.b_out = np.zeros((1, output_dim))
        self.history = []

    def _relu(self, x): return np.maximum(0, x)

    def _f(self, x):
        h = self._relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def forward(self, X):
        results = [X]
        for t in range(int(1.0 / max(self.dt, 1e-6))):
            x = results[-1]
            dxdt = self._f(x)
            x_next = x + self.dt * dxdt
            results.append(x_next)
        return results[-1] @ self.W_out + self.b_out

    def predict(self, X): return self.forward(X)

    def fit(self, X, y, epochs=50, verbose=False):
        n = X.shape[0]
        hist = []
        for ep in range(epochs):
            pred = self.forward(X)
            loss = np.mean((pred - y)**2)
            hist.append(loss)
            self._grad_update(X, y, loss)
            if verbose and (ep % 10 == 0 or ep == epochs-1):
                print("Epoch %d/%d loss=%.6f", ep, epochs, loss)
        self.history = hist
        return {"status": "success", "final_loss": hist[-1] if hist else None}

    def _grad_update(self, X, y, loss):
        eps = 1e-4
        for W in [self.W1, self.W2, self.W_out]:
            for i in range(min(2, W.shape[0])):
                for j in range(min(2, W.shape[1])):
                    old = W[i, j]
                    W[i, j] = old + eps
                    lp = np.mean((self.forward(X) - y)**2)
                    W[i, j] = old - eps
                    lm = np.mean((self.forward(X) - y)**2)
                    W[i, j] = old - self.lr * (lp - lm) / (2*eps)