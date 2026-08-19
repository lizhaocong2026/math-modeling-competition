import numpy as np
from typing import Dict, Any


class MoEModel:
    def __init__(self, input_dim=64, num_experts=4, expert_dim=32, output_dim=1, lr=0.01):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.output_dim = output_dim
        self.lr = lr
        self.expert_w1, self.expert_w2 = [], []
        self.expert_b1, self.expert_b2 = [], []
        for _ in range(num_experts):
            self.expert_w1.append(np.random.randn(input_dim, expert_dim) * 0.01)
            self.expert_b1.append(np.zeros((1, expert_dim)))
            self.expert_w2.append(np.random.randn(expert_dim, output_dim) * 0.01)
            self.expert_b2.append(np.zeros((1, output_dim)))
        self.gate_w = np.random.randn(input_dim, num_experts) * 0.01
        self.gate_b = np.zeros((1, num_experts))
        self.history = []

    def _relu(self, x): return np.maximum(0, x)
    def _relu_d(self, x): return (x > 0).astype(float)
    def _softmax(self, x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (e.sum(axis=axis, keepdims=True) + 1e-10)

    def forward(self, X):
        gate_in = X @ self.gate_w + self.gate_b
        gate_out = self._softmax(gate_in)
        expert_outs, h1_list = [], []
        for i in range(self.num_experts):
            h1 = self._relu(X @ self.expert_w1[i] + self.expert_b1[i])
            h1_list.append(h1)
            expert_outs.append(h1 @ self.expert_w2[i] + self.expert_b2[i])
        expert_stack = np.stack(expert_outs, axis=1)
        output = np.sum(gate_out[:, :, np.newaxis] * expert_stack, axis=1)
        self._gate_out, self._expert_outs, self._gate_in = gate_out, expert_stack, gate_in
        self._h1_list = h1_list
        return output

    def fit(self, X, y, epochs=50, batch_size=32, verbose=False):
        n, mse_hist = X.shape[0], []
        for ep in range(epochs):
            idx = np.random.permutation(n)
            ep_loss, nb = 0.0, 0
            for st in range(0, n, batch_size):
                en = min(st + batch_size, n)
                bi = idx[st:en]
                Xb, yb = X[bi], y[bi]
                err = self.forward(Xb) - yb
                ep_loss += np.mean(err ** 2)
                nb += 1
                self._backward(Xb, yb, err)
            mse_hist.append(ep_loss / max(nb, 1))
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f'Epoch {ep}/{epochs} MSE={mse_hist[-1]:.6f}')
        self.history = mse_hist
        return {'status': 'success', 'final_mse': mse_hist[-1] if mse_hist else None}

    def _backward(self, X, y, error):
        batch = X.shape[0]
        d_out = 2.0 * error / (batch * self.output_dim)
        go, eo, gi = self._gate_out, self._expert_outs, self._gate_in
        d_gate_raw = np.sum(d_out[:, :, np.newaxis] * eo, axis=2)
        go_s = self._softmax(gi)
        d_gate_sm = go_s * (d_gate_raw - np.sum(d_gate_raw * go_s, axis=1, keepdims=True))
        self.gate_w -= self.lr * (X.T @ d_gate_sm) / batch
        self.gate_b -= self.lr * np.mean(d_gate_sm, axis=0, keepdims=True)
        for i in range(self.num_experts):
            d_eo = d_out * go[:, i:i+1]
            h1 = self._h1_list[i]
            z1 = X @ self.expert_w1[i] + self.expert_b1[i]
            d_h1 = (d_eo @ self.expert_w2[i].T) * self._relu_d(z1)
            self.expert_w1[i] -= self.lr * (X.T @ d_h1) / batch
            self.expert_b1[i] -= self.lr * np.mean(d_h1, axis=0, keepdims=True)
            self.expert_w2[i] -= self.lr * (h1.T @ d_eo) / batch
            self.expert_b2[i] -= self.lr * np.mean(d_eo, axis=0, keepdims=True)

    def predict(self, X): return self.forward(X)

    def get_contributions(self, X):
        out = self.forward(X)
        return {'output': out, 'gate_weights': self._gate_out,
                'expert_contributions': self._gate_out[:, :, np.newaxis] * self._expert_outs}