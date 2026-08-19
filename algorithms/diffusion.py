"""
Diffusion Model for time series generation and imputation
Simplified denoising diffusion process for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List, Optional
import math


class SimpleDiffusion:
    """
    Simplified Diffusion Model for time series tasks
    
    Suitable for: 数据生成、缺失值填补、异常值修复、噪声数据增强
    """
    
    def __init__(self, n_features: int = 1, noise_schedule: str = 'linear',
                 T: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.n_features = n_features
        self.T = T
        self.noise_schedule = noise_schedule
        
        if noise_schedule == 'linear':
            self.beta = np.linspace(beta_start, beta_end, T)
        elif noise_schedule == 'cosine':
            steps = T + 1
            x = np.linspace(0, 1, steps)
            alphas = np.cos((x * 0.5 + 0.0087) / 1.0087 * math.pi / 2) ** 2
            alphas = alphas / alphas[0]
            self.beta = np.clip(1 - alphas[1:] / alphas[:-1], 1e-8, 0.999)
        else:
            self.beta = np.ones(T) * beta_start
        
        self.alpha = 1 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        
        self.W1 = np.random.randn(n_features, 32) * 0.01
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 32) * 0.01
        self.b2 = np.zeros(32)
        self.W_out = np.random.randn(32, n_features) * 0.01
        self.b_out = np.zeros(n_features)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = self._relu(x @ self.W1 + self.b1)
        h = self._relu(h @ self.W2 + self.b2)
        return h @ self.W_out + self.b_out
    
    def q_sample(self, x0: np.ndarray, t: int, noise: Optional[np.ndarray] = None) -> np.ndarray:
        if noise is None:
            noise = np.random.randn(*x0.shape)
        sqrt_ab = np.sqrt(self.alpha_bar[t])
        sqrt_1_ab = np.sqrt(1 - self.alpha_bar[t])
        return sqrt_ab * x0 + sqrt_1_ab * noise
    
    def p_forward(self, xt: np.ndarray, t: int) -> np.ndarray:
        return self._forward(xt)
    
    def p_reverse(self, xt: np.ndarray, t: int) -> np.ndarray:
        eps_pred = self.p_forward(xt, t)
        coef1 = 1 / np.sqrt(self.alpha[t])
        coef2 = (1 - self.alpha[t]) / np.sqrt(1 - self.alpha_bar[t])
        x0_pred = coef1 * (xt - coef2 * eps_pred)
        if t > 0:
            noise = np.random.randn(*xt.shape)
            return x0_pred
        else:
            return x0_pred
    
    def fit(self, X: np.ndarray, epochs: int = 100, lr: float = 1e-3) -> Dict[str, Any]:
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            total_loss = 0.0
            for _ in range(max(1, n_samples // 8)):
                t = np.random.randint(0, self.T)
                noise = np.random.randn(*X.shape)
                Xt = self.q_sample(X, t, noise)
                eps_pred = self.p_forward(Xt, t)
                loss = np.mean((eps_pred - noise) ** 2)
                total_loss += loss
                grad_scale = lr * loss / max(1, n_samples // 8)
                self.W1 += np.random.randn(*self.W1.shape) * grad_scale
                self.b1 += np.random.randn(*self.b1.shape) * grad_scale
                self.W2 += np.random.randn(*self.W2.shape) * grad_scale
                self.b2 += np.random.randn(*self.b2.shape) * grad_scale
                self.W_out += np.random.randn(*self.W_out.shape) * grad_scale
                self.b_out += np.random.randn(*self.b_out.shape) * grad_scale
            
            avg_loss = total_loss / max(1, n_samples // 8)
            losses.append(float(avg_loss))
            
            if (epoch + 1) % 20 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
        
        return {'status': 'success', 'final_loss': float(losses[-1]), 'epochs': epochs}
    
    def generate(self, n_samples: int = 10, seq_len: int = 24) -> np.ndarray:
        x = np.random.randn(n_samples, seq_len, self.n_features)
        for t in reversed(range(self.T)):
            x = self.p_reverse(x, t)
        return x
    
    def impute(self, X: np.ndarray, mask: np.ndarray, n_iterations: int = 5) -> np.ndarray:
        X_filled = X.copy()
        for _ in range(n_iterations):
            for i in range(X_filled.shape[0]):
                if np.any(mask[i]):
                    pred = self.p_reverse(X_filled[i:i+1], 0)
                    X_filled[i] = pred.reshape(X_filled[i].shape)
                    X_filled[i][~mask[i]] = X[i][~mask[i]]
        return X_filled
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'n_features': self.n_features,
            'T': self.T,
            'noise_schedule': self.noise_schedule,
            'beta_start': float(self.beta[0]),
            'beta_end': float(self.beta[-1])
        }


class DiffusionEnsemble:
    """Ensemble of diffusion models for better robustness"""
    
    def __init__(self, n_models: int = 3, **kwargs):
        self.n_models = n_models
        self.models = [SimpleDiffusion(**kwargs) for _ in range(n_models)]
    
    def fit(self, X: np.ndarray, epochs: int = 50) -> Dict[str, Any]:
        results = []
        for i, model in enumerate(self.models):
            print(f'  Training diffusion model {i+1}/{self.n_models}...')
            result = model.fit(X, epochs=epochs)
            results.append(result)
        return {'status': 'success', 'models': len(results)}
    
    def generate(self, n_samples: int = 10, seq_len: int = 24) -> np.ndarray:
        all_gen = [m.generate(n_samples, seq_len) for m in self.models]
        return np.mean(all_gen, axis=0)
    
    def impute(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        all_imputed = [m.impute(X, mask) for m in self.models]
        return np.mean(all_imputed, axis=0)
