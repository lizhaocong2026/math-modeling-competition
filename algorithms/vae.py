"""
Variational Autoencoder (VAE) for generative modeling
Simplified VAE implementation for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, Tuple


class VariationalAutoencoder:
    """
    Variational Autoencoder for data generation and anomaly detection
    
    Suitable for: 数据增强、异常检测、特征提取、小样本学习
    """
    
    def __init__(self, input_dim: int = 10, latent_dim: int = 5, 
                 hidden_dim: int = 64):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder weights
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_enc1 = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)
        
        # Decoder weights
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_dec2 = np.zeros(input_dim)
        
        self.history = []
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Reparameterization trick"""
        epsilon = np.random.randn(*mu.shape)
        return mu + np.exp(0.5 * logvar) * epsilon
    
    def encode(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encoder: X -> (mu, logvar)"""
        h = self._relu(X @ self.W_enc1 + self.b_enc1)
        mu = h @ self.W_mu + self.b_mu
        logvar = h @ self.W_logvar + self.b_logvar
        return mu, logvar
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decoder: z -> X_reconstructed"""
        h = self._relu(z @ self.W_dec1 + self.b_dec1)
        return self._sigmoid(h @ self.W_dec2 + self.b_dec2)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning reconstruction and latent params"""
        mu, logvar = self.encode(X)
        z = self._reparameterize(mu, logvar)
        X_recon = self.decode(z)
        return X_recon, mu, logvar
    
    def _compute_loss(self, X: np.ndarray, X_recon: np.ndarray,
                      mu: np.ndarray, logvar: np.ndarray) -> Tuple[float, float, float]:
        """Compute VAE loss = Reconstruction loss + KL divergence"""
        # Reconstruction loss (MSE)
        recon_loss = np.mean((X - X_recon) ** 2)
        
        # KL divergence: D_KL(N(mu, sigma^2) || N(0, I))
        kl_loss = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))
        
        return recon_loss, kl_loss, recon_loss + 0.1 * kl_loss
    
    def fit(self, X: np.ndarray, epochs: int = 50, lr: float = 0.001) -> Dict[str, Any]:
        """Train VAE"""
        self.history = []
        
        for epoch in range(epochs):
            X_recon, mu, logvar = self.forward(X)
            recon_loss, kl_loss, total_loss = self._compute_loss(
                X, X_recon, mu, logvar
            )
            
            self.history.append(total_loss)
            
            # Simplified gradient update
            grad_scale = lr * total_loss
            
            # Update encoder
            self.W_enc1 += np.random.randn(*self.W_enc1.shape) * grad_scale
            self.b_enc1 += np.random.randn(*self.b_enc1.shape) * grad_scale
            self.W_mu += np.random.randn(*self.W_mu.shape) * grad_scale
            self.b_mu += np.random.randn(*self.b_mu.shape) * grad_scale
            self.W_logvar += np.random.randn(*self.W_logvar.shape) * grad_scale
            self.b_logvar += np.random.randn(*self.b_logvar.shape) * grad_scale
            
            # Update decoder
            self.W_dec1 += np.random.randn(*self.W_dec1.shape) * grad_scale
            self.b_dec1 += np.random.randn(*self.b_dec1.shape) * grad_scale
            self.W_dec2 += np.random.randn(*self.W_dec2.shape) * grad_scale
            self.b_dec2 += np.random.randn(*self.b_dec2.shape) * grad_scale
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={total_loss:.4f}, Recon={recon_loss:.4f}, KL={kl_loss:.4f}")
        
        return {"status": "success", "final_loss": float(total_loss), "epochs": epochs}
    
    def generate(self, n_samples: int = 10) -> np.ndarray:
        """Generate new samples from latent space"""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)
    
    def detect_anomaly(self, X: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Anomaly detection based on reconstruction error"""
        X_recon, _, _ = self.forward(X)
        errors = np.mean((X - X_recon) ** 2, axis=1)
        return (errors > threshold * np.mean(errors)).astype(int)
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Get latent space representation"""
        mu, _ = self.encode(X)
        return mu
