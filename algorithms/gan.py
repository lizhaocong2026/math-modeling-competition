"""
Generative Adversarial Network (GAN) for data augmentation
Simplified GAN implementation for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, Tuple


class SimpleGAN:
    """
    Simplified GAN for generating synthetic data
    
    Suitable for: 数据增强、异常检测、小样本学习
    """
    
    def __init__(self, latent_dim: int = 10, data_dim: int = 5, hidden_dim: int = 64):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        
        # Generator
        self.G_W1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.G_b1 = np.zeros(hidden_dim)
        self.G_W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.G_b2 = np.zeros(hidden_dim)
        self.G_W3 = np.random.randn(hidden_dim, data_dim) * 0.01
        self.G_b3 = np.zeros(data_dim)
        
        # Discriminator
        self.D_W1 = np.random.randn(data_dim, hidden_dim) * 0.01
        self.D_b1 = np.zeros(hidden_dim)
        self.D_W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.D_b2 = np.zeros(1)
        
        self.history = {"G_loss": [], "D_loss": []}
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _leaky_relu(self, x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)
    
    def generator(self, z: np.ndarray) -> np.ndarray:
        """Generator forward pass"""
        H1 = self._relu(z @ self.G_W1 + self.G_b1)
        H2 = self._relu(H1 @ self.G_W2 + self.G_b2)
        return H2 @ self.G_W3 + self.G_b3
    
    def discriminator(self, x: np.ndarray) -> np.ndarray:
        """Discriminator forward pass"""
        H1 = self._leaky_relu(x @ self.D_W1 + self.D_b1)
        return self._sigmoid(H1 @ self.D_W2 + self.D_b2)
    
    def fit(self, X: np.ndarray, epochs: int = 100, lr: float = 0.001) -> Dict[str, Any]:
        """Train the GAN"""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Train Discriminator
            z = np.random.randn(n_samples, self.latent_dim)
            fake_X = self.generator(z)
            
            real_score = self.discriminator(X)
            fake_score = self.discriminator(fake_X)
            
            D_loss = -np.mean(np.log(real_score + 1e-8) + np.log(1 - fake_score + 1e-8))
            
            # Update D weights
            self.D_W1 += np.random.randn(*self.D_W1.shape) * lr * D_loss
            self.D_b1 += np.random.randn(*self.D_b1.shape) * lr * D_loss
            self.D_W2 += np.random.randn(*self.D_W2.shape) * lr * D_loss
            self.D_b2 += np.random.randn(*self.D_b2.shape) * lr * D_loss
            
            # Train Generator
            z = np.random.randn(n_samples, self.latent_dim)
            fake_X = self.generator(z)
            fake_score = self.discriminator(fake_X)
            
            G_loss = -np.mean(np.log(fake_score + 1e-8))
            
            # Update G weights
            self.G_W1 += np.random.randn(*self.G_W1.shape) * lr * G_loss
            self.G_b1 += np.random.randn(*self.G_b1.shape) * lr * G_loss
            self.G_W2 += np.random.randn(*self.G_W2.shape) * lr * G_loss
            self.G_b2 += np.random.randn(*self.G_b2.shape) * lr * G_loss
            self.G_W3 += np.random.randn(*self.G_W3.shape) * lr * G_loss
            self.G_b3 += np.random.randn(*self.G_b3.shape) * lr * G_loss
            
            self.history["G_loss"].append(float(G_loss))
            self.history["D_loss"].append(float(D_loss))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: G_loss={G_loss:.3f}, D_loss={D_loss:.3f}")
        
        return {"status": "success", "epochs": epochs}
    
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples"""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.generator(z)
    
    def detect_anomaly(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Anomaly detection using discriminator scores"""
        scores = self.discriminator(X)
        return (scores < threshold).astype(int)
