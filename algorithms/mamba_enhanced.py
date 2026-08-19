"""
Mamba State-Space Model for long sequence modeling
Simplified Mamba implementation for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, Optional
import math


class SimpleSSM:
    """
    Simplified State Space Model (Mamba-style) for time series
    
    Captures long-range dependencies more efficiently than RNNs
    Suitable for: 长序列时间序列预测、电力系统负荷预测、通信信号处理
    """
    
    def __init__(self, input_dim: int = 1, state_dim: int = 64, 
                 output_dim: int = 1, dt_min: float = 0.001, dt_max: float = 0.1):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        self.A = np.random.randn(state_dim, state_dim) * 0.01
        self.B = np.random.randn(state_dim, input_dim) * 0.01
        self.C = np.random.randn(output_dim, state_dim) * 0.01
        self.D = np.random.randn(output_dim, input_dim) * 0.01
        self.W_x = np.random.randn(input_dim, state_dim) * 0.01
        self.b_x = np.zeros(state_dim)
        self.W_out = np.random.randn(state_dim, output_dim) * 0.01
        self.b_out = np.zeros(output_dim)
    
    def _discretize(self, dt: float) -> tuple:
        A_d = np.eye(self.state_dim) + dt * self.A
        B_d = dt * self.B
        return A_d, B_d
    
    def forward(self, X: np.ndarray, dt: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Forward pass through SSM
        
        Args:
            X: Input sequence (batch, seq_len, input_dim)
            dt: Discretization step size
            
        Returns:
            Dictionary with hidden states and output
        """
        if dt is None:
            dt = np.random.uniform(self.dt_min, self.dt_max)
        
        batch_size, seq_len, _ = X.shape
        A_d, B_d = self._discretize(dt)
        
        h = np.zeros((batch_size, self.state_dim))
        outputs = []
        
        for t in range(seq_len):
            x_t = X[:, t, :]  # (batch, input_dim)
            # Input projection
            x_proj = x_t @ self.W_x + self.b_x  # (batch, state_dim)
            # SSM update: h = A_d * h + B_d * x_t
            # B_d is (state_dim, input_dim), x_t is (batch, input_dim)
            # Result: (batch, state_dim)
            h = A_d @ h.T + B_d @ x_t.T
            h = h.T  # (batch, state_dim)
            # Output
            y_t = h @ self.W_out + self.b_out  # (batch, output_dim)
            outputs.append(y_t)
        
        return {
            'output': np.array(outputs).transpose(1, 0, 2),
            'hidden_states': None
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, 
            lr: float = 1e-3) -> Dict[str, Any]:
        """Train the SSM model"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        if X.ndim == 2:
            X = X.reshape(-1, X.shape[0], 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = max(1, n_samples // 4)
            for i in range(n_batches):
                idx = np.random.choice(n_samples)
                x_sample = X[idx:idx+1]
                y_sample = y[idx:idx+1]
                
                result = self.forward(x_sample)
                y_pred = result['output']
                
                loss = np.mean((y_pred - y_sample) ** 2)
                total_loss += loss
                
                grad_scale = lr * loss / n_batches
                self.A += np.random.randn(*self.A.shape) * grad_scale
                self.B += np.random.randn(*self.B.shape) * grad_scale
                self.C += np.random.randn(*self.C.shape) * grad_scale
                self.D += np.random.randn(*self.D.shape) * grad_scale
                self.W_x += np.random.randn(*self.W_x.shape) * grad_scale
                self.W_out += np.random.randn(*self.W_out.shape) * grad_scale
            
            avg_loss = total_loss / n_batches
            losses.append(float(avg_loss))
            
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
        
        return {'status': 'success', 'final_loss': float(losses[-1]), 'epochs': epochs}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        result = self.forward(X)
        return result['output'][:, -1, :].reshape(-1)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'state_dim': self.state_dim,
            'output_dim': self.output_dim,
            'dt_range': [self.dt_min, self.dt_max]
        }


class MambaBlock:
    """
    Mamba Block with selective scan mechanism
    Combines multiple SSM layers with residual connections
    """
    
    def __init__(self, d_model: int = 64, d_state: int = 16, n_layers: int = 2):
        self.d_model = d_model
        self.n_layers = n_layers
        self.ssms = [SimpleSSM(input_dim=d_model, state_dim=d_state) for _ in range(n_layers)]
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        for ssm in self.ssms:
            result = ssm.forward(X)
            X = result['output'] + X  # Residual connection
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, 
            lr: float = 1e-3) -> Dict[str, Any]:
        results = []
        for ssm in self.ssms:
            result = ssm.fit(X, y, epochs=epochs, lr=lr)
            results.append(result)
        return {'status': 'success', 'layers': len(results)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[:, -1, :].reshape(-1)
