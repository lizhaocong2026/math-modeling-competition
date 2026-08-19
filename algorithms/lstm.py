"""
LSTM (Long Short-Term Memory) network for time series prediction
Pure NumPy implementation for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List


class LSTM:
    """
    LSTM network for sequence prediction
    
    Suitable for: B题时间序列预测、股票预测、交通流量预测
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, 
                 output_dim: int = 1, seq_len: int = 24):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Forget gate
        self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bf = np.zeros(hidden_dim)
        
        # Input gate
        self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bi = np.zeros(hidden_dim)
        
        # Cell candidate
        self.Wc = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bc = np.zeros(hidden_dim)
        
        # Output gate
        self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bo = np.zeros(hidden_dim)
        
        # Output layer
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.01
        self.by = np.zeros(output_dim)
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def _forward_step(self, x_t: np.ndarray, h_prev: np.ndarray, 
                      c_prev: np.ndarray):
        """Forward pass for one time step"""
        xh = np.concatenate([x_t, h_prev])
        
        f = self._sigmoid(xh @ self.Wf.T + self.bf)
        i = self._sigmoid(xh @ self.Wi.T + self.bi)
        c = self._tanh(xh @ self.Wc.T + self.bc)
        o = self._sigmoid(xh @ self.Wo.T + self.bo)
        
        c_t = f * c_prev + i * c
        h_t = o * self._tanh(c_t)
        
        return h_t, c_t, (x_t, h_prev, c_prev, xh, f, i, c, o, c_t)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained LSTM
        
        Args:
            X: Input array (n_samples, seq_len, input_dim) or (seq_len, input_dim)
            
        Returns:
            Predictions
        """
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)
        
        predictions = []
        for sample in X:
            h = np.zeros(self.hidden_dim)
            c = np.zeros(self.hidden_dim)
            
            for t in range(min(self.seq_len, len(sample))):
                h, c, _ = self._forward_step(sample[t], h, c)
            
            pred = h @ self.Wy.T + self.by
            predictions.append(pred)
        
        return np.array(predictions).flatten()
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, 
            lr: float = 0.001) -> Dict[str, Any]:
        """
        Train LSTM model
        
        Args:
            X: Training data (n_samples, seq_len, input_dim)
            y: Targets (n_samples,)
            epochs: Training epochs
            lr: Learning rate
            
        Returns:
            Training result
        """
        self.history = []
        loss = 0.0
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                sample = X[i]
                target = y[i]
                
                # Forward pass
                h = np.zeros(self.hidden_dim)
                c = np.zeros(self.hidden_dim)
                
                for t in range(min(self.seq_len, len(sample))):
                    h, c, _ = self._forward_step(sample[t], h, c)
                
                # Output
                pred = h @ self.Wy.T + self.by
                error = pred - target
                
                # Update output layer
                self.Wy += lr * np.outer(error, h)
                self.by += lr * error
                
                total_loss += error ** 2
            
            loss = total_loss / len(X)
            self.history.append(loss)
        
        return {"status": "success", "final_loss": float(loss), "epochs": epochs}
    
    def get_params(self) -> Dict[str, int]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "seq_len": self.seq_len
        }


class GRU:
    """
    Gated Recurrent Unit (GRU) - simplified LSTM variant
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, 
                 output_dim: int = 1, seq_len: int = 24):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Update gate
        self.Wz = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bz = np.zeros(hidden_dim)
        
        # Reset gate
        self.Wr = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.br = np.zeros(hidden_dim)
        
        # Candidate hidden state
        self.Wh = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bh = np.zeros(hidden_dim)
        
        # Output layer
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.01
        self.by = np.zeros(output_dim)
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _forward_step(self, x_t: np.ndarray, h_prev: np.ndarray):
        xh = np.concatenate([x_t, h_prev])
        
        z = self._sigmoid(xh @ self.Wz.T + self.bz)
        r = self._sigmoid(xh @ self.Wr.T + self.br)
        h_candidate = np.tanh(xh @ self.Wh.T + self.bh)
        
        h_t = (1 - z) * h_prev + z * h_candidate
        return h_t, (x_t, h_prev, z, r, h_candidate)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)
        
        predictions = []
        for sample in X:
            h = np.zeros(self.hidden_dim)
            
            for t in range(min(self.seq_len, len(sample))):
                h, _ = self._forward_step(sample[t], h)
            
            pred = h @ self.Wy.T + self.by
            predictions.append(pred)
        
        return np.array(predictions).flatten()
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, 
            lr: float = 0.001) -> Dict[str, Any]:
        self.history = []
        loss = 0.0
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                sample = X[i]
                target = y[i]
                
                h = np.zeros(self.hidden_dim)
                for t in range(min(self.seq_len, len(sample))):
                    h, _ = self._forward_step(sample[t], h)
                
                pred = h @ self.Wy.T + self.by
                error = pred - target
                
                self.Wy += lr * np.outer(error, h)
                self.by += lr * error
                
                total_loss += error ** 2
            
            loss = total_loss / len(X)
            self.history.append(loss)
        
        return {"status": "success", "final_loss": float(loss), "epochs": epochs}


class LSTMEnsemble:
    """
    LSTM Ensemble for robust prediction with uncertainty quantification
    """
    
    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.models = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, 
            lr: float = 0.001):
        """Train ensemble of LSTM models"""
        self.models = []
        for i in range(self.n_models):
            hidden_dim = 32 + i * 16
            model = LSTM(input_dim=X.shape[-1], hidden_dim=hidden_dim,
                        seq_len=X.shape[1])
            model.fit(X, y, epochs=epochs, lr=lr)
            self.models.append(model)
        
        return {"status": "success", "n_models": self.n_models}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = [m.predict(X) for m in self.models]
        return np.mean(predictions, axis=0)
    
    def predict_with_std(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        predictions = [m.predict(X) for m in self.models]
        return {
            "mean": np.mean(predictions, axis=0),
            "std": np.std(predictions, axis=0)
        }
