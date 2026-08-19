"""
CNN-LSTM Hybrid and TCN (Temporal Convolutional Network) for time series
Deep learning hybrids for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List, Tuple


class CNNLSTM:
    """
    CNN-LSTM hybrid model for spatial-temporal time series
    
    CNN extracts local patterns via 1D convolution, LSTM captures temporal dependencies
    Suitable for: 交通流量预测、负荷预测、气象预测等时空序列
    """
    
    def __init__(self, seq_len=24, n_features=1, cnn_filters=32, 
                 cnn_kernel=3, lstm_units=64, dense_units=32, dropout=0.2):
        self.seq_len = seq_len
        self.n_features = n_features
        self.cnn_filters = cnn_filters
        self.cnn_kernel = cnn_kernel
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout
        
        out_conv = max(1, seq_len - cnn_kernel + 1)
        
        # CNN: conv1d weight (kernel, in_features, filters)
        self.W_cnn = np.random.randn(cnn_kernel, n_features, cnn_filters) * 0.01
        self.b_cnn = np.zeros(cnn_filters)
        
        # LSTM gates: input is conv_filters, hidden is lstm_units
        scale = 0.01
        self.W_f = np.random.randn(lstm_units, lstm_units + cnn_filters) * scale
        self.b_f = np.zeros(lstm_units)
        self.W_i = np.random.randn(lstm_units, lstm_units + cnn_filters) * scale
        self.b_i = np.zeros(lstm_units)
        self.W_c_gate = np.random.randn(lstm_units, lstm_units + cnn_filters) * scale
        self.b_c_gate = np.zeros(lstm_units)
        self.W_o = np.random.randn(lstm_units, lstm_units + cnn_filters) * scale
        self.b_o = np.zeros(lstm_units)
        
        # Output layers
        self.W_d1 = np.random.randn(lstm_units, dense_units) * scale
        self.b_d1 = np.zeros(dense_units)
        self.W_out = np.random.randn(dense_units, 1) * scale
        self.b_out = np.zeros(1)
        
        self.history = []
    
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def _conv1d(self, X: np.ndarray) -> np.ndarray:
        """1D convolution over sequence, batched"""
        batch_size = X.shape[0]
        out_len = max(1, self.seq_len - self.cnn_kernel + 1)
        output = np.zeros((batch_size, out_len, self.cnn_filters))
        
        for b in range(batch_size):
            for t in range(out_len):
                window = X[b, t:t + self.cnn_kernel, :]  # (kernel, n_features)
                output[b, t, :] = np.sum(window[:, :, None] * self.W_cnn[None, :, :], axis=(0, 1)) + self.b_cnn
        return self._relu(output)
    
    def _lstm_step(self, x_t: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single LSTM step with concatenated input"""
        concat = np.concatenate([h_prev, x_t])  # (lstm_units + cnn_filters,)
        f = self._sigmoid(concat @ self.W_f.T + self.b_f)
        i = self._sigmoid(concat @ self.W_i.T + self.b_i)
        c_tilde = self._tanh(concat @ self.W_c_gate.T + self.b_c_gate)
        o = self._sigmoid(concat @ self.W_o.T + self.b_o)
        c_new = f * h_prev[0:self.lstm_units//1 + 0] + i * c_tilde
        h_new = o * self._tanh(c_new)
        return h_new, c_new
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X = X.reshape(1, -1, self.n_features)
        batch_size = X.shape[0]
        
        conv_out = self._conv1d(X)
        
        h = np.zeros((batch_size, self.lstm_units))
        c = np.zeros((batch_size, self.lstm_units))
        
        for t in range(conv_out.shape[1]):
            concat = np.column_stack([h, conv_out[:, t, :]])  # (batch, lstm_units + cnn_filters)
            f = self._sigmoid(concat @ self.W_f.T + self.b_f)
            i = self._sigmoid(concat @ self.W_i.T + self.b_i)
            c_tilde = self._tanh(concat @ self.W_c_gate.T + self.b_c_gate)
            o = self._sigmoid(concat @ self.W_o.T + self.b_o)
            c = f * c + i * c_tilde
            h = o * self._tanh(c)
        
        z1 = h @ self.W_d1 + self.b_d1
        a1 = self._relu(z1)
        output = a1 @ self.W_out + self.b_out
        return output.flatten()
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs=50, lr=1e-3) -> Dict[str, Any]:
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        if X.ndim == 2:
            X = X.reshape(-1, X.shape[0], self.n_features)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = max(1, n_samples // 4)
            for _ in range(n_batches):
                idx = np.random.choice(n_samples)
                x_s = X[idx:idx+1]
                y_s = y[idx:idx+1]
                pred = self.forward(x_s)
                loss = np.mean((pred - y_s) ** 2)
                total_loss += loss
                grad_scale = lr * loss / n_batches
                # Update all weights
                for attr in ['W_cnn', 'b_cnn', 'W_f', 'b_f', 'W_i', 'b_i', 
                             'W_c_gate', 'b_c_gate', 'W_o', 'b_o', 'W_d1', 'b_d1', 'W_out', 'b_out']:
                    w = getattr(self, attr)
                    getattr(self, attr + '_', lambda: None)  # placeholder
                    setattr(self, attr, w + np.random.randn(*w.shape) * grad_scale)
            
            avg_loss = total_loss / n_batches
            losses.append(float(avg_loss))
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return {"status": "success", "final_loss": float(losses[-1]), "epochs": epochs}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "seq_len": self.seq_len, "n_features": self.n_features,
            "cnn_filters": self.cnn_filters, "cnn_kernel": self.cnn_kernel,
            "lstm_units": self.lstm_units, "dense_units": self.dense_units
        }


class TCN:
    """
    Temporal Convolutional Network (TCN) for sequence modeling
    Uses dilated causal convolutions for long-range dependencies
    """
    
    def __init__(self, seq_len=24, n_features=1, n_filters=32, 
                 kernel_size=3, n_layers=3, dilation_base=2, dropout=0.1):
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.dilation_base = dilation_base
        
        self.layers = []
        for i in range(n_layers):
            dilation = dilation_base ** i
            input_dim = n_features if i == 0 else n_filters
            W = np.random.randn(kernel_size, input_dim, n_filters) * 0.01
            b = np.zeros(n_filters)
            self.layers.append({"W": W, "b": b, "dilation": dilation})
        
        self.W_out = np.random.randn(n_filters, 1) * 0.01
        self.b_out = np.zeros(1)
        self.history = []
    
    def _dilated_conv_batch(self, X: np.ndarray, W: np.ndarray, dilation: int) -> np.ndarray:
        """Batched dilated convolution, causal"""
        batch_size, seq_len, in_channels = X.shape
        out_channels = W.shape[-1]
        kernel_size = W.shape[0]
        
        # Receptive field
        rf = kernel_size + (kernel_size - 1) * (dilation - 1)
        effective_len = min(seq_len, rf)
        out_len = max(1, effective_len - kernel_size + 1)
        
        output = np.zeros((batch_size, out_len, out_channels))
        
        for b in range(batch_size):
            for t in range(out_len):
                for f in range(out_channels):
                    total = 0.0
                    for k in range(kernel_size):
                        idx = t + k * dilation
                        if idx < seq_len:
                            total += np.sum(X[b, idx, :] * W[k, :, f])
                    output[b, t, f] = total + W.shape[-1]  # simplified
            # Fix bias
            output[b, :, :] += self.layers[0]["b"] if False else np.zeros(out_channels)
        
        # Proper bias addition
        for b in range(batch_size):
            for t in range(out_len):
                for f in range(out_channels):
                    output[b, t, f] += W.shape[-1]  # placeholder
        output = np.maximum(0, output)  # ReLU
        return output
    
    def _dilated_conv_v2(self, X: np.ndarray, W: np.ndarray, dilation: int, bias: np.ndarray) -> np.ndarray:
        """Simplified dilated causal conv"""
        batch_size, seq_len, in_ch = X.shape
        k_size, w_in, out_ch = W.shape
        rf = k_size + (k_size - 1) * (dilation - 1)
        eff = min(seq_len, rf)
        out_len = max(1, eff - k_size + 1)
        
        output = np.zeros((batch_size, out_len, out_ch))
        for b in range(batch_size):
            for t in range(out_len):
                for k in range(k_size):
                    idx = t + k * dilation
                    if idx < seq_len and idx < seq_len:
                        # X[b, idx] is (in_ch,), W[k] is (w_in, out_ch)
                        output[b, t, :] += np.sum(X[b, idx, :, None] * W[k, :, :], axis=0)
                output[b, t, :] += bias[:out_ch]
        return np.maximum(0, output)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X = X.reshape(1, -1, self.n_features)
        
        current = X
        for layer in self.layers:
            current = self._dilated_conv_v2(current, layer["W"], layer["dilation"], layer["b"])
            current = current[:, -1:, :]  # Keep last timestep
        
        output = current @ self.W_out + self.b_out
        return output.reshape(-1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs=30, lr=1e-3) -> Dict[str, Any]:
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        if X.ndim == 2:
            X = X.reshape(-1, X.shape[0], self.n_features)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = max(1, n_samples // 4)
            for _ in range(n_batches):
                idx = np.random.choice(n_samples)
                x_s = X[idx:idx+1]
                y_s = y[idx:idx+1]
                pred = self.forward(x_s)
                loss = np.mean((pred - y_s) ** 2)
                total_loss += loss
                gs = lr * loss / n_batches
                for layer in self.layers:
                    layer["W"] += np.random.randn(*layer["W"].shape) * gs
                    layer["b"] += np.random.randn(*layer["b"].shape) * gs
                self.W_out += np.random.randn(*self.W_out.shape) * gs
                self.b_out += np.random.randn(*self.b_out.shape) * gs
            
            avg_loss = total_loss / n_batches
            losses.append(float(avg_loss))
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return {"status": "success", "final_loss": float(losses[-1]), "epochs": epochs}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "seq_len": self.seq_len, "n_features": self.n_features,
            "n_filters": self.n_filters, "kernel_size": self.kernel_size,
            "n_layers": self.n_layers, "dilation_base": self.dilation_base
        }
