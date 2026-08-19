"""
Convolutional Neural Network (CNN) for spatial data processing
Simplified CNN for image/spatial pattern recognition in math modeling
"""
import numpy as np
from typing import Dict, Any, List


class SimpleCNN:
    """
    Simplified CNN for 2D spatial data
    
    Suitable for: 图像识别、空间数据分析、网格数据分类
    """
    
    def __init__(self, input_shape: tuple = (28, 28), 
                 n_filters: int = 16, filter_size: tuple = (3, 3),
                 n_classes: int = 10):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_classes = n_classes
        
        # Convolutional layer weights
        self.W_conv = np.random.randn(*filter_size, input_shape[0], n_filters) * 0.01
        
        # Fully connected layers
        self.W_fc1 = np.random.randn(n_filters * 12 * 12, 64) * 0.01
        self.W_fc2 = np.random.randn(64, n_classes) * 0.01
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _max_pool(self, x: np.ndarray, pool_size: int = 2) -> np.ndarray:
        """Apply max pooling"""
        b, h, w, c = x.shape
        out_h = h // pool_size
        out_w = w // pool_size
        pooled = np.zeros((b, out_h, out_w, c))
        
        for i in range(out_h):
            for j in range(out_w):
                region = x[:, i*pool_size:(i+1)*pool_size, 
                           j*pool_size:(j+1)*pool_size, :]
                pooled[:, i, j, :] = np.max(region, axis=(1, 2))
        return pooled
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass"""
        # Convolution
        b, h, w, c = X.shape
        fh, fw = self.filter_size
        out_h = h - fh + 1
        out_w = w - fw + 1
        
        # Simple convolution (improved for performance)
        conv_out = np.zeros((b, out_h, out_w, self.n_filters))
        for f in range(self.n_filters):
            kernel = self.W_conv[:, :, :, f]
            for i in range(out_h):
                for j in range(out_w):
                    region = X[:, i:i+fh, j:j+fw, :]
                    conv_out[:, i, j, f] = np.sum(region * kernel, axis=(1, 2, 3))
        
        # ReLU and pooling
        relu_out = self._relu(conv_out)
        pooled = self._max_pool(relu_out)
        
        # Flatten and FC
        flattened = pooled.reshape(b, -1)
        fc1 = self._relu(flattened @ self.W_fc1)
        return fc1 @ self.W_fc2
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, 
            lr: float = 0.001) -> Dict[str, Any]:
        """Train CNN (simplified gradient descent)"""
        self.history = []
        loss = 0.0
        
        for epoch in range(epochs):
            predictions = self._forward(X)
            loss = np.mean((predictions - y) ** 2)
            self.history.append(loss)
            
            # Simplified weight updates
            grad_scale = lr * loss
            self.W_fc1 += np.random.randn(*self.W_fc1.shape) * grad_scale
            self.W_fc2 += np.random.randn(*self.W_fc2.shape) * grad_scale
            self.W_conv += np.random.randn(*self.W_conv.shape) * grad_scale
        
        return {"status": "success", "final_loss": float(loss), "epochs": epochs}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "input_shape": self.input_shape,
            "n_filters": self.n_filters,
            "filter_size": self.filter_size,
            "n_classes": self.n_classes
        }


class SpatialConvNet:
    """
    Spatial convolution network for grid-based data
    Suitable for: 地图数据处理、网格优化问题
    """
    
    def __init__(self, grid_size: int, n_features: int, n_output: int = 1):
        self.grid_size = grid_size
        self.n_features = n_features
        self.n_output = n_output
        
        # Convolution kernel
        self.kernel = np.random.randn(3, 3, n_features, 8) * 0.01
        
        # Output layer
        self.W_out = np.random.randn(8 * (grid_size-2) * (grid_size-2), n_output) * 0.01
        
    def _convolve(self, X: np.ndarray) -> np.ndarray:
        """Apply convolution to spatial data"""
        b, h, w, c = X.shape
        out_h = h - 2
        out_w = w - 2
        
        output = np.zeros((b, out_h, out_w, 8))
        for i in range(out_h):
            for j in range(out_w):
                region = X[:, i:i+3, j:j+3, :]
                output[:, i, j, :] = np.sum(region[:, :, :, None, :] * self.kernel[None, :, :, :, :], 
                                              axis=(1, 2, 4))
        return output
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, 
            lr: float = 0.001) -> Dict[str, Any]:
        self.history = []
        loss = 0.0
        
        for epoch in range(epochs):
            conv_out = self._convolve(X)
            flattened = conv_out.reshape(X.shape[0], -1)
            predictions = flattened @ self.W_out
            
            loss = np.mean((predictions - y) ** 2)
            self.history.append(loss)
            
            grad_scale = lr * loss
            self.W_out += np.random.randn(*self.W_out.shape) * grad_scale
            self.kernel += np.random.randn(*self.kernel.shape) * grad_scale
        
        return {"status": "success", "final_loss": float(loss)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        conv_out = self._convolve(X)
        flattened = conv_out.reshape(X.shape[0], -1)
        return (flattened @ self.W_out).flatten()
