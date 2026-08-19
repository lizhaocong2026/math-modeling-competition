"""
简单神经网络层 (用于模式识别和回归)
纯NumPy实现，不依赖深度学习框架
"""
import numpy as np
from typing import Optional, Dict, Any, List


class NeuralNetwork:
    """简单前馈神经网络"""
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        learning_rate: float = 0.01,
        verbose: bool = False
    ):
        """
        初始化神经网络
        
        参数:
            layer_sizes: 每层神经元数量 [输入, 隐藏1, 隐藏2, ..., 输出]
            activation: 激活函数 ('relu', 'sigmoid', 'tanh')
            learning_rate: 学习率
            verbose: 是否打印训练过程
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He初始化
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        self.loss_history = []
        
    def _activate(self, z: np.ndarray) -> np.ndarray:
        """激活函数"""
        if self.activation_name == "relu":
            return np.maximum(0, z)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_name == "tanh":
            return np.tanh(z)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation_name}")
    
    def _activate_derivative(self, a: np.ndarray) -> np.ndarray:
        """激活函数导数"""
        if self.activation_name == "relu":
            return (a > 0).astype(float)
        elif self.activation_name == "sigmoid":
            return a * (1 - a)
        elif self.activation_name == "tanh":
            return 1 - a ** 2
        else:
            raise ValueError(f"不支持的激活函数: {self.activation_name}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            # 最后一层使用sigmoid（二分类）或恒等（回归）
            if i == len(self.weights) - 1:
                current = z  # 线性输出用于回归
            else:
                current = self._activate(z)
            
            self.activations.append(current)
        
        return current
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 32,
        verbose: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        训练神经网络
        
        参数:
            X: 输入数据
            y: 目标数据
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印进度
            
        返回:
            训练历史记录
        """
        verbose = verbose if verbose is not None else self.verbose
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        n_samples = X.shape[0]
        self.loss_history = []
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # 批次训练
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # 前向传播
                output = self.forward(X_batch)
                
                # 计算损失 (MSE)
                loss = np.mean((output - y_batch) ** 2)
                epoch_loss += loss
                n_batches += 1
                
                # 反向传播
                self._backpropagate(X_batch, y_batch, output)
            
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        return {
            "loss_history": self.loss_history,
            "final_loss": float(self.loss_history[-1]) if self.loss_history else None,
            "epochs": epochs
        }
    
    def _backpropagate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output: np.ndarray
    ):
        """反向传播更新权重"""
        m = X.shape[0]
        
        # 输出层梯度
        delta = 2 * (output - y) / m
        
        for i in range(len(self.weights) - 1, -1, -1):
            # 计算梯度和更新
            dW = self.activations[i].T @ delta
            dB = np.sum(delta, axis=0, keepdims=True)
            
            # 正则化（L2）
            dW += 0.001 * self.weights[i]
            
            # 更新权重
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * dB
            
            # 传递到下一层
            if i > 0:
                delta = (delta @ self.weights[i].T) * \
                        self._activate_derivative(self.z_values[i - 1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.forward(np.asarray(X, dtype=float))
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """二分类预测"""
        probs = self.predict(X)
        return (probs >= threshold).astype(int)
