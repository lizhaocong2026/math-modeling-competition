"""
支持向量机 SVM
用于分类和回归
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple


class SVM:
    """支持向量机（简化版，使用梯度下降）"""
    
    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01,
                 n_iters: int = 1000, kernel: str = "linear"):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.w = None
        self.b = None
        self.support_vectors = None
        self.alpha = None
        
    def _kernel(self, X1, X2):
        """核函数"""
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "rbf":
            sigma = 1.0
            sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                      np.sum(X2**2, axis=1).reshape(1, -1) - 2 * X1 @ X2.T
            return np.exp(-sq_dist / (2 * sigma**2))
        else:
            return X1 @ X2.T
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练SVM"""
        n_samples, n_features = X.shape
        
        # 初始化
        self.w = np.zeros(n_features)
        self.b = 0
        
        # 简化版：使用梯度下降
        for _ in range(self.n_iters):
            for i in range(n_samples):
                if y[i] * (np.dot(self.w, X[i]) + self.b) >= 1:
                    # 正确分类且在边界外
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # 分类错误或边界内
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w 
                                                    - y[i] * X[i])
                    self.b -= self.learning_rate * y[i]
        
        # 找支持向量
        margin = 1.0 / np.linalg.norm(self.w)
        self.support_vectors = X[np.abs(y * (X @ self.w + self.b) - 1) < 0.1]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        predictions = X @ self.w + self.b
        return np.sign(predictions)
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """拟合并预测"""
        self.fit(X, y)
        y_pred = self.predict(X)
        
        accuracy = np.mean(y_pred == y)
        
        return {
            "predictions": y_pred.astype(int).tolist(),
            "accuracy": float(accuracy),
            "n_support_vectors": len(self.support_vectors) if self.support_vectors is not None else 0
        }


class SVR:
    """支持向量回归"""
    
    def __init__(self, epsilon: float = 0.1, C: float = 1.0, 
                 learning_rate: float = 0.001, n_iters: int = 1000):
        self.epsilon = epsilon
        self.C = C
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练SVR"""
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            for i in range(n_samples):
                error = y[i] - (np.dot(self.w, X[i]) + self.b)
                
                if abs(error) <= self.epsilon:
                    # 在epsilon tube内
                    pass
                elif error > self.epsilon:
                    # 上方误差
                    self.w -= self.learning_rate * (-self.C * (error - self.epsilon) * X[i])
                    self.b -= self.learning_rate * (-self.C * (error - self.epsilon))
                else:
                    # 下方误差
                    self.w -= self.learning_rate * (self.C * (error + self.epsilon) * X[i])
                    self.b -= self.learning_rate * (self.C * (error + self.epsilon))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return X @ self.w + self.b