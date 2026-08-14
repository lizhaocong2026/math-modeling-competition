"""
线性回归预测器
最小二乘法线性回归
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any


class LinearRegression:
    """线性回归预测器"""
    
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        self.r_squared = None
        self.std_error = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        拟合线性回归模型
        
        参数:
            X: 特征矩阵 (n×m)
            y: 目标向量 (n,)
            
        返回:
            拟合后的模型实例
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        n, m = X.shape
        
        if self.fit_intercept:
            # 添加截距项
            X_aug = np.column_stack([np.ones(n), X])
        else:
            X_aug = X
        
        # 最小二乘解: β = (X^T X)^(-1) X^T y
        try:
            self.coefficients = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # 矩阵奇异时使用伪逆
            self.coefficients = np.linalg.pinv(X_aug).dot(y)
        
        if self.fit_intercept:
            self.intercept = self.coefficients[0]
            self.coefficients = self.coefficients[1:]
        else:
            self.intercept = 0.0
        
        # 计算 R²
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # 计算标准误差
        if n > m + 1:
            self.std_error = np.sqrt(ss_res / (n - m - 1))
        else:
            self.std_error = np.sqrt(ss_res / max(n - 1, 1))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        X = np.asarray(X, dtype=float)
        return X.dot(self.coefficients) + self.intercept
    
    def fit_predict(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """拟合并返回预测结果"""
        self.fit(X, y)
        y_pred = self.predict(X)
        
        return {
            "coefficients": self.coefficients.tolist(),
            "intercept": float(self.intercept),
            "r_squared": float(self.r_squared),
            "predictions": y_pred.tolist(),
            "std_error": float(self.std_error) if self.std_error else None
        }
    
    def forecast(self, X_new: np.ndarray) -> Dict[str, Any]:
        """预测新数据并给出置信区间"""
        X_new = np.asarray(X_new, dtype=float)
        y_pred = self.predict(X_new)
        
        # 简单预测区间（95%）
        margin = 1.96 * (self.std_error or 0) if self.std_error else 0
        lower = y_pred - margin
        upper = y_pred + margin
        
        return {
            "predictions": y_pred.tolist(),
            "confidence_interval": {
                "lower": lower.tolist(),
                "upper": upper.tolist()
            }
        }
