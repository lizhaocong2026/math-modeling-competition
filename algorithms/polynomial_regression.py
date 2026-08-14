"""
多项式回归预测器
支持任意阶多项式拟合
"""
import numpy as np
from typing import Optional, Dict, Any


class PolynomialRegression:
    """多项式回归预测器"""
    
    def __init__(self, degree: int = 2):
        """
        初始化
        
        参数:
            degree: 多项式阶数
        """
        self.degree = degree
        self.coefficients = None
        self.r_squared = None
        self.preprocessing_params = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegression':
        """
        拟合多项式回归模型
        
        参数:
            X: 特征 (n, 1)
            y: 目标 (n,)
            
        返回:
            拟合后的模型实例
        """
        X = np.asarray(X, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float)
        
        # 保存预处理参数
        self.preprocessing_params = {
            'x_min': float(np.min(X)),
            'x_max': float(np.max(X))
        }
        
        # 构建多项式特征矩阵
        X_poly = self._build_polynomial_matrix(X, self.degree)
        
        # 最小二乘拟合
        try:
            self.coefficients = np.linalg.lstsq(X_poly, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.pinv(X_poly).dot(y)
        
        # 计算 R²
        y_pred = X_poly.dot(self.coefficients)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return self
    
    def _build_polynomial_matrix(self, X: np.ndarray, degree: int) -> np.ndarray:
        """构建多项式特征矩阵"""
        n = X.shape[0]
        X_poly = np.ones((n, degree + 1))
        
        for d in range(1, degree + 1):
            X_poly[:, d] = X[:, 0] ** d
        
        return X_poly
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        X = np.asarray(X, dtype=float).reshape(-1, 1)
        X_poly = self._build_polynomial_matrix(X, self.degree)
        return X_poly.dot(self.coefficients)
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """拟合并返回结果"""
        self.fit(X, y)
        
        return {
            "coefficients": self.coefficients.tolist(),
            "degree": self.degree,
            "r_squared": float(self.r_squared),
            "preprocessing": self.preprocessing_params
        }
    
    def predict_series(self, X: np.ndarray, steps: int = 1) -> Dict[str, Any]:
        """预测序列"""
        predictions = self.predict(X)
        
        return {
            "predictions": predictions.tolist(),
            "r_squared": float(self.r_squared)
        }
