"""
扩展算法：最小二乘法拟合曲线
"""
import numpy as np
from typing import List, Tuple, Dict, Any


class CurveFitting:
    """曲线拟合工具类"""
    
    @staticmethod
    def linear_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """线性拟合 y = ax + b"""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return {"a": 0, "b": np.mean(y), "r_squared": 0}
        
        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / n
        
        # R²计算
        y_pred = a * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            "slope": float(a),
            "intercept": float(b),
            "r_squared": float(r_squared),
            "equation": f"y = {a:.4f}x + {b:.4f}"
        }
    
    @staticmethod
    def exponential_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """指数拟合 y = a * exp(bx)"""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # 取对数转换为线性问题
        log_y = np.log(y)
        
        result = CurveFitting.linear_fit(x, log_y)
        
        a = np.exp(result["intercept"])
        b = result["slope"]
        
        # 重新计算R²
        y_pred = a * np.exp(b * x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            "a": float(a),
            "b": float(b),
            "r_squared": float(r_squared),
            "equation": f"y = {a:.4f} * exp({b:.4f} * x)"
        }
    
    @staticmethod
    def power_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """幂函数拟合 y = a * x^b"""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # 取对数
        log_x = np.log(x)
        log_y = np.log(y)
        
        result = CurveFitting.linear_fit(log_x, log_y)
        
        a = np.exp(result["intercept"])
        b = result["slope"]
        
        return {
            "a": float(a),
            "b": float(b),
            "r_squared": float(result["r_squared"]),
            "equation": f"y = {a:.4f} * x^{b:.4f}"
        }
