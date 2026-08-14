"""
熵权法 Entropy Weight Method
客观赋权算法，基于信息熵原理
"""
import numpy as np
from typing import Optional, Dict, Any, List


class EntropyWeight:
    """熵权法客观赋权"""
    
    def __init__(self):
        self.weights = None
        self.entropies = None
        self.differences = None
        
    def evaluate(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        计算熵权
        
        参数:
            matrix: 评价矩阵 (n个样本 × m个指标)
            
        返回:
            包含权重、熵值、差异系数的字典
        """
        matrix = np.asarray(matrix, dtype=float)
        n, m = matrix.shape
        
        # 数据标准化（极差标准化，正向化）
        normalized = self._normalize(matrix)
        
        # 计算每个指标下各样本所占比重 p_ij
        # 防止log(0)，添加微小量
        p = normalized / (np.sum(normalized, axis=0, keepdims=True) + 1e-10)
        
        # 计算信息熵 e_j
        # e_j = -k * sum(p_ij * ln(p_ij))
        k = 1.0 / np.log(n)
        entropy = -k * np.sum(
            np.where(p > 0, p * np.log(p), 0),
            axis=0
        )
        
        # 计算差异系数 g_j = 1 - e_j
        differences = 1 - entropy
        
        # 计算权重 w_j = g_j / sum(g_j)
        weights = differences / np.sum(differences)
        
        self.weights = weights
        self.entropies = entropy
        self.differences = differences
        
        return {
            "weights": weights.tolist(),
            "entropies": entropy.tolist(),
            "differences": differences.tolist(),
            "summary": f"共 {m} 个指标，差异系数总和: {np.sum(differences):.4f}"
        }
    
    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        """
        极差标准化处理
        
        参数:
            matrix: 原始数据矩阵
            
        返回:
            标准化后的矩阵
        """
        n, m = matrix.shape
        normalized = np.zeros_like(matrix)
        
        for j in range(m):
            col = matrix[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            
            if max_val == min_val:
                normalized[:, j] = 0.5  # 所有值相同，赋均值
            else:
                # 正向化处理 (0-1标准化)
                normalized[:, j] = (col - min_val) / (max_val - min_val)
        
        return normalized
    
    def combine_with_ahp(
        self, 
        entropy_weights: np.ndarray, 
        ahp_weights: np.ndarray, 
        entropy_ratio: float = 0.5
    ) -> np.ndarray:
        """
        组合赋权：熵权法 + AHP
        
        参数:
            entropy_weights: 熵权法权重
            ahp_weights: AHP权重
            entropy_ratio: 熵权法权重占比
            
        返回:
            组合权重
        """
        entropy_weights = np.asarray(entropy_weights)
        ahp_weights = np.asarray(ahp_weights)
        
        combined = entropy_ratio * entropy_weights + (1 - entropy_ratio) * ahp_weights
        return combined / np.sum(combined)
