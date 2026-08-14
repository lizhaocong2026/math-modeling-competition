"""
层次分析法 AHP
用于多准则决策分析
"""
import numpy as np
from typing import Optional, Dict, Any, List
import warnings


class AHP:
    """层次分析法 (Analytic Hierarchy Process)"""
    
    def __init__(self):
        self.consistency_ratio = None
        self.weights = None
        self.priority_vector = None
        self.maximum_eigenvalue = None
        self.rank = None
        
    def compare(
        self, 
        comparison_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        进行AHP分析
        
        参数:
            comparison_matrix: 判断矩阵 (n×n)
            
        返回:
            包含权重、一致性检验结果等的字典
        """
        n = comparison_matrix.shape[0]
        
        if comparison_matrix.shape != (n, n):
            raise ValueError("判断矩阵必须是方阵")
        
        # 使用几何平均法计算权重
        weights = self._geometric_mean_weight(comparison_matrix)
        self.weights = weights
        self.priority_vector = weights
        
        # 计算最大特征值
        lambda_max = self._calculate_lambda_max(comparison_matrix, weights)
        self.maximum_eigenvalue = lambda_max
        
        # 一致性检验
        consistency_result = self._consistency_check(n, lambda_max)
        self.consistency_ratio = consistency_result['CR']
        self.rank = consistency_result['rank']
        
        return {
            "weights": weights.tolist(),
            "maximum_eigenvalue": float(lambda_max),
            "consistency_ratio": float(self.consistency_ratio),
            "consistency_level": self.rank,
            "is_consistent": self.consistency_ratio < 0.1
        }
    
    def _geometric_mean_weight(self, matrix: np.ndarray) -> np.ndarray:
        """几何平均法求权重"""
        # 计算每行的几何平均
        geo_mean = np.exp(np.mean(np.log(np.abs(matrix) + 1e-10), axis=1))
        # 归一化
        weights = geo_mean / np.sum(geo_mean)
        return weights
    
    def _calculate_lambda_max(self, matrix: np.ndarray, weights: np.ndarray) -> float:
        """计算最大特征值"""
        Aw = matrix.dot(weights)
        lambda_max = np.sum(Aw / weights) / len(weights)
        return lambda_max
    
    def _consistency_check(
        self, 
        n: int, 
        lambda_max: float
    ) -> Dict[str, Any]:
        """
        一致性检验
        
        返回:
            一致性指标CI、RI、CR及等级
        """
        # 一致性指标 CI = (λmax - n) / (n - 1)
        CI = (lambda_max - n) / (n - 1)
        
        # 随机一致性指标 RI (查表)
        RI_TABLE = {
            1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
            5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
            9: 1.45, 10: 1.49
        }
        
        RI = RI_TABLE.get(n, 1.49)
        
        # 一致性比率 CR = CI / RI
        CR = CI / RI if RI > 0 else 0
        
        # 判断等级
        if CR < 0.01:
            rank = "完全一致"
        elif CR < 0.1:
            rank = "通过一致性检验"
        else:
            rank = "未通过一致性检验，需重新调整判断矩阵"
        
        return {
            "CI": float(CI),
            "RI": float(RI),
            "CR": float(CR),
            "rank": rank
        }
    
    def create_comparison_matrix(self, n: int) -> np.ndarray:
        """
        创建判断矩阵（用户交互）
        
        参数:
            n: 因素数量
            
        返回:
            待填写的判断矩阵
        """
        print(f"\n请构建 {n}×{n} 的判断矩阵")
        print("标度说明：1=同等重要, 3=稍微重要, 5=明显重要, 7=强烈重要, 9=极端重要")
        print("2,4,6,8为中间值，倒数表示反方向")
        print()
        
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    val = float(input(f"因素 {i+1} 相对于因素 {j+1} 的重要性 (1-9): "))
                    matrix[i, j] = val
                    matrix[j, i] = 1.0 / val
                except:
                    print("输入无效，使用默认值 1")
        
        return matrix
