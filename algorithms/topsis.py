"""
TOPSIS逼近理想解排序法
多属性决策评价算法
"""
import numpy as np
from typing import Optional, Dict, Any, List


class TOPSIS:
    """TOPSIS多属性决策评价算法"""
    
    def __init__(self, weights: Optional[np.ndarray] = None):
        """
        初始化TOPSIS
        
        参数:
            weights: 各指标权重向量，None表示等权重
        """
        self.weights = weights
        self.normalized_matrix = None
        self.weighted_normalized = None
        self.positive_ideal = None
        self.negative_ideal = None
        self.scores = None
        self.rankings = None
        
    def evaluate(
        self, 
        matrix: np.ndarray, 
        types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        进行TOPSIS评价
        
        参数:
            matrix: 评价矩阵 (n个方案 × m个指标)
            types: 指标类型，'benefit'为效益型（越大越好），
                   'cost'为成本型（越小越好），None表示全为效益型
            
        返回:
            包含评价结果的字典
        """
        matrix = np.asarray(matrix, dtype=float)
        n, m = matrix.shape
        
        # 处理权重
        if self.weights is None:
            weights = np.ones(m) / m
        else:
            weights = np.asarray(self.weights)
            if len(weights) != m:
                raise ValueError(f"权重长度 {len(weights)} 与指标数 {m} 不匹配")
            weights = weights / np.sum(weights)
        
        # 处理指标类型
        if types is None:
            types = ['benefit'] * m
        elif len(types) != m:
            raise ValueError(f"types长度 {len(types)} 与指标数 {m} 不匹配")
        
        # 标准化矩阵
        norms = np.sqrt(np.sum(matrix ** 2, axis=0))
        normalized = matrix / norms
        
        # 加权标准化矩阵
        weighted_normalized = normalized * weights
        self.weighted_normalized = weighted_normalized
        
        # 确定正负理想解
        positive_ideal = np.max(weighted_normalized, axis=0)
        negative_ideal = np.min(weighted_normalized, axis=0)
        
        # 对于成本型指标，正负理想解互换
        for i, t in enumerate(types):
            if t == 'cost':
                positive_ideal[i], negative_ideal[i] = negative_ideal[i], positive_ideal[i]
        
        self.positive_ideal = positive_ideal
        self.negative_ideal = negative_ideal
        
        # 计算距离
        dist_positive = np.sqrt(np.sum((weighted_normalized - positive_ideal) ** 2, axis=1))
        dist_negative = np.sqrt(np.sum((weighted_normalized - negative_ideal) ** 2, axis=1))
        
        # 计算贴近度
        self.scores = dist_negative / (dist_positive + dist_negative + 1e-10)
        
        # 排序
        self.rankings = np.argsort(-self.scores) + 1  # 排名（越小越好）
        
        return self._format_result(matrix, weights)
    
    def _format_result(
        self, 
        original_matrix: np.ndarray,
        weights: np.ndarray
    ) -> Dict[str, Any]:
        """格式化结果"""
        n = len(self.scores)
        
        return {
            "scores": self.scores.tolist(),
            "rankings": self.rankings.tolist(),
            "weights": weights.tolist(),
            "positive_ideal": self.positive_ideal.tolist(),
            "negative_ideal": self.negative_ideal.tolist(),
            "score_ranking": sorted(
                zip(range(1, n + 1), self.scores.tolist(), self.rankings.tolist()),
                key=lambda x: x[2]
            )
        }
    
    def get_recommendation(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """获取前K名推荐"""
        if self.scores is None:
            raise RuntimeError("请先调用evaluate()方法")
        
        result = []
        for i in range(min(top_k, len(self.scores))):
            rank = np.where(self.rankings == i + 1)[0][0] + 1
            result.append({
                "rank": i + 1,
                "scheme_index": rank,
                "score": float(self.scores[rank - 1])
            })
        
        return result
