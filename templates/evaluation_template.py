"""
数学建模竞赛模板 - 评价类问题
适用于CUMCM C题（评价类题目）
"""
import numpy as np
from typing import Dict, Any, List
import sys
sys.path.insert(0, '..')

from algorithms.ahp import AHP
from algorithms.topcis import TOPSIS
from algorithms.entropy_weight import EntropyWeight
from algorithms.pca import PCA
from utils.data_preprocessor import DataPreprocessor
from visualizations.model_viz import ModelVisualization


class EvaluationProblemTemplate:
    """评价类问题模板"""
    
    def __init__(self):
        self.ahp = AHP()
        self.topsis = TOPSIS()
        self.entropy = EntropyWeight()
        self.pca = PCA()
        self.preprocessor = DataPreprocessor()
        self.viz = ModelVisualization()
        
    def ahp_evaluation(
        self, 
        comparison_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        层次分析法评价
        
        参数:
            comparison_matrix: 判断矩阵
            
        返回:
            包含权重和一致性检验结果的字典
        """
        result = self.ahp.compare(comparison_matrix)
        
        print("=" * 50)
        print("AHP层次分析法评价结果")
        print("=" * 50)
        print(f"权重向量: {result['weights']}")
        print(f"最大特征值: {result['maximum_eigenvalue']:.6f}")
        print(f"一致性比率 CR: {result['consistency_ratio']:.6f}")
        print(f"一致性检验: {result['consistency_level']}")
        
        return result
    
    def topsis_evaluation(
        self,
        data: np.ndarray,
        weights: Optional[np.ndarray] = None,
        types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        TOPSIS评价
        
        参数:
            data: 评价矩阵 (n个方案 × m个指标)
            weights: 各指标权重，None表示等权重
            types: 指标类型 'benefit'或'cost'
            
        返回:
            包含得分和排名的字典
        """
        if weights is not None:
            self.topsis = TOPSIS(weights=weights)
        
        result = self.topsis.evaluate(data, types=types)
        
        print("=" * 50)
        print("TOPSIS评价结果")
        print("=" * 50)
        
        # 打印得分和排名
        score_ranking = result['score_ranking']
        print(f"{'方案':<8} {'得分':<10} {'排名':<8}")
        print("-" * 30)
        for scheme_idx, score, rank in score_ranking:
            print(f"方案{scheme_idx:<6} {score:<10.6f} {rank:<8}")
        
        return result
    
    def entropy_weight_evaluation(
        self,
        data: np.ndarray
    ) -> Dict[str, Any]:
        """
        熵权法赋权
        
        参数:
            data: 评价矩阵
            
        返回:
            包含权重和熵值的字典
        """
        result = self.entropy.evaluate(data)
        
        print("=" * 50)
        print("熵权法赋权结果")
        print("=" * 50)
        print(f"各指标权重: {result['weights']}")
        print(f"各指标熵值: {result['entropies']}")
        print(f"差异系数: {result['differences']}")
        print(result['summary'])
        
        return result
    
    def pca_analysis(
        self,
        data: np.ndarray,
        n_components: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        PCA主成分分析
        
        参数:
            data: 数据矩阵
            n_components: 主成分数量
            
        返回:
            包含降维结果和解释方差比的字典
        """
        if n_components:
            self.pca = PCA(n_components=n_components)
        
        result = self.pca.transform_with_details(data)
        
        print("=" * 50)
        print("PCA主成分分析结果")
        print("=" * 50)
        print(f"主成分数量: {result['n_components']}")
        print(f"解释方差比例: {result['explained_variance_ratio']}")
        print(f"累积解释方差: {result['cumulative_explained_variance']}")
        print(f"噪声方差: {result['noise_variance']:.6f}")
        
        return result
    
    def comprehensive_evaluation(
        self,
        data: np.ndarray,
        method: str = "topsis",
        weights: Optional[np.ndarray] = None,
        types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        综合评价（支持多种方法）
        
        参数:
            data: 评价矩阵
            method: 评价方法 ('topsis', 'entropy', 'pca')
            weights: 权重（用于topsis）
            types: 指标类型
        """
        if method == "topsis":
            return self.topsis_evaluation(data, weights, types)
        elif method == "entropy":
            return self.entropy_weight_evaluation(data)
        elif method == "pca":
            return self.pca_analysis(data)
        else:
            raise ValueError(f"不支持的评价方法: {method}")
