"""
数学建模竞赛模板 - 分类与聚类问题
适用于图像识别、模式分类等题目
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys
sys.path.insert(0, '..')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.data_preprocessor import DataPreprocessor
from visualizations.model_viz import ModelVisualization


class ClassificationClusteringTemplate:
    """分类与聚类问题模板"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.viz = ModelVisualization()
        
    def kmeans_clustering(
        self,
        data: np.ndarray,
        n_clusters: int = 3,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        KMeans聚类
        
        参数:
            data: 数据矩阵
            n_clusters: 聚类数
            random_state: 随机种子
            
        返回:
            包含聚类结果和评估指标的字典
        """
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        
        # 计算轮廓系数
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(data_scaled, labels)
        
        # 计算惯性（簇内平方和）
        inertia = kmeans.inertia_
        
        result = {
            'labels': labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': float(inertia),
            'silhouette_score': float(silhouette),
            'n_clusters': n_clusters
        }
        
        print("=" * 50)
        print(f"KMeans聚类结果 (k={n_clusters})")
        print("=" * 50)
        print(f"轮廓系数: {silhouette:.4f}")
        print(f"惯性: {inertia:.4f}")
        
        return result
    
    def determine_optimal_clusters(
        self,
        data: np.ndarray,
        k_range: range = range(2, 11)
    ) -> Dict[str, Any]:
        """
        确定最优聚类数（肘部法则 + 轮廓系数）
        
        参数:
            data: 数据矩阵
            k_range: 聚类数范围
            
        返回:
            包含最优聚类数和评估结果的字典
        """
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, labels))
        
        # 肘部法则：寻找惯性下降放缓的点
        # 计算二阶差分
        inertia_diff = np.diff(inertias)
        inertia_diff2 = np.diff(inertia_diff)
        elbow_idx = np.argmax(inertia_diff2) + 2  # +2因为二阶差分
        
        optimal_k = max(min(elbow_idx, len(k_range)), 2)
        
        result = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_elbow': optimal_k,
            'optimal_k_silhouette': list(k_range)[np.argmax(silhouette_scores)]
        }
        
        print("=" * 50)
        print("最优聚类数确定")
        print("=" * 50)
        print(f"肘部法则推荐: k = {optimal_k}")
        print(f"轮廓系数推荐: k = {result['optimal_k_silhouette']}")
        
        return result
    
    def hierarchical_clustering(
        self,
        data: np.ndarray,
        n_clusters: int = 3,
        linkage: str = 'ward'
    ) -> Dict[str, Any]:
        """
        层次聚类
        
        参数:
            data: 数据矩阵
            n_clusters: 聚类数
            linkage: 连接方式 ('ward', 'complete', 'average', 'single')
            
        返回:
            包含聚类结果的字典
        """
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = clustering.fit_predict(data)
        
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(data, labels)
        
        result = {
            'labels': labels.tolist(),
            'silhouette_score': float(silhouette),
            'linkage': linkage,
            'n_clusters': n_clusters
        }
        
        print("=" * 50)
        print(f"层次聚类结果 (linkage={linkage}, k={n_clusters})")
        print("=" * 50)
        print(f"轮廓系数: {silhouette:.4f}")
        
        return result
    
    def dbscan_clustering(
        self,
        data: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Dict[str, Any]:
        """
        DBSCAN密度聚类
        
        参数:
            data: 数据矩阵
            eps: 邻域半径
            min_samples: 最小样本数
            
        返回:
            包含聚类结果的字典
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        result = {
            'labels': labels.tolist(),
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples
        }
        
        print("=" * 50)
        print(f"DBSCAN聚类结果")
        print("=" * 50)
        print(f"聚类数: {n_clusters}")
        print(f"噪声点: {n_noise}")
        
        return result
