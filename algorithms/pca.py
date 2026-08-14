"""
主成分分析 PCA
用于数据降维和特征提取
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple


class PCA:
    """主成分分析"""
    
    def __init__(self, n_components: Optional[int] = None, whiten: bool = False):
        """
        初始化PCA
        
        参数:
            n_components: 主成分数量，None表示保留所有成分
            whiten: 是否进行白化处理
        """
        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.noise_variance_ = None
        
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        拟合PCA模型
        
        参数:
            X: 数据矩阵 (n_samples, n_features)
            
        返回:
            拟合后的PCA实例
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        
        # 计算均值并中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 确定主成分数量
        n_components = self.n_components or min(eigenvalues.size, n_samples - 1)
        
        self.components_ = eigenvectors[:, :n_components]
        self.explained_variance_ = eigenvalues[:n_components]
        
        # 计算解释方差比例
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:n_components] / total_variance
        
        # 噪声方差
        if n_components < eigenvalues.size:
            self.noise_variance_ = np.mean(eigenvalues[n_components:])
        else:
            self.noise_variance_ = 0.0
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将数据投影到主成分空间
        
        参数:
            X: 输入数据
            
        返回:
            降维后的数据
        """
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        
        # 投影
        X_transformed = X_centered.dot(self.components_)
        
        # 白化处理
        if self.whiten:
            X_transformed = X_transformed / np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        从主成分空间逆变换回原始空间
        
        参数:
            X_transformed: 降维后的数据
            
        返回:
            近似还原的原始数据
        """
        X_centered = X_transformed.dot(self.components_.T)
        
        if self.whiten:
            X_centered = X_centered * np.sqrt(self.explained_variance_)
        
        return X_centered + self.mean_
    
    def transform_with_details(self, X: np.ndarray) -> Dict[str, Any]:
        """
        转换并返回详细信息
        
        返回:
            包含降维结果和详细信息的字典
        """
        X_transformed = self.transform(X)
        
        return {
            "transformed": X_transformed,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": np.cumsum(self.explained_variance_ratio_).tolist(),
            "n_components": self.components_.shape[1],
            "noise_variance": float(self.noise_variance_)
        }
