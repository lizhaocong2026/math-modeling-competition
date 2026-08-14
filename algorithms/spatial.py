"""
空间统计与地理信息系统算法
用于空间数据分析、热点检测等
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.spatial import distance


class SpatialStatistics:
    """空间统计分析"""
    
    @staticmethod
    def moran_i(data: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
        """
        Moran's I 空间自相关指数
        
        参数:
            data: 属性值数组 (n,)
            weights: 空间权重矩阵 (n×n)
            
        返回:
            Moran's I值及显著性
        """
        n = len(data)
        data_mean = np.mean(data)
        data_centered = data - data_mean
        
        # 标准化
        s0 = np.sum(weights)
        s1 = 0.5 * np.sum(np.abs(weights - weights.T))
        
        # Moran's I
        numerator = n * np.sum(weights * np.outer(data_centered, data_centered))
        denominator = np.sum(data_centered ** 2) * s0
        
        if denominator == 0:
            return {"I": 0, "expected": -1/(n-1), "variance": 0}
        
        I = numerator / denominator
        
        # 期望和方差
        E_I = -1 / (n - 1)
        
        # 简化方差计算
        w_col_sum = np.sum(weights, axis=0)
        w_row_sum = np.sum(weights, axis=1)
        
        m2 = np.mean(data_centered ** 2)
        m4 = np.mean(data_centered ** 4)
        
        var_I = (m4 / m2**2 - 1) * (s1 / s0)**2 / n
        
        # Z值
        z = (I - E_I) / np.sqrt(var_I) if var_I > 0 else 0
        
        return {
            "moran_I": I,
            "expected": E_I,
            "variance": var_I,
            "z_score": z,
            "significant": abs(z) > 1.96
        }
    
    @staticmethod
    def getis_ord(data: np.ndarray, weights: np.ndarray, 
                  percentile: float = 90) -> Dict[str, Any]:
        """
        Getis-Ord G统计量
        用于检测热点和冷点
        """
        n = len(data)
        data_mean = np.mean(data)
        data_std = np.std(data)
        
        if data_std == 0:
            return {"G": 0}
        
        # 标准化数据
        data_std_norm = (data - data_mean) / data_std
        
        # 计算G统计量
        g_values = []
        for i in range(n):
            numerator = np.sum(weights[i] * data_std_norm)
            denominator = np.sum(weights[i])
            g_values.append(numerator / denominator if denominator > 0 else 0)
        
        g_values = np.array(g_values)
        
        # 阈值
        threshold = np.percentile(g_values, percentile)
        
        hotspots = np.where(g_values > threshold)[0]
        coldspots = np.where(g_values < -threshold)[0]
        
        return {
            "G_values": g_values.tolist(),
            "hotspots": hotspots.tolist(),
            "coldspots": coldspots.tolist(),
            "n_hotspots": len(hotspots),
            "n_coldspots": len(coldspots)
        }
    
    @staticmethod
    def kernel_density_estimate(points: np.ndarray, 
                                 grid_points: np.ndarray,
                                 bandwidth: float = 1.0) -> np.ndarray:
        """
        核密度估计
        """
        n_points = len(points)
        n_grid = len(grid_points)
        
        kde = np.zeros(n_grid)
        
        for i in range(n_grid):
            dists = distance.cdist(grid_points[i:i+1], points)[0]
            # 高斯核
            kde[i] = np.mean(np.exp(-0.5 * (dists / bandwidth) ** 2)) / (bandwidth * np.sqrt(2 * np.pi))
        
        return kde


class SpatialRegression:
    """空间回归模型"""
    
    @staticmethod
    def OLS(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """普通最小二乘回归"""
        n, p = X.shape
        
        # 添加截距
        X = np.column_stack([np.ones(n), X])
        
        # OLS估计
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        
        # 残差
        residuals = y - y_pred
        
        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # 调整R²
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # 标准误差
        sigma2 = ss_res / (n - p - 1)
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))
        
        # t统计量
        t_stats = beta / se_beta
        
        return {
            "coefficients": beta,
            "std_errors": se_beta,
            "t_statistics": t_stats,
            "R2": r2,
            "adj_R2": adj_r2,
            "sigma": np.sqrt(sigma2)
        }
    
    @staticmethod
    def spatial_lag(y: np.ndarray, X: np.ndarray, 
                    weights: np.ndarray) -> Dict[str, Any]:
        """
        空间滞后模型 (SAR)
        y = ρWy + Xβ + ε
        """
        n = len(y)
        
        # 标准化权重矩阵
        W = weights / np.sum(weights)
        
        # 简化的空间滞后估计
        # 实际应用中应使用MLE或2SLS
        
        # 初始OLS估计
        ols_result = SpatialRegression.OLS(X, y)
        residuals = y - X @ ols_result["coefficients"][1:]
        
        # 估计空间自相关系数ρ
        rho = np.sum(y * W @ y) / np.sum(y * y) if np.sum(y * y) > 0 else 0
        
        return {
            "rho": rho,
            "beta": ols_result["coefficients"][1:],
            "R2": ols_result["R2"],
            "method": "SAR-simplified"
        }


class NetworkAnalysis:
    """网络分析"""
    
    @staticmethod
    def betweenness_centrality(adjacency: np.ndarray) -> np.ndarray:
        """
        介数中心性
        """
        n = adjacency.shape[0]
        betweenness = np.zeros(n)
        
        for s in range(n):
            # BFS
            stack = []
            pred = [[] for _ in range(n)]
            sigma = np.zeros(n)
            sigma[s] = 1
            dist = -1 * np.ones(n, dtype=int)
            dist[s] = 0
            
            queue = [s]
            while queue:
                v = queue.pop(0)
                stack.append(v)
                for w in np.where(adjacency[v] > 0)[0]:
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            
            # 回溯
            delta = np.zeros(n)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]
        
        # 归一化
        if n > 2:
            betweenness /= ((n - 1) * (n - 2))
        
        return betweenness
    
    @staticmethod
    def eigenvector_centrality(adjacency: np.ndarray, 
                                max_iter: int = 100,
                                tol: float = 1e-6) -> np.ndarray:
        """特征向量中心性"""
        n = adjacency.shape[0]
        
        # 初始化
        centrality = np.ones(n) / n
        
        for _ in range(max_iter):
            new_centrality = adjacency @ centrality
            norm = np.linalg.norm(new_centrality)
            
            if norm == 0:
                break
            
            new_centrality = new_centrality / norm
            
            if np.linalg.norm(new_centrality - centrality) < tol:
                break
            
            centrality = new_centrality
        
        return centrality