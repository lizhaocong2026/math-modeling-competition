"""
蒙特卡洛方法进阶
涵盖方差缩减技术和重要性采样
"""
import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any, List
from scipy import stats


class VarianceReduction:
    """方差缩减技术"""
    
    @staticmethod
    def antithetic_variates(func: Callable, n: int = 10000) -> Dict[str, Any]:
        """
        对偶变量法
        利用负相关降低方差
        """
        u = np.random.uniform(0, 1, n)
        
        # 生成对偶样本
        samples1 = func(u)
        samples2 = func(1 - u)
        
        # 平均
        estimators = (samples1 + samples2) / 2
        
        return {
            "estimate": np.mean(estimators),
            "std_error": np.std(estimators) / np.sqrt(n),
            "variance": np.var(estimators),
            "n_samples": n
        }
    
    @staticmethod
    def control_variates(
        func: Callable,
        control_func: Callable,
        control_mean: float,
        n: int = 10000
    ) -> Dict[str, Any]:
        """
        控制变量法
        利用已知期望的辅助变量降低方差
        """
        u = np.random.uniform(0, 1, n)
        
        y = func(u)
        z = control_func(u)
        
        # 计算最优控制系数
        cov_yz = np.cov(y, z)[0, 1]
        var_z = np.var(z)
        
        if var_z == 0:
            beta = 0
        else:
            beta = cov_yz / var_z
        
        # 修正估计
        y_corrected = y - beta * (z - control_mean)
        
        return {
            "estimate": np.mean(y_corrected),
            "std_error": np.std(y_corrected) / np.sqrt(n),
            "variance": np.var(y_corrected),
            "beta": beta,
            "variance_reduction": 1 - np.var(y_corrected) / np.var(y)
        }
    
    @staticmethod
    def stratified_sampling(
        func: Callable,
        n_strata: int = 10,
        n_per_stratum: int = 100
    ) -> Dict[str, Any]:
        """
        分层抽样
        将样本空间分成若干层分别抽样
        """
        total_n = n_strata * n_per_stratum
        estimates = []
        
        for i in range(n_strata):
            # 在第i层采样
            a = i / n_strata
            b = (i + 1) / n_strata
            
            u = np.random.uniform(a, b, n_per_stratum)
            # 缩放到[0,1]
            u_scaled = (u - a) / (b - a)
            
            samples = func(u_scaled)
            estimates.append(np.mean(samples))
        
        # 分层加权平均
        estimate = np.mean(estimates)
        variance = np.sum([np.var(func(np.random.uniform(i/n_strata, (i+1)/n_strata, n_per_stratum))) 
                          for i in range(n_strata)]) / (n_strata**2 * n_per_stratum)
        
        return {
            "estimate": estimate,
            "std_error": np.sqrt(variance / total_n),
            "n_strata": n_strata,
            "n_per_stratum": n_per_stratum
        }


class ImportanceSampling:
    """重要性采样"""
    
    def __init__(self, proposal_dist: str = "normal", **params):
        """
        参数:
            proposal_dist: 建议分布类型
            **params: 分布参数
        """
        self.proposal_dist = proposal_dist
        self.params = params
        
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """采样并计算权重"""
        if self.proposal_dist == "normal":
            mean = self.params.get("mean", 0)
            std = self.params.get("std", 1)
            samples = np.random.normal(mean, std, n)
            proposal_pdf = stats.norm.pdf(samples, mean, std)
        elif self.proposal_dist == "uniform":
            low = self.params.get("low", 0)
            high = self.params.get("high", 1)
            samples = np.random.uniform(low, high, n)
            proposal_pdf = 1 / (high - low) * np.ones(n)
        else:
            raise ValueError(f"不支持的分布: {self.proposal_dist}")
        
        return samples, proposal_pdf
    
    def estimate(
        self,
        target_func: Callable,
        target_pdf: Callable,
        n: int = 10000
    ) -> Dict[str, Any]:
        """
        估计积分
        
        参数:
            target_func: 被积函数
            target_pdf: 目标概率密度
            n: 采样数
        """
        samples, proposal_pdf = self.sample(n)
        
        # 重要性权重
        weights = target_pdf(samples) / proposal_pdf
        
        # 估计
        weighted_samples = target_func(samples) * weights
        estimate = np.mean(weighted_samples)
        
        # 方差估计
        variance = np.var(weighted_samples) / n
        
        return {
            "estimate": estimate,
            "std_error": np.sqrt(variance),
            "n_samples": n
        }


class LatinHypercubeSampling:
    """拉丁超立方抽样（改进版）"""
    
    def __init__(self, n_params: int):
        self.n_params = n_params
        
    def sample(self, n: int, bounds: List[Tuple[float, float]] = None) -> np.ndarray:
        """
        生成LHS样本
        
        参数:
            n: 样本数
            bounds: 各参数的边界，默认[0,1]
            
        返回:
            样本矩阵 (n × n_params)
        """
        if bounds is None:
            bounds = [(0, 1)] * self.n_params
        
        n_params = len(bounds)
        samples = np.zeros((n, n_params))
        
        for j in range(n_params):
            low, high = bounds[j]
            # 分层
            intervals = np.linspace(low, high, n + 1)
            points = np.random.uniform(intervals[:-1], intervals[1:], n)
            # 随机排列
            np.random.shuffle(points)
            samples[:, j] = points
        
        return samples
    
    def sample_normal(self, n: int, means: np.ndarray = None, stds: np.ndarray = None) -> np.ndarray:
        """生成正态分布的LHS样本"""
        if means is None:
            means = np.zeros(self.n_params)
        if stds is None:
            stds = np.ones(self.n_params)
        
        # 生成均匀LHS
        uniform_samples = self.sample(n)
        
        # 转换为正态
        normal_samples = stats.norm.ppf(uniform_samples) * stds + means
        
        return normal_samples


class SequentialMonteCarlo:
    """序贯蒙特卡洛（粒子滤波简化版）"""
    
    def __init__(self, n_particles: int = 1000):
        self.n_particles = n_particles
        self.particles = []
        self.weights = []
        
    def initialize(self, prior_sample: Callable):
        """初始化粒子"""
        self.particles = np.array([prior_sample() for _ in range(self.n_particles)])
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def resample(self):
        """重采样"""
        cum_weights = np.cumsum(self.weights)
        indices = np.searchsorted(cum_weights, np.random.random(self.n_particles))
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def update(self, observations: np.ndarray, 
               likelihood_func: Callable) -> np.ndarray:
        """
        更新粒子权重
        
        参数:
            observations: 观测序列
            likelihood_func: 似然函数
            
        返回:
            状态估计
        """
        for t, obs in enumerate(observations):
            # 计算新权重
            new_weights = np.zeros(self.n_particles)
            for i in range(self.n_particles):
                new_weights[i] = self.weights[i] * likelihood_func(obs, self.particles[i])
            
            # 归一化
            total = np.sum(new_weights)
            if total > 0:
                self.weights = new_weights / total
            else:
                self.weights = np.ones(self.n_particles) / self.n_particles
            
            # 重采样
            if np.var(self.weights) > 0.5:  # 有效样本数过少时重采样
                self.resample()
        
        return np.mean(self.particles, axis=0)