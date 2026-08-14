"""
马尔可夫链蒙特卡洛 MCMC
用于贝叶斯推断和复杂分布采样
"""
import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any, List


class MetropolisHastings:
    """Metropolis-Hastings采样器"""
    
    def __init__(self, proposal_std: float = 1.0):
        self.proposal_std = proposal_std
        self.samples = []
        self.acceptance_rate = 0.0
        
    def sample(self, log_target: Callable[[np.ndarray], float],
               initial: np.ndarray, n_samples: int = 10000,
               burn_in: int = 1000) -> np.ndarray:
        """
        使用MH算法采样
        
        参数:
            log_target: 目标分布的对数概率
            initial: 初始点
            n_samples: 采样数量
            burn_in: burn-in期
            
        返回:
            采样样本
        """
        current = initial.copy()
        current_log = log_target(current)
        
        all_samples = []
        accepted = 0
        
        for i in range(n_samples + burn_in):
            # 生成 Proposal
            proposal = current + np.random.randn(len(current)) * self.proposal_std
            proposal_log = log_target(proposal)
            
            # 计算接受概率
            log_alpha = proposal_log - current_log
            u = np.log(np.random.random())
            
            if u < log_alpha:
                current = proposal
                current_log = proposal_log
                accepted += 1
            
            if i >= burn_in:
                all_samples.append(current.copy())
        
        self.acceptance_rate = accepted / (n_samples + burn_in)
        self.samples = np.array(all_samples)
        
        return self.samples
    
    def get_summary(self) -> Dict[str, Any]:
        """获取采样摘要"""
        if len(self.samples) == 0:
            return {}
        
        return {
            "mean": np.mean(self.samples, axis=0).tolist(),
            "std": np.std(self.samples, axis=0).tolist(),
            "acceptance_rate": self.acceptance_rate,
            "n_samples": len(self.samples)
        }


class GibbsSampler:
    """Gibbs采样器"""
    
    def __init__(self):
        self.samples = []
        
    def sample(self, conditional_funcs: List[Callable], 
               initial: np.ndarray, n_samples: int = 10000,
               burn_in: int = 1000) -> np.ndarray:
        """
        Gibbs采样
        
        参数:
            conditional_funcs: 条件分布采样函数列表
            initial: 初始点
            n_samples: 采样数
            burn_in: burn-in
            
        返回:
            采样样本
        """
        current = initial.copy()
        n_params = len(initial)
        all_samples = []
        
        for i in range(n_samples + burn_in):
            for j in range(n_params):
                current[j] = conditional_funcs[j](current, j)
            
            if i >= burn_in:
                all_samples.append(current.copy())
        
        self.samples = np.array(all_samples)
        return self.samples


class HamiltonianMC:
    """哈密顿蒙特卡洛采样器"""
    
    def __init__(self, step_size: float = 0.1, n_steps: int = 20):
        self.step_size = step_size
        self.n_steps = n_steps
        
    def sample(self, log_target: Callable[[np.ndarray], float],
               initial: np.ndarray, n_samples: int = 10000) -> np.ndarray:
        """
        HMC采样（简化版）
        
        参数:
            log_target: 目标对数概率
            initial: 初始点
            n_samples: 采样数
            
        返回:
            采样样本
        """
        samples = [initial.copy()]
        
        for _ in range(n_samples - 1):
            current = samples[-1]
            
            # 生成动量
            momentum = np.random.randn(len(current))
            
            # Leapfrog积分
            position = current.copy()
            velocity = momentum.copy()
            
            for _ in range(self.n_steps):
                # 梯度计算（数值近似）
                grad = self._numerical_gradient(log_target, position)
                velocity += self.step_size * grad
                position += self.step_size * velocity
            
            # Metropolis接受步骤
            current_log = log_target(current)
            proposed_log = log_target(position)
            
            # 哈密顿量
            H_current = -current_log + 0.5 * np.sum(momentum**2)
            H_proposed = -proposed_log + 0.5 * np.sum(velocity**2)
            
            # 接受/拒绝
            if np.random.random() < np.exp(-(H_proposed - H_current)):
                samples.append(position)
            else:
                samples.append(current)
        
        return np.array(samples)
    
    def _numerical_gradient(self, func: Callable, x: np.ndarray, 
                            h: float = 1e-5) -> np.ndarray:
        """数值梯度计算"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad