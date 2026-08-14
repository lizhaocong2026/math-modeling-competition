"""
蒙特卡洛模拟模块
用于概率计算、积分、风险评估等
"""
import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any
import math


class MonteCarlo:
    """蒙特卡洛模拟器"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化
        
        参数:
            seed: 随机种子，用于结果可复现
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        
    def estimate_integral(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        bounds: Tuple[float, float, float, float],
        n_samples: int = 100000
    ) -> Dict[str, Any]:
        """
        蒙特卡洛数值积分
        
        参数:
            func: 被积函数 f(x, y)
            bounds: 积分区域 (x_min, x_max, y_min, y_max)
            n_samples: 采样点数
            
        返回:
            包含积分近似值和误差估计的字典
        """
        x_min, x_max, y_min, y_max = bounds
        area = (x_max - x_min) * (y_max - y_min)
        
        # 随机采样
        x = np.random.uniform(x_min, x_max, n_samples)
        y = np.random.uniform(y_min, y_max, n_samples)
        
        # 计算函数值
        z = func(x, y)
        
        # 积分估计 = 面积 × 均值
        integral = area * np.mean(z)
        
        # 误差估计 (标准误差)
        std_dev = np.std(z)
        error = area * std_dev / np.sqrt(n_samples)
        
        return {
            "integral": float(integral),
            "error_estimate": float(error),
            "relative_error": float(error / abs(integral)) if integral != 0 else float('inf'),
            "n_samples": n_samples
        }
    
    def estimate_pi(self, n_samples: int = 100000) -> Dict[str, Any]:
        """
        蒙特卡洛估算圆周率π
        
        参数:
            n_samples: 采样点数
            
        返回:
            包含π的估算值和误差的字典
        """
        # 在[0,1]×[0,1]正方形内随机撒点
        x = np.random.uniform(0, 1, n_samples)
        y = np.random.uniform(0, 1, n_samples)
        
        # 计算在四分之一圆内的点数
        in_circle = (x**2 + y**2) <= 1
        n_in_circle = np.sum(in_circle)
        
        # π ≈ 4 × (圆内点数 / 总点数)
        pi_estimate = 4.0 * n_in_circle / n_samples
        
        # 误差估计
        error = 4.0 * np.sqrt(pi_estimate * (4 - pi_estimate) / n_samples)
        
        return {
            "pi_estimate": float(pi_estimate),
            "error": float(error),
            "n_samples": n_samples,
            "accuracy": f"{abs(pi_estimate - math.pi):.6f}"
        }
    
    def risk_analysis(
        self,
        revenue_func: Callable[[np.ndarray], np.ndarray],
        distributions: Dict[str, Dict[str, float]],
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        蒙特卡洛风险模拟
        
        参数:
            revenue_func: 收益函数，输入参数数组，输出收益
            distributions: 各参数的分布 {
                'param_name': {'type': 'normal', 'mean': mu, 'std': sigma}
            }
            n_simulations: 模拟次数
            
        返回:
            包含风险指标的字典
        """
        # 生成随机样本
        samples = {}
        for param_name, dist in distributions.items():
            dist_type = dist['type']
            if dist_type == 'normal':
                samples[param_name] = np.random.normal(
                    dist['mean'], dist['std'], n_simulations
                )
            elif dist_type == 'uniform':
                samples[param_name] = np.random.uniform(
                    dist['low'], dist['high'], n_simulations
                )
            elif dist_type == 'triangular':
                samples[param_name] = np.random.triangular(
                    dist['low'], dist['mode'], dist['high'], n_simulations
                )
        
        # 构建输入矩阵
        input_matrix = np.column_stack([samples[p] for p in distributions.keys()])
        
        # 计算收益
        revenues = revenue_func(input_matrix)
        
        # 统计指标
        result = {
            "expected_value": float(np.mean(revenues)),
            "std_deviation": float(np.std(revenues)),
            "min_value": float(np.min(revenues)),
            "max_value": float(np.max(revenues)),
            "var_95": float(np.percentile(revenues, 5)),
            "var_99": float(np.percentile(revenues, 1)),
            "prob_profit": float(np.mean(revenues > 0)),
            "n_simulations": n_simulations
        }
        
        return result
    
    def optimize_random_search(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        n_iterations: int = 10000
    ) -> Dict[str, Any]:
        """
        随机搜索优化（蒙特卡洛方法）
        
        参数:
            func: 目标函数
            bounds: 变量边界
            n_iterations: 迭代次数
            
        返回:
            最优解和最优值
        """
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        best_x = None
        best_f = float('inf')
        
        for _ in range(n_iterations):
            x = np.random.uniform(lower, upper)
            f = func(x)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        return {
            "optimal_solution": best_x.tolist() if best_x is not None else None,
            "optimal_value": float(best_f),
            "n_iterations": n_iterations
        }
