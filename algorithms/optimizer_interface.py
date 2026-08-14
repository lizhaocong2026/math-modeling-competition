"""
优化算法的统一接口
便于快速切换不同优化方法
"""
import numpy as np
from typing import Callable, List, Tuple, Dict, Any, Optional
import sys
sys.path.insert(0, '..')

from algorithms.ga import GeneticAlgorithm
from algorithms.pso import ParticleSwarm
from algorithms.sa import SimulatedAnnealing
from algorithms.de import DifferentialEvolution


class OptimizerInterface:
    """优化算法统一接口"""
    
    def __init__(self, method: str = "ga"):
        """
        初始化优化器
        
        参数:
            method: 优化方法 ('ga', 'pso', 'sa', 'de')
        """
        self.method = method
        self.results = {}
        
        if method == "ga":
            self.optimizer = GeneticAlgorithm()
        elif method == "pso":
            self.optimizer = ParticleSwarm()
        elif method == "sa":
            self.optimizer = SimulatedAnnealing()
        elif method == "de":
            self.optimizer = DifferentialEvolution()
        else:
            raise ValueError(f"不支持的优化方法: {method}")
    
    def optimize(
        self,
        fitness_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        is_maximization: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行优化
        
        参数:
            fitness_func: 适应度函数
            bounds: 变量边界
            is_maximization: 是否最大化
            **kwargs: 其他参数
            
        返回:
            优化结果
        """
        if self.method == "ga":
            return self.optimizer.optimize(fitness_func, bounds, is_maximization, **kwargs)
        elif self.method == "pso":
            return self.optimizer.optimize(fitness_func, bounds, is_maximization, **kwargs)
        elif self.method == "sa":
            initial_point = np.array([(b[0] + b[1]) / 2 for b in bounds])
            return self.optimizer.optimize(fitness_func, initial_point, bounds, is_maximization=is_maximization, **kwargs)
        elif self.method == "de":
            return self.optimizer.optimize(fitness_func, bounds, is_maximization, **kwargs)
