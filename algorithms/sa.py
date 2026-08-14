"""
模拟退火算法 (Simulated Annealing)
基于金属退火过程的随机优化算法
"""
import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any
import math


class SimulatedAnnealing:
    """模拟退火优化算法"""
    
    def __init__(
        self,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.995,
        min_temp: float = 1e-10,
        steps_per_temp: int = 100,
        verbose: bool = False
    ):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.steps_per_temp = steps_per_temp
        self.verbose = verbose
        
        self.best_solution = None
        self.best_fitness = None
        self.temperature_history = []
        self.fitness_history = []
        
    def optimize(
        self,
        fitness_func: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        bounds: List[Tuple[float, float]],
        perturbation_scale: float = 0.1,
        is_maximization: bool = True
    ) -> Dict[str, Any]:
        """
        运行模拟退火优化
        
        参数:
            fitness_func: 适应度函数
            initial_point: 初始点
            bounds: 变量边界
            perturbation_scale: 扰动幅度
            is_maximization: 是否最大化
            
        返回:
            包含最优解、最优值的字典
        """
        current_point = np.array(initial_point, dtype=float)
        current_fitness = fitness_func(current_point)
        
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        best_solution = current_point.copy()
        best_fitness = current_fitness
        temp = self.initial_temp
        
        self.fitness_history = []
        self.temperature_history = []
        
        while temp > self.min_temp:
            for _ in range(self.steps_per_temp):
                # 生成新解
                new_point = current_point + np.random.randn(len(current_point)) * perturbation_scale * temp
                new_point = np.clip(new_point, lower, upper)
                
                new_fitness = fitness_func(new_point)
                
                # Metropolis准则
                if is_maximization:
                    delta = new_fitness - current_fitness
                else:
                    delta = current_fitness - new_fitness
                
                if delta > 0 or np.random.random() < math.exp(delta / temp):
                    current_point = new_point
                    current_fitness = new_fitness
            
            # 更新全局最优
            if (is_maximization and current_fitness > best_fitness) or \
               (not is_maximization and current_fitness < best_fitness):
                best_solution = current_point.copy()
                best_fitness = current_fitness
            
            self.fitness_history.append(best_fitness)
            self.temperature_history.append(temp)
            temp *= self.cooling_rate
            
            if self.verbose and len(self.fitness_history) % 50 == 0:
                print(f"温度 {temp:.4f}: 最优适应度 = {best_fitness:.6f}")
        
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        
        return {
            "success": True,
            "optimal_solution": best_solution.tolist(),
            "optimal_value": float(best_fitness),
            "fitness_history": self.fitness_history,
            "temperature_history": self.temperature_history,
            "final_temperature": temp
        }
