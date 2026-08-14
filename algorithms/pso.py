"""
粒子群优化算法 (Particle Swarm Optimization)
基于群体智能的元启发式优化算法
"""
import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any


class ParticleSwarm:
    """粒子群优化算法"""
    
    def __init__(
        self,
        n_particles: int = 30,
        max_iter: int = 500,
        w: float = 0.7,          # 惯性权重
        c1: float = 1.5,         # 认知系数
        c2: float = 1.5,         # 社会系数
        verbose: bool = False
    ):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        
        self.best_position = None
        self.best_fitness = None
        self.best_history = []
        
    def optimize(
        self,
        fitness_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        is_maximization: bool = True,
        tolerance: float = 1e-8
    ) -> Dict[str, Any]:
        """
        运行粒子群优化
        
        参数:
            fitness_func: 适应度函数
            bounds: 变量边界
            is_maximization: 是否最大化
            tolerance: 收敛容差
            
        返回:
            包含最优解、最优值的字典
        """
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        # 初始化粒子位置和速度
        positions = np.random.uniform(lower, upper, (self.n_particles, n_dims))
        velocities = np.random.randn(self.n_particles, n_dims) * 0.1
        
        # 初始化个体最优
        individual_best = positions.copy()
        individual_best_fitness = np.array([
            fitness_func(p) for p in positions
        ])
        
        # 初始化全局最优
        global_best_idx = np.argmax(individual_best_fitness) if is_maximization else np.argmin(individual_best_fitness)
        global_best = positions[global_best_idx].copy()
        global_best_fitness = individual_best_fitness[global_best_idx]
        
        best_history = []
        
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                # 更新速度
                r1 = np.random.rand(n_dims)
                r2 = np.random.rand(n_dims)
                
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (individual_best[i] - positions[i])
                    + self.c2 * r2 * (global_best - positions[i])
                )
                
                # 速度限制
                max_vel = (upper - lower) * 0.1
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
                
                # 更新位置
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lower, upper)
                
                # 计算适应度
                fitness = fitness_func(positions[i])
                
                # 更新个体最优
                if (is_maximization and fitness > individual_best_fitness[i]) or \
                   (not is_maximization and fitness < individual_best_fitness[i]):
                    individual_best[i] = positions[i].copy()
                    individual_best_fitness[i] = fitness
            
            # 更新全局最优
            current_global_idx = np.argmax(individual_best_fitness) if is_maximization else np.argmin(individual_best_fitness)
            if (is_maximization and individual_best_fitness[current_global_idx] > global_best_fitness) or \
               (not is_maximization and individual_best_fitness[current_global_idx] < global_best_fitness):
                global_best = individual_best[current_global_idx].copy()
                global_best_fitness = individual_best_fitness[current_global_idx]
            
            best_history.append(global_best_fitness)
            
            if self.verbose and iteration % 50 == 0:
                print(f"迭代 {iteration}: 最优适应度 = {global_best_fitness:.6f}")
        
        self.best_position = global_best
        self.best_fitness = global_best_fitness
        self.best_history = best_history
        
        return {
            "success": True,
            "optimal_solution": global_best.tolist(),
            "optimal_value": float(global_best_fitness),
            "best_fitness_history": best_history,
            "iterations": len(best_history)
        }
