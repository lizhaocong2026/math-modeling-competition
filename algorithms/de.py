"""
差分进化算法 (Differential Evolution)
适用于连续空间全局优化问题
"""
import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any


class DifferentialEvolution:
    """差分进化优化算法"""
    
    def __init__(
        self,
        pop_size: int = 50,
        max_gen: int = 500,
        F: float = 0.8,          # 缩放因子
        CR: float = 0.9,         # 交叉概率
        verbose: bool = False
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.F = F
        self.CR = CR
        self.verbose = verbose
        
        self.best_history = []
        self.best_solution = None
        self.best_fitness = None
        
    def optimize(
        self,
        fitness_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        is_maximization: bool = True
    ) -> Dict[str, Any]:
        """
        运行差分进化优化
        
        参数:
            fitness_func: 适应度函数
            bounds: 变量边界
            is_maximization: 是否最大化
            
        返回:
            包含最优解、最优值的字典
        """
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        # 初始化种群
        population = np.random.uniform(lower, upper, (self.pop_size, n_dims))
        fitnesses = np.array([fitness_func(ind) for ind in population])
        
        best_idx = np.argmin(fitnesses) if not is_maximization else np.argmax(fitnesses)
        best_solution = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        
        self.best_history = [best_fitness]
        
        for gen in range(self.max_gen):
            new_population = np.zeros_like(population)
            new_fitnesses = np.zeros(len(population))
            
            for i in range(self.pop_size):
                # 选择三个不同的个体
                indices = [j for j in range(self.pop_size) if j != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # 变异
                mutant = population[a] + self.F * (population[b] - population[c])
                
                # 交叉
                j_rand = np.random.randint(n_dims)
                trial = np.where(
                    np.random.random(n_dims) < self.CR,
                    mutant,
                    population[i]
                )
                trial[j_rand] = mutant[j_rand]  # 确保至少有一维来自突变体
                
                # 边界处理
                trial = np.clip(trial, lower, upper)
                
                new_population[i] = trial
            
            # 计算新种群适应度
            new_fitnesses = np.array([fitness_func(ind) for ind in new_population])
            
            # 选择
            keep = new_fitnesses < fitnesses if not is_maximization else new_fitnesses > fitnesses
            population[keep] = new_population[keep]
            fitnesses[keep] = new_fitnesses[keep]
            
            # 更新全局最优
            current_best_idx = np.argmin(fitnesses) if not is_maximization else np.argmax(fitnesses)
            if (not is_maximization and fitnesses[current_best_idx] < best_fitness) or \
               (is_maximization and fitnesses[current_best_idx] > best_fitness):
                best_fitness = fitnesses[current_best_idx]
                best_solution = population[current_best_idx].copy()
            
            self.best_history.append(best_fitness)
            
            if self.verbose and gen % 100 == 0:
                print(f"代 {gen}: 最佳适应度 = {best_fitness:.6f}")
        
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        
        return {
            "success": True,
            "optimal_solution": best_solution.tolist(),
            "optimal_value": float(best_fitness),
            "best_fitness_history": self.best_history,
            "generations": len(self.best_history)
        }
