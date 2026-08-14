"""
约束优化算法
处理复杂约束条件的优化问题
"""
import numpy as np
from typing import Callable, List, Tuple, Dict, Any, Optional
import sys
sys.path.insert(0, '..')

from algorithms.ga import GeneticAlgorithm
from algorithms.de import DifferentialEvolution


class ConstrainedOptimizer:
    """约束优化器"""
    
    def __init__(self, method: str = "ga_penalty"):
        """
        初始化
        
        参数:
            method: 优化方法 ('ga_penalty', 'de_penalty', 'slsqp')
        """
        self.method = method
        self.result = None
        
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        eq_constraints: List[Callable] = None,
        ineq_constraints: List[Callable] = None,
        max_iter: int = 500,
        penalty_weight: float = 1000.0
    ) -> Dict[str, Any]:
        """
        求解带约束优化问题
        
        参数:
            objective: 目标函数
            bounds: 变量边界
            eq_constraints: 等式约束列表
            ineq_constraints: 不等式约束列表 (g(x) <= 0)
            max_iter: 最大迭代次数
            penalty_weight: 惩罚系数
            
        返回:
            优化结果
        """
        n_vars = len(bounds)
        
        if self.method in ["ga_penalty", "de_penalty"]:
            # 罚函数法
            def penalized_objective(x):
                obj = objective(x)
                penalty = 0.0
                
                # 等式约束惩罚
                if eq_constraints:
                    for c in eq_constraints:
                        val = c(x)
                        penalty += penalty_weight * val ** 2
                
                # 不等式约束惩罚
                if ineq_constraints:
                    for c in ineq_constraints:
                        val = c(x)
                        if val > 0:  # 违反约束
                            penalty += penalty_weight * val ** 2
                
                return obj + penalty
            
            if self.method == "ga_penalty":
                ga = GeneticAlgorithm(pop_size=100, max_gen=max_iter)
                self.result = ga.optimize(penalized_objective, bounds, is_maximization=False)
            else:
                de = DifferentialEvolution(pop_size=50, max_gen=max_iter)
                self.result = de.optimize(penalized_objective, bounds, is_maximization=False)
            
        else:
            raise ValueError(f"不支持的方法: {self.method}")
        
        # 检查约束违反
        violation = self._check_constraint_violation(self.result.get('optimal_solution'))
        
        return {
            **self.result,
            "constraint_violation": violation,
            "method": self.method
        }
    
    def _check_constraint_violation(self, x: List[float]) -> float:
        """检查约束违反程度"""
        if x is None:
            return float('inf')
        
        x = np.array(x)
        violation = 0.0
        
        # 边界检查
        for i, (lower, upper) in enumerate([(b[0], b[1]) for b in []]):
            if x[i] < lower:
                violation += (lower - x[i]) ** 2
            if x[i] > upper:
                violation += (x[i] - upper) ** 2
        
        return violation


class MultiObjectiveOptimizer:
    """多目标优化（Pareto前沿）"""
    
    def __init__(self, n_objectives: int = 2):
        self.n_objectives = n_objectives
        self.pareto_front = []
        
    def dominate(self, a: np.ndarray, b: np.ndarray) -> bool:
        """检查a是否支配b"""
        a_better = False
        for i in range(self.n_objectives):
            if a[i] > b[i]:
                a_better = True
                break
            elif a[i] < b[i]:
                return False
        return a_better
    
    def get_pareto_front(self, solutions: List[np.ndarray]) -> List[np.ndarray]:
        """提取Pareto前沿"""
        front = []
        for i, sol_a in enumerate(solutions):
            dominated = False
            for j, sol_b in enumerate(solutions):
                if i != j and self.dominate(sol_b, sol_a):
                    dominated = True
                    break
            if not dominated:
                front.append(sol_a)
        
        self.pareto_front = front
        return front
