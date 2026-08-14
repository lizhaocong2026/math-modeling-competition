"""
数学建模竞赛模板 - 优化类问题
适用于CUMCM A题（优化类题目）
"""
import numpy as np
from typing import Dict, Any, List
import sys
sys.path.insert(0, '..')

from algorithms.optimization import LinearProgramming, NonlinearProgramming
from algorithms.ga import GeneticAlgorithm
from utils.data_preprocessor import DataPreprocessor
from visualizations.model_viz import ModelVisualization


class OptimizationProblemTemplate:
    """优化类问题模板"""
    
    def __init__(self):
        self.lp_solver = LinearProgramming(verbose=True)
        self.nlp_solver = NonlinearProgramming(verbose=True)
        self.ga = GeneticAlgorithm(pop_size=100, max_gen=500)
        self.preprocessor = DataPreprocessor()
        self.viz = ModelVisualization()
        
    def solve_linear_programming(
        self,
        objective_coeffs: np.ndarray,
        constraints: Dict[str, Any],
        bounds: List[tuple] = None
    ) -> Dict[str, Any]:
        """
        求解线性规划问题
        
        标准形式: min c^T x, s.t. A_eq x = b_eq, A_ub x <= b_ub
        
        参数:
            objective_coeffs: 目标函数系数 c
            constraints: 约束条件 {
                'A_eq': 等式约束矩阵,
                'b_eq': 等式约束向量,
                'A_ub': 不等式约束矩阵,
                'b_ub': 不等式约束向量
            }
            bounds: 变量边界 [(lower, upper), ...]
        """
        result = self.lp_solver.solve(
            c=objective_coeffs,
            A_eq=constraints.get('A_eq'),
            b_eq=constraints.get('b_eq'),
            A_ub=constraints.get('A_ub'),
            b_ub=constraints.get('b_ub'),
            bounds=bounds
        )
        
        print("=" * 50)
        print("线性规划求解结果")
        print("=" * 50)
        print(f"最优值: {result['optimal_value']:.6f}")
        print(f"最优解: {result['optimal_solution']}")
        print(f"状态: {result['message']}")
        
        return result
    
    def solve_nlp_with_constraints(
        self,
        objective_func,
        x0: np.ndarray,
        eq_constraints: List[callable] = None,
        ineq_constraints: List[callable] = None,
        bounds: List[tuple] = None
    ) -> Dict[str, Any]:
        """
        求解带约束的非线性规划
        
        参数:
            objective_func: 目标函数 f(x)
            x0: 初始点
            eq_constraints: 等式约束列表
            ineq_constraints: 不等式约束列表
            bounds: 变量边界
        """
        result = self.nlp_solver.solve_constrained(
            fun=objective_func,
            x0=x0,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
            bounds=bounds
        )
        
        print("=" * 50)
        print("非线性规划求解结果")
        print("=" * 50)
        print(f"最优值: {result['optimal_value']:.6f}")
        print(f"最优解: {result['optimal_solution']}")
        
        return result
    
    def solve_with_genetic_algorithm(
        self,
        fitness_func,
        bounds: List[tuple],
        is_maximization: bool = True
    ) -> Dict[str, Any]:
        """
        使用遗传算法求解优化问题
        
        参数:
            fitness_func: 适应度函数
            bounds: 变量边界
            is_maximization: 是否最大化
        """
        result = self.ga.optimize(
            fitness_func=fitness_func,
            bounds=bounds,
            is_maximization=is_maximization
        )
        
        print("=" * 50)
        print("遗传算法求解结果")
        print("=" * 50)
        print(f"最优值: {result['optimal_value']:.6f}")
        print(f"最优解: {result['optimal_solution']}")
        print(f"迭代次数: {result['generations']}")
        
        # 绘制收敛曲线
        self.viz.plot_optimization_convergence(
            result['best_fitness_history'],
            title="遗传算法收敛曲线"
        )
        
        return result
    
    def run_sensitivity_analysis(
        self,
        model_func,
        param_name: str,
        param_range: tuple,
        n_points: int = 20
    ) -> Dict[str, Any]:
        """
        灵敏度分析
        
        参数:
            model_func: 模型函数
            param_name: 参数名称
            param_range: 参数变化范围 (min, max)
            n_points: 采样点数
        """
        params = np.linspace(param_range[0], param_range[1], n_points)
        results = []
        
        for p in params:
            result = model_func(**{param_name: p})
            results.append({
                param_name: p,
                'objective_value': result.get('optimal_value', None)
            })
        
        return {
            'parameter': param_name,
            'range': param_range,
            'results': results
        }
