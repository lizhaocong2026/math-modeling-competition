"""
示例1: 资源分配优化问题 (CUMCM常见A题类型)
使用线性规划和整数规划求解资源分配问题
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from algorithms.optimization import LinearProgramming, IntegerProgramming
from templates.optimization_template import OptimizationProblemTemplate


def example_linear_programming():
    """示例：线性规划资源分配"""
    print("\n" + "="*60)
    print("示例1: 线性规划资源分配问题")
    print("="*60)
    
    # 问题描述：
    # 某工厂生产两种产品，需要三种原材料
    # 产品A利润100元/件，产品B利润80元/件
    # 原材料限制：甲150kg，乙200kg，丙180kg
    # 产品A消耗：甲2kg，乙3kg，丙1kg
    # 产品B消耗：甲1kg，乙2kg，丙2kg
    # 求最大利润
    
    # 目标函数系数（最小化负利润 = 最大化利润）
    c = np.array([-100, -80])
    
    # 不等式约束矩阵（<=）
    A_ub = np.array([
        [2, 1],  # 原材料甲
        [3, 2],  # 原材料乙
        [1, 2]   # 原材料丙
    ])
    b_ub = np.array([150, 200, 180])
    
    # 变量边界（非负）
    bounds = [(0, None), (0, None)]
    
    # 求解
    lp = LinearProgramming(verbose=True)
    result = lp.solve(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    print("\n【结果】")
    print(f"最优生产方案: A产品 {result['optimal_solution'][0]:.2f} 件, B产品 {result['optimal_solution'][1]:.2f} 件")
    print(f"最大利润: {result['optimal_value']:.2f} 元")
    
    return result


def example_integer_programming():
    """示例：整数规划投资组合"""
    print("\n" + "="*60)
    print("示例2: 整数规划投资组合问题")
    print("="*60)
    
    # 问题描述：
    # 有5个投资项目，每个项目投资金额必须为整数（万元）
    # 总预算100万元，要求最大化收益
    
    np.random.seed(42)
    
    # 各项目的预期收益率
    returns = np.array([0.12, 0.15, 0.10, 0.18, 0.14])
    
    # 各项目单位投资
    costs = np.array([20, 25, 15, 30, 20])
    
    # 目标：最大化收益 = 最小化 -收益
    c = -returns
    
    # 预算约束：总成本 <= 100
    A_ub = np.array([costs])
    b_ub = np.array([100])
    
    # 变量边界和整数约束
    bounds = [(0, None)] * 5
    integrality = np.ones(5)  # 全部为整数
    
    # 求解
    ip = IntegerProgramming()
    result = ip.solve(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=integrality)
    
    print("\n【结果】")
    print(f"各项目投资额: {result['optimal_solution']} 万元")
    print(f"最大收益: {result['optimal_value']:.4f} 万元")
    
    return result


def example_assignment_problem():
    """示例：指派问题"""
    print("\n" + "="*60)
    print("示例3: 指派问题（任务分配）")
    print("="*60)
    
    # 问题描述：
    # 4个工人完成4项任务，每人只能做一个任务
    # 成本矩阵（完成任务所需时间）
    cost_matrix = np.array([
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4]
    ])
    
    print("\n成本矩阵:")
    print(cost_matrix)
    
    # 求解
    ip = IntegerProgramming()
    result = ip.solve_assignment(cost_matrix)
    
    print("\n【结果】")
    print(f"最小总成本: {result['optimal_value']:.2f}")
    print(f"最优分配方案: {result['optimal_solution']}")
    
    return result


def example_genetic_algorithm():
    """示例：遗传算法求解复杂优化问题"""
    print("\n" + "="*60)
    print("示例4: 遗传算法求解非线性优化")
    print("="*60)
    
    # 问题：Sphere函数优化（经典测试函数）
    # min f(x) = sum(xi^2), x in [-5.12, 5.12]^n
    
    def sphere_function(x):
        return np.sum(x ** 2)
    
    n_vars = 10
    bounds = [(-5.12, 5.12)] * n_vars
    
    # 求解
    ga = GeneticAlgorithm(pop_size=100, max_gen=300, verbose=False)
    result = ga.optimize(sphere_function, bounds, is_maximization=False)
    
    print("\n【结果】")
    print(f"最优解: {result['optimal_solution']}")
    print(f"最优值: {result['optimal_value']:.6f}")
    print(f"迭代次数: {result['generations']}")
    
    return result


if __name__ == "__main__":
    # 运行所有示例
    example_linear_programming()
    example_integer_programming()
    example_assignment_problem()
    example_genetic_algorithm()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
