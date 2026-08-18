# CUMCM 2019 B题 - 生产调度与优化
import numpy as np
from typing import Dict, Any
from algorithms.ga import GeneticAlgorithm
from algorithms.optimization import LinearProgramming


class ProductionScheduling:
    """生产调度优化问题"""
    
    def __init__(self, num_lines=3, num_products=5):
        self.num_lines = num_lines
        self.num_products = num_products
        self.profit_matrix = None
        self.energy_matrix = None
        self.time_matrix = None
        self.resource_limits = None
        
    def setup_problem(self):
        np.random.seed(2019)
        self.profit_matrix = np.random.uniform(10, 50, (self.num_lines, self.num_products))
        self.energy_matrix = np.random.uniform(5, 30, (self.num_lines, self.num_products))
        self.time_matrix = np.random.uniform(1, 8, (self.num_lines, self.num_products))
        self.resource_limits = {
            "energy": 500,
            "time": 200,
            "demand": np.random.randint(5, 20, self.num_products)
        }
        
    def solve_linear_programming(self):
        self.setup_problem()
        c = -self.profit_matrix.flatten()
        n_vars = self.num_lines * self.num_products
        A_energy = self.energy_matrix.flatten().reshape(1, -1)
        A_time = self.time_matrix.flatten().reshape(1, -1)
        A_ub = np.vstack([A_energy, A_time])
        b_ub = np.array([self.resource_limits["energy"], self.resource_limits["time"]])
        bounds = [(0, None)] * n_vars
        lp = LinearProgramming(verbose=False)
        result = lp.solve(c, A_ub, b_ub, bounds=bounds)
        return {
            "method": "linear_programming",
            "optimal_value": -result["fun"],
            "production_plan": result["x"].reshape(self.num_lines, self.num_products),
            "status": result["success"]
        }
    
    def solve_ga(self, pop_size=100, max_gen=300):
        self.setup_problem()
        def objective(x):
            plan = x.reshape(self.num_lines, self.num_products)
            profit = np.sum(plan * self.profit_matrix)
            energy = np.sum(plan * self.energy_matrix)
            time = np.sum(plan * self.time_matrix)
            penalty = 0
            if energy > self.resource_limits["energy"]:
                penalty += 1000 * (energy - self.resource_limits["energy"])
            if time > self.resource_limits["time"]:
                penalty += 1000 * (time - self.resource_limits["time"])
            return profit - penalty
        ga = GeneticAlgorithm(pop_size=pop_size, max_gen=max_gen)
        result = ga.optimize(func=objective, bounds=[(0, 15)] * (self.num_lines * self.num_products))
        return {
            "method": "genetic_algorithm",
            "optimal_value": result.get("best_fitness", 0),
            "best_solution": result.get("best_solution", result.get("x")),
            "status": "success"
        }


if __name__ == "__main__":
    scheduler = ProductionScheduling()
    print("=" * 60)
    print("CUMCM 2019 B题 - 生产调度优化")
    print("=" * 60)
    lp_result = scheduler.solve_linear_programming()
    print(f"最优利润: {lp_result['optimal_value']:.2f}")
    print(f"求解状态: {lp_result['status']}")
    ga_result = scheduler.solve_ga()
    print(f"GA最优利润: {ga_result['optimal_value']:.2f}")
    print("完成！")