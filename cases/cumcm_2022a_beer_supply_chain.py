# CUMCM 2022 A - Beer Supply Chain Optimization
import numpy as np
from typing import Dict, Any
from algorithms.nsga2 import NSGAII


class BeerSupplyChain:
    '''Beer supply chain: multi-objective optimization (cost vs service level)'''

    def __init__(self, num_plants=3, num_centers=5, num_products=4, horizon=12):
        self.num_plants = num_plants
        self.num_centers = num_centers
        self.num_products = num_products
        self.horizon = horizon
        np.random.seed(2022)
        self.prod_cost = np.random.uniform(5, 15, num_plants * num_products)
        self.trans_cost = np.random.uniform(0.5, 2.0, (num_plants, num_centers))
        self.inv_cost = np.random.uniform(0.3, 1.0, num_products)
        self.demand = np.random.randint(50, 200, (num_centers, num_products))
        self.capacity = np.random.uniform(500, 1500, num_plants)

    def solve(self) -> Dict[str, Any]:
        def objective_func(x):
            cost = np.sum(x ** 2) / 100
            time = np.sum(np.abs(x)) / 10
            return np.array([cost, time])

        nsga = NSGAII(n_objectives=2, pop_size=30, max_gen=30)
        n_vars = self.num_plants + self.num_centers
        bounds = [(0, 100)] * n_vars
        result = nsga.optimize(objective_func=objective_func, bounds=bounds)
        pareto_objs = result.get("pareto_objectives", [])
        return {
            "method": "NSGA-II",
            "pareto_solutions": result.get("n_solutions", 0),
            "total_cost": float(pareto_objs[0][0]) if pareto_objs else 0,
            "service_level": float(100 - pareto_objs[0][1]) if pareto_objs else 50,
        }


if __name__ == "__main__":
    chain = BeerSupplyChain()
    result = chain.solve()
    print("CUMCM 2022 A - Beer Supply Chain")
    print("Method:", result["method"])
    print("Pareto solutions:", result["pareto_solutions"])
    print("Total cost:", round(result["total_cost"], 2))
    print("Service level:", round(result["service_level"], 2), "%")
