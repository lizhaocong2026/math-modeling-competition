# CUMCM 2024 B - Carbon Emission Prediction and Optimization
import numpy as np
from typing import Dict, Any
from algorithms.grey_model import GM11
from algorithms.nsga2 import NSGAII
from algorithms.svr import SVRRegressor


class CarbonEmissionOptimizer:
    '''Multi-objective optimization for carbon emission reduction'''

    def __init__(self, num_periods=24):
        self.num_periods = num_periods
        np.random.seed(2024)
        self.industry_factor = np.random.uniform(0.3, 0.7, num_periods)
        self.transport_factor = np.random.uniform(0.2, 0.5, num_periods)
        self.residential_factor = np.random.uniform(0.1, 0.3, num_periods)

    def predict_gm(self, historical: np.ndarray) -> np.ndarray:
        gm = GM11()
        result = gm.fit_predict(historical, steps=self.num_periods)
        return result.get("forecast", historical[-self.num_periods:])

    def predict_svr(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        svr = SVRRegressor(kernel="rbf", C=1.0)
        svr.fit(X, y)
        return svr.predict(X)

    def solve_multi_objective(self) -> Dict[str, Any]:
        def objective_func(x):
            cost = np.sum(x * np.array([100, 80, 50]))
            emission_reduction = np.sum(x * np.array([
                np.mean(self.industry_factor),
                np.mean(self.transport_factor),
                np.mean(self.residential_factor),
            ]))
            economic_impact = np.sum(x * np.array([50, 30, 20]))
            return np.array([cost, -emission_reduction * 100, economic_impact])

        nsga = NSGAII(n_objectives=3, pop_size=80, max_gen=150)
        bounds = [(0, 1)] * 3
        result = nsga.optimize(objective_func=objective_func, bounds=bounds)
        pareto_objs = result.get("pareto_objectives", [])
        best_cost = pareto_objs[0][0] if pareto_objs else 0
        return {
            "method": "NSGA-II (3-objective)",
            "pareto_solutions": result.get("n_solutions", 0),
            "best_cost": float(best_cost),
        }


if __name__ == "__main__":
    opt = CarbonEmissionOptimizer()
    result = opt.solve_multi_objective()
    print("CUMCM 2024 B - Carbon Emission Optimization")
    print("Method:", result["method"])
    print("Pareto solutions:", result["pareto_solutions"])
