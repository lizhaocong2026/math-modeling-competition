# CUMCM 2025 A - Smart Grid Load Forecasting and Scheduling Optimization
import numpy as np
from typing import Dict, Any
from algorithms.transformer import SimpleTransformer, TransformerEnsemble
from algorithms.nsga2 import NSGAII
from algorithms.grey_model import GM11
from algorithms.stl_decompose import STLDecomposer


class SmartGridOptimizer:
    '''Smart grid load forecasting and multi-objective scheduling'''

    def __init__(self, horizon=24):
        self.horizon = horizon
        np.random.seed(2025)

    def load_forecast_transformer(self, historical_data):
        seq_len = min(48, len(historical_data))
        X = historical_data[-seq_len:].reshape(1, seq_len, 1)
        y = historical_data[-seq_len:]
        ensemble = TransformerEnsemble()
        ensemble.fit(X, y, epochs=20)
        forecast = ensemble.predict_steps(X, steps=self.horizon)
        return {"method": "TransformerEnsemble", "forecast": forecast}

    def load_forecast_gm(self, historical, steps=24):
        gm = GM11()
        result = gm.fit_predict(historical[:min(20, len(historical))], steps=steps)
        return result.get("forecast", historical[-steps:])

    def solve_multi_objective(self, forecast):
        def objectives(x):
            cost = np.sum(x[:3] * np.array([0.3, 0.5, 0.8]))
            carbon = np.sum(x[:3] * np.array([0.1, 0.3, 0.9]))
            reliability = 1.0 - np.abs(np.sum(x[3:]) / max(len(x[3:]), 1) - 0.5)
            return np.array([cost * 1000, carbon * 100, -reliability * 100])

        nsga = NSGAII(n_objectives=3, pop_size=40, max_gen=60)
        n_vars = 6
        bounds = [(0, 1)] * n_vars
        result = nsga.optimize(objective_func=objectives, bounds=bounds)
        pareto_objs = result.get("pareto_objectives", [])
        return {
            "method": "NSGA-II (3-objective)",
            "pareto_solutions": result.get("n_solutions", 0),
            "best_cost": float(pareto_objs[0][0]) if pareto_objs else None,
            "best_carbon": float(pareto_objs[0][1]) if pareto_objs else None,
            "best_reliability": float(-pareto_objs[0][2] / 100) if pareto_objs else None,
        }

    def full_pipeline(self, historical):
        decomposer = STLDecomposer(period=24, robust=True)
        decomp_result = decomposer.fit(historical)
        transformer_result = self.load_forecast_transformer(historical)
        optimization_result = self.solve_multi_objective(transformer_result["forecast"])
        return {
            "forecast": transformer_result["forecast"],
            "optimization": optimization_result,
            "decomposition_status": decomp_result.get("status", "unknown"),
        }


if __name__ == "__main__":
    np.random.seed(42)
    t = np.arange(0, 720, 1)
    base_load = 500
    daily_pattern = 100 * np.sin(2 * np.pi * (t - 14) / 24)
    weekly_pattern = 30 * np.sin(2 * np.pi * t / 168)
    noise = np.random.normal(0, 20, 720)
    load_data = np.maximum(base_load + daily_pattern + weekly_pattern + noise, 100)

    optimizer = SmartGridOptimizer(horizon=24)
    result = optimizer.full_pipeline(load_data)
    print("CUMCM 2025 A - Smart Grid")
    print("Method:", result["optimization"]["method"])
    print("Pareto solutions:", result["optimization"]["pareto_solutions"])
    print("Best cost:", round(result["optimization"]["best_cost"], 2))
    print("Best reliability:", round(result["optimization"]["best_reliability"] * 100, 2), "%")
