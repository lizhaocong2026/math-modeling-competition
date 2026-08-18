# CUMCM 2024 A题 - 智慧园区综合能源系统优化调度
import numpy as np
from typing import Dict, Any
from algorithms.nsga2 import NSGAII
from algorithms.grey_model import GM11


class SmartParkEnergyManager:
    """智慧园区综合能源系统优化调度"""
    
    def __init__(self, num_periods=24):
        self.num_periods = num_periods
        self.solar_forecast = None
        self.wind_forecast = None
        self.load_forecast = None
        self.electricity_price = None
        
    def load_forecast_data(self, seed=2024):
        np.random.seed(seed)
        t = np.arange(self.num_periods)
        self.solar_forecast = np.maximum(0, 80 * np.sin(np.pi * (t - 6) / 12)) + np.random.randn(24) * 5
        self.wind_forecast = 50 + 20 * np.sin(2 * np.pi * t / 8) + np.random.randn(24) * 10
        self.load_forecast = 100 + 30 * np.sin(np.pi * (t - 14) / 12) + \
                            20 * np.sin(2 * np.pi * t / 24) + np.random.randn(24) * 5
        self.electricity_price = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.4,
                                            0.8, 1.2, 1.5, 1.4, 1.3, 1.0,
                                            0.8, 0.9, 1.1, 1.3, 1.4, 1.2,
                                            0.9, 0.7, 0.5, 0.4, 0.4, 0.3])
    
    def forecast_load_gm(self, historical, steps=24):
        gm = GM11()
        result = gm.fit_predict(historical, steps=steps)
        return result.get("forecast", historical[-steps:])
    
    def solve_multi_objective(self):
        self.load_forecast_data()
        
        def objectives(x):
            charge = np.maximum(x, 0)
            discharge = np.maximum(-x, 0)
            net_load = self.load_forecast - self.solar_forecast - self.wind_forecast + charge - discharge
            net_load = np.maximum(net_load, 0)
            cost = np.sum(self.electricity_price * net_load)
            carbon_factor = 0.8
            emissions = carbon_factor * np.sum(net_load)
            return [cost, emissions]
        
        nsga = NSGAII(pop_size=80, max_gen=150)
        result = nsga.optimize(
            objectives=objectives,
            bounds=[(-10, 10)] * self.num_periods,
            n_objectives=2
        )
        
        return {
            "method": "NSGA-II",
            "pareto_solutions": len(result.get("pareto_front", [])),
            "best_cost": result.get("best_objectives", [None, None])[0],
            "best_emissions": result.get("best_objectives", [None, None])[1],
            "schedule": result.get("best_solution")
        }
    
    def analyze_results(self, result):
        lines = []
        lines.append(f"求解方法: {result['method']}")
        lines.append(f"Pareto解数量: {result['pareto_solutions']}")
        lines.append(f"最低运行成本: {result['best_cost']:.2f} 元")
        lines.append(f"最低碳排放: {result['best_emissions']:.2f} kg CO2")
        return chr(10).join(lines)


if __name__ == "__main__":
    manager = SmartParkEnergyManager(num_periods=24)
    print("=" * 60)
    print("CUMCM 2024 A题 - 智慧园区综合能源系统优化调度")
    print("=" * 60)
    result = manager.solve_multi_objective()
    print("\n优化结果:")
    print(manager.analyze_results(result))
    print("\n完成！")