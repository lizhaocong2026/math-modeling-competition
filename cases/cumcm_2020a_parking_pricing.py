# CUMCM 2020 A - Parking Pricing and Slot Allocation
import numpy as np
from typing import Dict, Any
from scipy.optimize import minimize


class ParkingPricing:
    def __init__(self, n_parking=5):
        self.n = n_parking
        np.random.seed(2020)
        
    def demand_model(self, prices, total_capacity):
        utility = np.array([10 - 0.1 * p for p in prices])
        exp_util = np.exp(utility)
        probs = exp_util / np.sum(exp_util)
        return probs * total_capacity
    
    def operator_profit(self, prices, costs, demands):
        return np.sum(demands * (prices - costs))
    
    def solve_stackelberg(self):
        costs = np.random.uniform(5, 15, self.n)
        capacity = np.ones(self.n) * 200
        
        def objective(x):
            prices = x[:self.n]
            demands = self.demand_model(prices, capacity)
            profit = self.operator_profit(prices, costs, demands)
            return -profit
        
        bounds = [(5, 30)] * self.n
        x0 = np.ones(self.n) * 15
        
        result = minimize(objective, x0, bounds=bounds, method='SLSQP')
        optimal_prices = result.x
        demands = self.demand_model(optimal_prices, capacity)
        
        return {
            'method': 'Stackelberg Game',
            'optimal_prices': optimal_prices,
            'demands': demands,
            'total_profit': -result.fun
        }
    
    def full_analysis(self):
        return self.solve_stackelberg()


if __name__ == '__main__':
    model = ParkingPricing(n_parking=5)
    result = model.full_analysis()
    print('CUMCM 2020 A - Parking Pricing')
    print('Optimal prices:', np.round(result['optimal_prices'], 2))
    print('Total profit:', round(result['total_profit'], 2))
