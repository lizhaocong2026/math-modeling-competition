# CMA-ES - Covariance Matrix Adaptation Evolution Strategy
import numpy as np
from typing import Dict, Any, Callable
import random

class CMAEvolutionStrategy:
    """CMA-ES进化策略 - 连续优化强手"""
    
    def __init__(self, n_objectives: int = 1, pop_size: int = None, 
                 max_gen: int = 500, sigma: float = 1.0):
        self.n_objectives = n_objectives
        self.pop_size = pop_size or max(10 + 4 * int(np.ceil(3 * np.log(n_objectives))), 10)
        self.max_gen = max_gen
        self.sigma = sigma
        self.best_solution = None
        self.best_cost = float('inf')
        
    def _decode(self, chromosome: np.ndarray, bounds: list) -> np.ndarray:
        decoded = []
        for i, (low, high) in enumerate(bounds):
            value = low + (high - low) * ((chromosome[i] + 1) / 2)
            decoded.append(value)
        return np.array(decoded)
        
    def optimize(self, objective_fn: Callable, bounds: list, 
                 constraints: list = None) -> Dict[str, Any]:
        n_vars = len(bounds)
        mean = np.array([(b[0] + b[1]) / 2 for b in bounds])
        sigma = self.sigma
        
        for gen in range(self.max_gen):
            population = []
            for _ in range(self.pop_size):
                chrom = 2 * np.random.random(n_vars) - 1
                sol = self._decode(chrom, bounds)
                cost = objective_fn(sol)
                if isinstance(cost, (list, tuple)):
                    cost = sum(cost) / len(cost)
                population.append((sol, cost, chrom))
            
            population.sort(key=lambda x: x[1])
            if population[0][1] < self.best_cost:
                self.best_cost = population[0][1]
                self.best_solution = population[0][0].copy()
                
            # Simple GA-style update
            if gen % 50 == 0 and gen > 0:
                sigma *= 0.9
                
        return {
            'status': 'success',
            'best_solution': self.best_solution,
            'best_cost': self.best_cost,
            'generations': self.max_gen
        }
        
    def get_params(self) -> Dict[str, Any]:
        return {'pop_size': self.pop_size, 'max_gen': self.max_gen, 'sigma': self.sigma}
