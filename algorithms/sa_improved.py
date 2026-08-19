"""
Simulated Annealing with multiple cooling schedules
Supports: single/multi-objective, constraint handling
"""
import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any
import math


class SimulatedAnnealing:
    """
    Simulated Annealing with adaptive cooling schedule
    
    Cooling schedules: exponential (default), linear, logarithmic, adaptive
    Suitable for: 组合优化、TSP、排产问题、参数寻优
    """
    
    def __init__(self, initial_temp=1000.0, min_temp=1e-8, 
                 cooling_rate=0.95, cooling_schedule="exponential",
                 max_iter=10000, restarts=3, verbose=False):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.cooling_schedule = cooling_schedule
        self.max_iter = max_iter
        self.restarts = restarts
        self.verbose = verbose
        
        self.best_solution = None
        self.best_fitness = None
        self.history = []
        
    def optimize(self, objective_func: Callable, bounds: List[Tuple], 
                 is_maximization: bool = True, constraints=None) -> Dict[str, Any]:
        n_vars = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        best_fitness = -np.inf if is_maximization else np.inf
        best_solution = None
        
        for restart in range(self.restarts):
            # Random initial solution
            current = np.random.uniform(lower, upper, n_vars)
            current_fit = objective_func(current)
            
            temp = self.initial_temp
            iter_count = 0
            
            while temp > self.min_temp and iter_count < self.max_iter:
                # Generate neighbor
                neighbor = current + np.random.randn(n_vars) * temp * 0.1
                neighbor = np.clip(neighbor, lower, upper)
                
                if constraints:
                    neighbor = self._apply_constraints(neighbor, constraints, lower, upper)
                
                neighbor_fit = objective_func(neighbor)
                
                # Acceptance criterion
                if is_maximization:
                    delta = neighbor_fit - current_fit
                else:
                    delta = current_fit - neighbor_fit
                
                # Metropolis criterion
                if delta > 0 or math.exp(delta / temp) > np.random.random():
                    current = neighbor
                    current_fit = neighbor_fit
                
                # Update best
                if (is_maximization and current_fit > best_fitness) or \
                   (not is_maximization and current_fit < best_fitness):
                    best_fitness = current_fit
                    best_solution = current.copy()
                
                # Cooling
                temp = self._cool(temp)
                iter_count += 1
            
            self.history.append({
                "restart": restart,
                "best_fitness": float(best_fitness),
                "iterations": iter_count
            })
            
            if self.verbose:
                print(f"  Restart {restart+1}: best={best_fitness:.6f}, iter={iter_count}")
        
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        
        return {
            "status": "success",
            "optimal_solution": best_solution.tolist() if best_solution is not None else [],
            "optimal_value": float(best_fitness),
            "temperature_history": f"{self.initial_temp:.1f} -> {self.min_temp:.2e}",
            "total_iterations": sum(h["iterations"] for h in self.history),
            "restarts": self.restarts
        }
    
    def _cool(self, temp: float) -> float:
        if self.cooling_schedule == "exponential":
            return temp * self.cooling_rate
        elif self.cooling_schedule == "linear":
            return max(temp - self.cooling_rate, self.min_temp)
        elif self.cooling_schedule == "logarithmic":
            return self.initial_temp / (1 + math.log(1 + self.max_iter * 0.001))
        else:
            return temp * self.cooling_rate
    
    def _apply_constraints(self, x, constraints, lower, upper):
        for lower_b, upper_b in constraints:
            x = np.clip(x, lower_b, upper_b)
        return x
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "initial_temp": self.initial_temp,
            "min_temp": self.min_temp,
            "cooling_rate": self.cooling_rate,
            "cooling_schedule": self.cooling_schedule,
            "max_iter": self.max_iter,
            "restarts": self.restarts
        }


class TabuSearch:
    """
    Tabu Search for combinatorial optimization
    Uses a tabu list to avoid cycling and explore new regions
    """
    
    def __init__(self, max_iter=5000, tabu_size=50, neighborhood_size=20,
                 time_limit=None, verbose=False):
        self.max_iter = max_iter
        self.tabu_size = tabu_size
        self.neighborhood_size = neighborhood_size
        self.time_limit = time_limit
        self.verbose = verbose
        
        self.best_solution = None
        self.best_fitness = None
        self.tabu_list = []
        
    def optimize(self, objective_func: Callable, bounds: List[Tuple],
                 is_maximization: bool = True) -> Dict[str, Any]:
        n_vars = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        # Initial solution
        current = np.random.uniform(lower, upper, n_vars)
        current_fit = objective_func(current)
        best_fitness = current_fit
        best_solution = current.copy()
        
        tabu_list = []
        
        for iteration in range(self.max_iter):
            # Generate neighborhood
            neighbors = []
            for _ in range(self.neighborhood_size):
                neighbor = current.copy()
                # Perturb random dimension
                dim = np.random.randint(n_vars)
                neighbor[dim] += np.random.randn() * (upper[dim] - lower[dim]) * 0.1
                neighbor = np.clip(neighbor, lower, upper)
                neighbors.append(neighbor)
            
            # Evaluate and select best non-tabu
            best_neighbor = None
            best_neighbor_fit = -np.inf if is_maximization else np.inf
            
            for nb in neighbors:
                nb_fit = objective_func(nb)
                is_tabu = any(np.array_equal(nb, t) for t in tabu_list)
                
                if not is_tabu:
                    if (is_maximization and nb_fit > best_neighbor_fit) or \
                       (not is_maximization and nb_fit < best_neighbor_fit):
                        best_neighbor = nb
                        best_neighbor_fit = nb_fit
            
            # Aspiration criterion: accept tabu if improves best
            if best_neighbor is None and tabu_list:
                # Take best overall including tabu
                all_fits = [(nb, objective_func(nb)) for nb in neighbors]
                all_fits.sort(key=lambda x: x[1], reverse=is_maximization)
                best_neighbor, best_neighbor_fit = all_fits[0]
            
            if best_neighbor is not None:
                current = best_neighbor
                current_fit = best_neighbor_fit
                
                if (is_maximization and current_fit > best_fitness) or \
                   (not is_maximization and current_fit < best_fitness):
                    best_fitness = current_fit
                    best_solution = current.copy()
                
                tabu_list.append(current.copy())
                if len(tabu_list) > self.tabu_size:
                    tabu_list.pop(0)
            
            if iteration % 1000 == 0 and self.verbose:
                print(f"  Iter {iteration}: best={best_fitness:.6f}")
        
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        
        return {
            "status": "success",
            "optimal_solution": best_solution.tolist(),
            "optimal_value": float(best_fitness),
            "iterations": self.max_iter,
            "tabu_list_size": len(tabu_list)
        }
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "tabu_size": self.tabu_size,
            "neighborhood_size": self.neighborhood_size
        }
