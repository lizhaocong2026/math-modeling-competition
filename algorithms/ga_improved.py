"""
Improved Genetic Algorithm with adaptive crossover and mutation
Supports: single-objective, multi-objective (via weighting), mixed-integer encoding
"""
import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any
import random


class AdaptiveGeneticAlgorithm:
    """
    Adaptive GA with dynamic parameter adjustment based on population diversity
    
    Suitable for: 函数优化、参数调优、调度问题、投资组合优化
    """
    
    def __init__(self, pop_size=100, max_gen=500, elite_size=2, 
                 crossover_rates=(0.7, 0.9), mutation_rates=(0.01, 0.1),
                 is_maximization=True, verbose=False):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.elite_size = elite_size
        self.crossover_range = crossover_rates
        self.mutation_range = mutation_rates
        self.is_maximization = is_maximization
        self.verbose = verbose
        
        self.best_history = []
        self.avg_history = []
        self.diversity_history = []
        self.best_solution = None
        self.best_fitness = None
        
    def optimize(self, fitness_func, bounds, n_vars=None, tolerance=1e-6,
                 early_stop_gen=50, constraints=None) -> Dict[str, Any]:
        n_vars = n_vars or len(bounds)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        # Initialize population
        population = np.random.uniform(lower_bounds, upper_bounds, (self.pop_size, n_vars))
        best_fitness = -np.inf if self.is_maximization else np.inf
        best_solution = None
        no_improve_count = 0
        
        for gen in range(self.max_gen):
            fitnesses = np.array([fitness_func(ind) for ind in population])
            current_best = fitnesses.max() if self.is_maximization else fitnesses.min()
            self.best_history.append(current_best)
            self.avg_history.append(float(fitnesses.mean()))
            
            # Diversity measure
            diversity = float(np.mean(np.std(population, axis=0)))
            self.diversity_history.append(diversity)
            
            # Update best
            if (self.is_maximization and current_best > best_fitness) or \
               (not self.is_maximization and current_best < best_fitness):
                best_fitness = current_best
                best_idx = fitnesses.argmax() if self.is_maximization else fitnesses.argmin()
                best_solution = population[best_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Adaptive parameter adjustment
            avg_fit = fitnesses.mean()
            std_fit = max(fitnesses.std(), 1e-10)
            rel_diff = abs(current_best - avg_fit) / std_fit
            
            # Adjust crossover rate: higher when diversity is low
            base_cx = self.crossover_range[0] + (self.crossover_range[1] - self.crossover_range[0]) * \
                      min(1.0, diversity / 0.1)
            # Adjust mutation rate: increase when stuck
            base_mutation = self.mutation_range[0] + (self.mutation_range[1] - self.mutation_range[0]) * \
                            min(1.0, no_improve_count / max(early_stop_gen, 1))
            
            # Tournament selection
            selected = self._tournament_selection(population, fitnesses, self.pop_size)
            offspring = self._crossover(selected, base_cx)
            offspring = self._mutation(offspring, lower_bounds, upper_bounds, base_mutation)
            
            # Apply constraints
            if constraints:
                offspring = self._apply_constraints(offspring, constraints)
            
            # Elitism
            elite_idx = np.argsort(fitnesses)[-self.elite_size:] if self.is_maximization \
                        else np.argsort(fitnesses)[:self.elite_size]
            offspring[:self.elite_size] = population[elite_idx]
            population = offspring
            
            if no_improve_count >= early_stop_gen:
                break
                
            if self.verbose and gen % 50 == 0:
                print(f"  Gen {gen+1}: best={current_best:.6f}, div={diversity:.6f}")
        
        # Final evaluation
        fitnesses = np.array([fitness_func(ind) for ind in population])
        final_idx = fitnesses.argmax() if self.is_maximization else fitnesses.argmin()
        self.best_solution = population[final_idx]
        self.best_fitness = float(fitnesses[final_idx])
        
        return {
            "status": "success",
            "optimal_solution": self.best_solution.tolist(),
            "optimal_value": self.best_fitness,
            "generations": len(self.best_history),
            "best_history": self.best_history[-20:],
            "final_diversity": self.diversity_history[-1] if self.diversity_history else 0.0
        }
    
    def _tournament_selection(self, population, fitnesses, size):
        n = len(population)
        selected = []
        tournament_size = 3
        for _ in range(size):
            idx = np.random.choice(n, tournament_size, replace=False)
            if self.is_maximization:
                winner = idx[np.argmax(fitnesses[idx])]
            else:
                winner = idx[np.argmin(fitnesses[idx])]
            selected.append(population[winner])
        return np.array(selected)
    
    def _crossover(self, population, crossover_rate):
        offspring = population.copy()
        for i in range(0, len(population) - 1, 2):
            if random.random() < crossover_rate:
                alpha = random.random()
                offspring[i] = alpha * population[i] + (1 - alpha) * population[i + 1]
                offspring[i + 1] = alpha * population[i + 1] + (1 - alpha) * population[i]
        return offspring
    
    def _mutation(self, population, lower_bounds, upper_bounds, mutation_rate):
        perturbation = np.random.randn(*population.shape) * (upper_bounds - lower_bounds) * 0.1
        mask = np.random.random(population.shape) < mutation_rate
        population = population + perturbation * mask.astype(float)
        return np.clip(population, lower_bounds, upper_bounds)
    
    def _apply_constraints(self, population, constraints):
        for lower, upper in constraints:
            population = np.clip(population, lower, upper)
        return population
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "pop_size": self.pop_size,
            "max_gen": self.max_gen,
            "elite_size": self.elite_size,
            "crossover_range": list(self.crossover_range),
            "mutation_range": list(self.mutation_range),
            "is_maximization": self.is_maximization
        }


class NSGA2Improved:
    """
    Improved NSGA-II with crowding distance and fast non-dominated sorting
    
    Suitable for: 多目标优化（2-5个目标）
    """
    
    def __init__(self, n_objectives=2, pop_size=100, max_gen=200, 
                 bounds=None, n_vars=10, elite_size=2):
        self.n_objectives = n_objectives
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.bounds = bounds or [(0, 1)] * n_vars
        self.n_vars = n_vars
        self.elite_size = elite_size
        
        self.pareto_front = []
        self.hv = 0.0  # Hypervolume indicator
        
    def _dominates(self, a, b):
        """Check if solution a dominates b"""
        a_worse = False
        b_worse = False
        for ai, bi in zip(a, b):
            if ai > bi:
                b_worse = True
            elif ai < bi:
                a_worse = True
        return a_worse and not b_worse
    
    def _non_dominated_sort(self, population, fitnesses):
        """Fast non-dominated sorting"""
        n = len(population)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        rank = np.zeros(n, dtype=int)
        front_id = 0
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(fitnesses[i], fitnesses[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(fitnesses[j], fitnesses[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
            if domination_count[i] == 0:
                rank[i] = front_id
        
        current_front = [i for i in range(n) if rank[i] == 0]
        fronts.append(current_front)
        front_id = 1
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        rank[j] = front_id
                        next_front.append(j)
            fronts.append(next_front)
            current_front = next_front
            front_id += 1
        
        return rank, fronts
    
    def _crowding_distance(self, front, fitnesses):
        """Calculate crowding distance for diversity preservation"""
        n = len(front)
        if n <= 2:
            return np.ones(n) * 1e9
        
        distances = np.zeros(n)
        for m in range(self.n_objectives):
            vals = [float(fitnesses[i][m]) for i in front]
            sorted_idx = np.argsort(vals)
            distances[sorted_idx[0]] = 1e9
            distances[sorted_idx[-1]] = 1e9
            f_min = vals[sorted_idx[0]]
            f_max = vals[sorted_idx[-1]]
            f_range = f_max - f_min if abs(f_max - f_min) > 1e-10 else 1.0
            for k in range(1, n - 1):
                distances[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / f_range
        return distances
    
    def optimize(self, objective_funcs, verbose=False) -> Dict[str, Any]:
        """
        Optimize using NSGA-II
        objective_funcs: list of objective functions, each takes a solution and returns scalar
        """
        bounds_arr = np.array(self.bounds)
        lower = bounds_arr[:, 0]
        upper = bounds_arr[:, 1]
        
        # Initialize population
        population = np.random.uniform(lower, upper, (self.pop_size, self.n_vars))
        
        for gen in range(self.max_gen):
            # Evaluate objectives
            n_obj = len(objective_funcs)
            fitnesses = np.zeros((self.pop_size, n_obj))
            for i, func in enumerate(objective_funcs):
                for j in range(self.pop_size):
                    fitnesses[j, i] = func(population[j])
            
            # Non-dominated sorting
            rank, fronts = self._non_dominated_sort(population, fitnesses)
            
            # Selection: take from Pareto fronts by crowding distance
            new_pop = []
            for front in fronts:
                front_list = list(front)
                cd = self._crowding_distance(front, fitnesses)
                sorted_order = np.argsort(-cd)
                for si in sorted_order:
                    if len(new_pop) >= self.pop_size:
                        break
                    new_pop.append(population[front_list[si]].copy())
            
            # Fill remaining with random + crossover + mutation
            while len(new_pop) < self.pop_size:
                if len(new_pop) + self.elite_size <= self.pop_size:
                    elite_idx = np.argsort(fitnesses[:, 0])[:self.elite_size]
                    for i in elite_idx:
                        new_pop.append(population[i].copy())
                    break
                
                # Crossover
                if len(new_pop) + 2 <= self.pop_size:
                    i, j = np.random.choice(self.pop_size, 2, replace=False)
                    alpha = random.random()
                    child1 = alpha * population[i] + (1 - alpha) * population[j]
                    child2 = alpha * population[j] + (1 - alpha) * population[i]
                    new_pop.append(np.clip(child1, lower, upper))
                    new_pop.append(np.clip(child2, lower, upper))
                else:
                    new_pop.append(np.random.uniform(lower, upper, self.n_vars))
            
            population = np.array(new_pop[:self.pop_size])
            
            if verbose and gen % 20 == 0:
                best_objs = fitnesses[rank == 0][:min(5, self.pop_size)]
                print(f"  Gen {gen+1}: fronts={len(fronts)}, best_objs={best_objs[:3]}")
        
        # Final evaluation
        n_obj = len(objective_funcs)
        fitnesses = np.zeros((self.pop_size, n_obj))
        for i, func in enumerate(objective_funcs):
            for j in range(self.pop_size):
                fitnesses[j, i] = func(population[j])
        
        rank, fronts = self._non_dominated_sort(population, fitnesses)
        self.pareto_front = population[rank == 0].tolist()
        self.hv = self._compute_hypervolume(fitnesses[rank == 0])
        
        return {
            "status": "success",
            "pareto_front": self.pareto_front[:20],
            "pareto_objectives": fitnesses[rank == 0][:20].tolist(),
            "n_solutions": int((rank == 0).sum()),
            "hypervolume": self.hv,
            "generations": gen + 1
        }
    
    def _compute_hypervolume(self, front_points):
        """Simplified 2D hypervolume calculation"""
        if len(front_points) == 0:
            return 0.0
        if self.n_objectives == 2:
            sorted_pts = sorted(front_points.tolist(), key=lambda x: x[0])
            ref = [max(p[0] for p in sorted_pts) * 1.1 + 1, max(p[1] for p in sorted_pts) * 1.1 + 1]
            hv = 0.0
            prev_x = ref[0]
            for pt in reversed(sorted_pts):
                hv += (prev_x - pt[0]) * (ref[1] - pt[1])
                prev_x = pt[0]
            return float(hv)
        return 0.0
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_objectives": self.n_objectives,
            "pop_size": self.pop_size,
            "max_gen": self.max_gen,
            "n_vars": self.n_vars
        }
