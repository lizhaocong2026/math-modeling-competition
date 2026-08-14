"""
遗传算法 (Genetic Algorithm)
基于轮盘赌选择、交叉、变异的进化优化算法
"""
import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any
import random


class GeneticAlgorithm:
    """遗传算法求解器"""
    
    def __init__(self, pop_size=100, max_gen=500, crossover_rate=0.8, 
                 mutation_rate=0.1, elite_size=2, verbose=False):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.verbose = verbose
        self.best_history = []
        self.avg_history = []
        self.best_solution = None
        self.best_fitness = None
        
    def optimize(self, fitness_func, bounds, is_maximization=True, tolerance=1e-6, early_stop_gen=50):
        n_vars = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        population = np.random.uniform(lower_bounds, upper_bounds, (self.pop_size, n_vars))
        best_fitness = -np.inf if is_maximization else np.inf
        best_solution = None
        no_improve_count = 0
        
        for gen in range(self.max_gen):
            fitnesses = np.array([fitness_func(ind) for ind in population])
            current_best = fitnesses.max() if is_maximization else fitnesses.min()
            self.best_history.append(current_best)
            self.avg_history.append(fitnesses.mean())
            
            if (is_maximization and current_best > best_fitness) or (not is_maximization and current_best < best_fitness):
                best_fitness = current_best
                best_idx = fitnesses.argmax() if is_maximization else fitnesses.argmin()
                best_solution = population[best_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= early_stop_gen:
                break
            
            selected = self._roulette_selection(population, fitnesses, is_maximization)
            offspring = self._crossover(selected, n_vars)
            offspring = self._mutation(offspring, lower_bounds, upper_bounds)
            elite_indices = np.argsort(fitnesses)[-self.elite_size:] if is_maximization else np.argsort(fitnesses)[:self.elite_size]
            offspring[:self.elite_size] = population[elite_indices]
            population = offspring
            
            if self.verbose and gen % 50 == 0:
                print(f"Gen {gen+1}: best = {current_best:.6f}")
        
        fitnesses = np.array([fitness_func(ind) for ind in population])
        final_best_idx = fitnesses.argmax() if is_maximization else fitnesses.argmin()
        self.best_solution = population[final_best_idx]
        self.best_fitness = fitnesses[final_best_idx]
        
        return {"success": True, "optimal_solution": self.best_solution.tolist(),
                "optimal_value": float(self.best_fitness), "best_fitness_history": self.best_history,
                "avg_fitness_history": self.avg_history, "generations": len(self.best_history)}
    
    def _roulette_selection(self, population, fitnesses, is_maximization):
        if is_maximization:
            probs = (fitnesses - fitnesses.min()) / (fitnesses.max() - fitnesses.min() + 1e-10)
        else:
            probs = (fitnesses.max() - fitnesses) / (fitnesses.max() - fitnesses.min() + 1e-10)
        probs = np.maximum(probs, 1e-10)
        probs /= probs.sum()
        indices = np.random.choice(len(population), size=len(population), p=probs)
        return population[indices]
    
    def _crossover(self, population, n_vars):
        offspring = population.copy()
        for i in range(0, len(population) - 1, 2):
            if random.random() < self.crossover_rate:
                alpha = random.random()
                offspring[i] = alpha * population[i] + (1 - alpha) * population[i + 1]
                offspring[i + 1] = alpha * population[i + 1] + (1 - alpha) * population[i]
        return offspring
    
    def _mutation(self, population, lower_bounds, upper_bounds):
        mask = np.random.random(population.shape) < self.mutation_rate
        population[mask] += np.random.randn(*population.shape)[mask] * 0.1
        return np.clip(population, lower_bounds, upper_bounds)