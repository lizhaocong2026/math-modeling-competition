"""
Particle Swarm Optimization (PSO) for continuous optimization
Enhanced PSO with adaptive parameters for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List, Tuple


class ParticleSwarmOptimizer:
    """
    Enhanced Particle Swarm Optimization
    
    Suitable for: 连续空间优化、参数寻优、函数优化
    """
    
    def __init__(self, n_particles: int = 30, max_iter: int = 100,
                 w_min: float = 0.4, w_max: float = 0.9,
                 c1: float = 2.0, c2: float = 2.0):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_min = w_min
        self.w_max = w_max
        self.c1 = c1
        self.c2 = c2
        
        self.best_positions = None
        self.best_scores = None
        self.global_best = None
        self.global_best_score = None
        
    def _fitness(self, X: np.ndarray, bounds: np.ndarray, 
                 objective_func) -> np.ndarray:
        """Evaluate fitness of particles"""
        return np.array([objective_func(x) for x in X])
    
    def optimize(self, objective_func, bounds: List[Tuple[float, float]], 
                 callback=None) -> Dict[str, Any]:
        """
        Run PSO optimization
        
        Args:
            objective_func: Objective function to minimize
            bounds: List of (min, max) for each dimension
            callback: Optional callback function
            
        Returns:
            Optimization results
        """
        bounds = np.array(bounds)
        n_dims = len(bounds)
        
        # Initialize particles
        X = np.random.uniform(
            bounds[:, 0], bounds[:, 1], 
            (self.n_particles, n_dims)
        )
        V = np.random.uniform(-1, 1, (self.n_particles, n_dims))
        
        # Initialize personal best
        personal_best = X.copy()
        personal_best_score = self._fitness(X, bounds, objective_func)
        
        # Initialize global best
        self.global_best_score = np.min(personal_best_score)
        self.global_best = personal_best[np.argmin(personal_best_score)].copy()
        
        self.history = []
        
        for iteration in range(self.max_iter):
            # Adaptive inertia weight
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iter
            
            # Update velocities and positions
            r1 = np.random.rand(self.n_particles, n_dims)
            r2 = np.random.rand(self.n_particles, n_dims)
            
            V = w * V + self.c1 * r1 * (personal_best - X) + self.c2 * r2 * (self.global_best - X)
            
            # Velocity clipping
            V = np.clip(V, -5 * (bounds[:, 1] - bounds[:, 0]), 5 * (bounds[:, 1] - bounds[:, 0]))
            
            X = X + V
            
            # Position clipping
            X = np.clip(X, bounds[:, 0], bounds[:, 1])
            
            # Evaluate fitness
            score = self._fitness(X, bounds, objective_func)
            
            # Update personal best
            improve = score < personal_best_score
            personal_best[improve] = X[improve]
            personal_best_score[improve] = score[improve]
            
            # Update global best
            best_idx = np.argmin(score)
            if score[best_idx] < self.global_best_score:
                self.global_best_score = score[best_idx]
                self.global_best = X[best_idx].copy()
            
            self.history.append({
                "iteration": iteration,
                "best_score": float(self.global_best_score)
            })
            
            if callback:
                callback(iteration, self.global_best, self.global_best_score)
        
        return {
            "status": "success",
            "best_solution": self.global_best.tolist(),
            "best_score": float(self.global_best_score),
            "n_iterations": self.max_iter,
            "history": self.history
        }
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_particles": self.n_particles,
            "max_iter": self.max_iter,
            "w_range": (self.w_min, self.w_max),
            "c1": self.c1,
            "c2": self.c2
        }
