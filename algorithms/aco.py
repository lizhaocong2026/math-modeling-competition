"""
蚁群算法 (Ant Colony Optimization)
用于求解旅行商问题(TSP)等组合优化问题
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import copy


class AntColony:
    """蚁群优化算法"""
    
    def __init__(
        self,
        n_ants: int = 50,
        max_iter: int = 200,
        alpha: float = 1.0,      # 信息素重要性
        beta: float = 2.5,       # 启发式因子重要性
        rho: float = 0.5,        # 信息素挥发系数
        Q: float = 100.0,        # 信息素释放量
        verbose: bool = False
    ):
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.verbose = verbose
        
        self.best_path = None
        self.best_length = None
        self.best_history = []
        
    def solve_tsp(
        self,
        distance_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        求解TSP问题
        
        参数:
            distance_matrix: 距离矩阵 (n×n)
            
        返回:
            包含最优路径和长度的字典
        """
        n = distance_matrix.shape[0]
        
        # 初始化信息素矩阵
        pheromone = np.ones((n, n)) * 0.1
        
        self.best_length = float('inf')
        self.best_path = None
        self.best_history = []
        
        for iteration in range(self.max_iter):
            # 所有蚂蚁构建解
            all_paths = []
            all_lengths = []
            
            for _ in range(self.n_ants):
                path, length = self._construct_solution(
                    distance_matrix, pheromone, n
                )
                all_paths.append(path)
                all_lengths.append(length)
                
                # 更新全局最优
                if length < self.best_length:
                    self.best_length = length
                    self.best_path = path.copy()
            
            self.best_history.append(self.best_length)
            
            # 更新信息素
            pheromone = self._update_pheromone(
                pheromone, all_paths, all_lengths, distance_matrix
            )
            
            if self.verbose and iteration % 50 == 0:
                print(f"迭代 {iteration}: 最短路径 = {self.best_length:.4f}")
        
        return {
            "success": True,
            "optimal_path": self.best_path,
            "optimal_length": float(self.best_length),
            "best_fitness_history": self.best_history,
            "iterations": len(self.best_history)
        }
    
    def _construct_solution(
        self,
        dist_matrix: np.ndarray,
        pheromone: np.ndarray,
        n: int
    ) -> Tuple[List[int], float]:
        """单只蚂蚁构建完整路径"""
        visited = [False] * n
        path = []
        length = 0
        current = np.random.randint(n)
        path.append(current)
        visited[current] = True
        
        for _ in range(n - 1):
            # 计算转移概率
            probabilities = np.zeros(n)
            for j in range(n):
                if not visited[j]:
                    tau = pheromone[current, j] ** self.alpha
                    eta = (1.0 / dist_matrix[current, j]) ** self.beta
                    probabilities[j] = tau * eta
            
            probabilities /= probabilities.sum()
            
            # 轮盘赌选择
            next_city = np.random.choice(n, p=probabilities)
            
            path.append(next_city)
            visited[next_city] = True
            length += dist_matrix[current, next_city]
            current = next_city
        
        # 回到起点
        length += dist_matrix[current, path[0]]
        
        return path, length
    
    def _update_pheromone(
        self,
        pheromone: np.ndarray,
        paths: List[List[int]],
        lengths: List[float],
        dist_matrix: np.ndarray
    ) -> np.ndarray:
        """信息素更新"""
        # 蒸发
        pheromone = (1 - self.rho) * pheromone
        
        # 释放
        for path, length in zip(paths, lengths):
            delta = self.Q / length if length > 0 else 0
            for i in range(len(path)):
                j = (i + 1) % len(path)
                pheromone[path[i], path[j]] += delta
                pheromone[path[j], path[i]] += delta
        
        return pheromone
