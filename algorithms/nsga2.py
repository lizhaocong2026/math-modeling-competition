"""
NSGA-II 多目标优化算法
用于解决多目标优化问题，生成Pareto前沿
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import copy


class NSGAII:
    """NSGA-II多目标优化算法"""
    
    def __init__(self, n_objectives: int, n_constraints: int = 0,
                 pop_size: int = 100, max_gen: int = 300,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.1):
        self.n_objectives = n_objectives
        self.n_constraints = n_constraints
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
    def non_dominated_sort(self, population: np.ndarray, 
                          objectives: np.ndarray) -> List[List[int]]:
        """非支配排序"""
        n = len(population)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        rank = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(objectives[i], objectives[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        # 第一前沿
        front = [i for i in range(n) if domination_count[i] == 0]
        fronts.append(front)
        rank[front] = 1
        
        # 后续前沿
        while front:
            next_front = []
            for i in front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
                        rank[j] = rank[i] + 1
            fronts.append(next_front)
            front = next_front
        
        return fronts
    
    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """检查a是否支配b"""
        at_least_one_better = False
        for i in range(len(a)):
            if a[i] > b[i]:
                return False
            if a[i] < b[i]:
                at_least_one_better = True
        return at_least_one_better
    
    def crowding_distance(self, population: np.ndarray, 
                          objectives: np.ndarray) -> np.ndarray:
        """拥挤度计算"""
        n = len(population)
        distance = np.zeros(n)
        
        for m in range(self.n_objectives):
            # 按第m个目标排序
            idx = np.argsort(objectives[:, m])
            distance[idx[0]] = float('inf')
            distance[idx[-1]] = float('inf')
            
            obj_range = objectives[idx[-1], m] - objectives[idx[0], m]
            if obj_range == 0:
                continue
            
            for i in range(1, n - 1):
                distance[idx[i]] += (objectives[idx[i+1], m] - objectives[idx[i-1], m]) / obj_range
        
        return distance
    
    def tournament_selection(self, population: np.ndarray, 
                            ranks: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """锦标赛选择"""
        idx1 = np.random.randint(len(population))
        idx2 = np.random.randint(len(population))
        
        if ranks[idx1] < ranks[idx2]:
            return population[idx1]
        elif ranks[idx2] < ranks[idx1]:
            return population[idx2]
        else:
            return population[idx1] if distances[idx1] > distances[idx2] else population[idx2]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                  bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """模拟二进制交叉"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() < self.crossover_rate:
                # 分布指数交叉
                eta = 20
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (0.5 / (1 - u)) ** (1 / (eta + 1))
                
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                
                # 边界处理
                child1[i] = np.clip(child1[i], bounds[i][0], bounds[i][1])
                child2[i] = np.clip(child2[i], bounds[i][0], bounds[i][1])
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray, 
               bounds: List[Tuple[float, float]]) -> np.ndarray:
        """多项式变异"""
        mutation = individual.copy()
        eta = 20
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                delta = bounds[i][1] - bounds[i][0]
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    beta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                mutation[i] = individual[i] + beta * delta
                mutation[i] = np.clip(mutation[i], bounds[i][0], bounds[i][1])
        
        return mutation
    
    def optimize(self, objective_func, bounds: List[Tuple[float, float]],
                 constraints: List[callable] = None) -> Dict[str, Any]:
        """
        执行NSGA-II优化
        
        参数:
            objective_func: 目标函数，返回 ndarray
            bounds: 变量边界
            constraints: 约束函数列表
            
        返回:
            Pareto前沿信息
        """
        n_dims = len(bounds)
        
        # 初始化种群
        population = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            (self.pop_size, n_dims)
        )
        
        self.best_history = []
        
        for gen in range(self.max_gen):
            # 计算目标函数
            objectives = np.array([objective_func(ind) for ind in population])
            
            # 非支配排序
            fronts = self.non_dominated_sort(population, objectives)
            
            # 计算拥挤度
            all_distances = []
            for front in fronts:
                if len(front) > 0:
                    front_obj = objectives[front]
                    dists = self.crowding_distance(population[front], front_obj)
                    all_distances.extend(dists.tolist())
                else:
                    all_distances.extend([0] * len(front))
            
            rank = np.zeros(len(population))
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # 选择、交叉、变异
            new_population = []
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, rank, all_distances)
                parent2 = self.tournament_selection(population, rank, all_distances)
                
                child1, child2 = self.crossover(parent1, parent2, bounds)
                child1 = self.mutate(child1, bounds)
                child2 = self.mutate(child2, bounds)
                
                new_population.extend([child1, child2])
            
            # 合并种群
            combined = np.vstack([population, new_population[:self.pop_size]])
            combined_obj = np.array([objective_func(ind) for ind in combined])
            
            # 选择下一代
            fronts = self.non_dominated_sort(combined, combined_obj)
            
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(front)
                else:
                    # 拥挤度选择
                    remaining = self.pop_size - len(new_population)
                    front_obj = combined_obj[front]
                    dists = self.crowding_distance(combined[front], front_obj)
                    sorted_idx = np.argsort(-dists)
                    new_population.extend([front[i] for i in sorted_idx[:remaining]])
                    break
            
            population = combined[new_population]
            
            # 记录最佳前沿
            if gen % 50 == 0:
                best_front = fronts[0] if fronts else []
                self.best_history.append(objectives[best_front[0]] if best_front else objectives[0])
        
        # 最终结果
        objectives = np.array([objective_func(ind) for ind in population])
        fronts = self.non_dominated_sort(population, objectives)
        
        return {
            "pareto_front": population[fronts[0]].tolist() if fronts else [],
            "pareto_objectives": objectives[fronts[0]].tolist() if fronts else [],
            "n_solutions": len(fronts[0]) if fronts else 0,
            "generations": self.max_gen
        }