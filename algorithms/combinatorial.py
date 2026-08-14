"""
组合优化算法
解决排列、背包、调度等问题
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import random
import itertools


class TravelingSalesman:
    """旅行商问题求解器"""
    
    def __init__(self, distance_matrix: np.ndarray):
        self.n = distance_matrix.shape[0]
        self.dist = distance_matrix
        self.best_tour = None
        self.best_length = float('inf')
        
    def nearest_neighbor(self) -> Tuple[List[int], float]:
        """最近邻启发式"""
        unvisited = list(range(self.n))
        tour = [unvisited.pop(0)]
        
        while unvisited:
            last = tour[-1]
            # 找最近的未访问城市
            nearest = min(unvisited, key=lambda x: self.dist[last, x])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        tour.append(tour[0])  # 返回起点
        length = self._tour_length(tour)
        
        return tour, length
    
    def two_opt(self, initial_tour: List[int], max_iter: int = 1000) -> Tuple[List[int], float]:
        """2-opt局部搜索"""
        tour = initial_tour.copy()
        best_tour = tour.copy()
        best_length = self._tour_length(tour)
        
        for _ in range(max_iter):
            improved = False
            
            # 尝试所有2-opt交换
            for i in range(1, self.n - 1):
                for j in range(i + 1, self.n):
                    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    new_length = self._tour_length(new_tour)
                    
                    if new_length < best_length:
                        best_tour = new_tour
                        best_length = new_length
                        improved = True
            
            if not improved:
                break
        
        self.best_tour = best_tour
        self.best_length = best_length
        
        return best_tour, best_length
    
    def three_opt(self, initial_tour: List[int], max_iter: int = 1000) -> Tuple[List[int], float]:
        """3-opt局部搜索"""
        tour = initial_tour.copy()
        best_tour = tour.copy()
        best_length = self._tour_length(tour)
        
        for _ in range(max_iter):
            improved = False
            
            for i in range(1, self.n - 2):
                for j in range(i + 1, self.n - 1):
                    for k in range(j + 1, self.n):
                        # 尝试不同的3-opt操作
                        new_tours = self._three_opt_variants(tour, i, j, k)
                        for new_tour in new_tours:
                            new_length = self._tour_length(new_tour)
                            if new_length < best_length:
                                best_tour = new_tour
                                best_length = new_length
                                improved = True
            
            if not improved:
                break
        
        return best_tour, best_length
    
    def _three_opt_variants(self, tour, i, j, k):
        """生成3-opt变换后的路径"""
        variants = []
        
        # 原始路径: ... A ... B ... C ... D ...
        # A=tour[i], B=tour[j], C=tour[k], D=tour[k+1]
        
        # 操作1: 反转(i,j)段
        t1 = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        variants.append(t1)
        
        # 操作2: 反转(j,k)段
        t2 = tour[:j+1] + tour[j+1:k+1][::-1] + tour[k+1:]
        variants.append(t2)
        
        # 操作3: 反转(i,k)段
        t3 = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
        variants.append(t3)
        
        return variants
    
    def _tour_length(self, tour: List[int]) -> float:
        """计算路径长度"""
        length = 0
        for i in range(len(tour) - 1):
            length += self.dist[tour[i], tour[i+1]]
        return length


class KnapsackSolver:
    """背包问题求解器"""
    
    @staticmethod
    def dynamic_programming(weights: List[float], values: List[float], capacity: float) -> Dict[str, Any]:
        """
        0-1背包问题 - 动态规划
        
        参数:
            weights: 物品重量
            values: 物品价值
            capacity: 背包容量
            
        返回:
            最优解信息
        """
        n = len(weights)
        # DP表
        dp = [[0] * (int(capacity) + 1) for _ in range(n + 1)]
        
        # 填表
        for i in range(1, n + 1):
            for w in range(int(capacity) + 1):
                dp[i][w] = dp[i-1][w]
                if weights[i-1] <= w:
                    dp[i][w] = max(dp[i][w], dp[i-1][w-int(weights[i-1])] + values[i-1])
        
        # 回溯找解
        selected = []
        w = int(capacity)
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i - 1)
                w -= int(weights[i-1])
        
        return {
            "max_value": dp[n][int(capacity)],
            "selected_items": selected[::-1],
            "total_weight": sum(weights[i] for i in selected),
            "dp_table": dp
        }
    
    @staticmethod
    def greedy(weights: List[float], values: List[float], capacity: float) -> Dict[str, Any]:
        """贪心算法（按价值密度）"""
        n = len(weights)
        # 计算价值密度
        density = [(values[i] / weights[i], i) for i in range(n)]
        density.sort(reverse=True)
        
        selected = []
        total_weight = 0
        total_value = 0
        
        for ratio, idx in density:
            if total_weight + weights[idx] <= capacity:
                selected.append(idx)
                total_weight += weights[idx]
                total_value += values[idx]
        
        return {
            "max_value": total_value,
            "selected_items": selected,
            "total_weight": total_weight,
            "method": "greedy"
        }


class JobShopScheduler:
    """作业车间调度问题"""
    
    def __init__(self, processing_times: np.ndarray, priority_rule: str = "SPT"):
        """
        参数:
            processing_times: 处理时间矩阵 [jobs x machines]
            priority_rule: 优先级规则 (SPT, EDD, CR)
        """
        self.n_jobs = processing_times.shape[0]
        self.n_machines = processing_times.shape[1]
        self.processing_times = processing_times
        self.priority_rule = priority_rule
        
    def schedule_spt(self) -> Dict[str, Any]:
        """最短加工时间优先"""
        job_order = np.argsort(self.processing_times.mean(axis=1))
        return self._make_schedule(job_order)
    
    def schedule_edd(self, due_dates: np.ndarray = None) -> Dict[str, Any]:
        """最早交货期优先"""
        if due_dates is None:
            due_dates = np.arange(1, self.n_jobs + 1) * 10
        job_order = np.argsort(due_dates)
        return self._make_schedule(job_order)
    
    def _make_schedule(self, job_order: np.ndarray) -> Dict[str, Any]:
        """生成调度方案"""
        # 简化版调度
        completion_times = np.zeros((self.n_jobs, self.n_machines))
        
        for j in job_order:
            for m in range(self.n_machines):
                if m == 0:
                    start_time = 0
                else:
                    start_time = completion_times[j, m-1]
                
                # 机器可用时间
                machine_available = max(0, completion_times[j-1, m] if j > 0 else 0)
                start_time = max(start_time, machine_available)
                
                completion_times[j, m] = start_time + self.processing_times[j, m]
        
        makespan = completion_times[-1, -1]
        
        return {
            "job_order": job_order.tolist(),
            "completion_times": completion_times.tolist(),
            "makespan": makespan
        }