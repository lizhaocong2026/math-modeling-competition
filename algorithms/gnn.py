"""
图论与网络分析
用于交通网络、社交网络等问题
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import heapq


class Graph:
    """图数据结构"""
    
    def __init__(self, n_nodes: int):
        self.n = n_nodes
        self.adjacency = {}
        self.weights = {}
        
        for i in range(n_nodes):
            self.adjacency[i] = []
            self.weights[i] = {}
    
    def add_edge(self, u: int, v: int, weight: float = 1.0):
        """添加边"""
        self.adjacency[u].append(v)
        self.adjacency[v].append(u)
        self.weights[u][v] = weight
        self.weights[v][u] = weight
    
    def dijkstra(self, source: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dijkstra最短路径算法
        
        参数:
            source: 源节点
            
        返回:
            (距离数组, 前驱节点数组)
        """
        dist = np.inf * np.ones(self.n)
        prev = np.ones(self.n) * -1
        dist[source] = 0
        
        pq = [(0, source)]
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist[u]:
                continue
            
            for v in self.adjacency[u]:
                new_dist = dist[u] + self.weights[u].get(v, 1)
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))
        
        return dist, prev
    
    def bfs(self, source: int) -> np.ndarray:
        """BFS遍历"""
        visited = np.zeros(self.n, dtype=bool)
        order = []
        queue = [source]
        visited[source] = True
        
        while queue:
            u = queue.pop(0)
            order.append(u)
            
            for v in self.adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        return np.array(order)
    
    def dfs(self, source: int) -> np.ndarray:
        """DFS遍历"""
        visited = np.zeros(self.n, dtype=bool)
        order = []
        
        def _dfs(u):
            visited[u] = True
            order.append(u)
            for v in self.adjacency[u]:
                if not visited[v]:
                    _dfs(v)
        
        _dfs(source)
        return np.array(order)


class NetworkFlow:
    """网络流问题"""
    
    @staticmethod
    def ford_fulkerson(capacity: np.ndarray, source: int, sink: int) -> Dict[str, Any]:
        """
        Ford-Fulkerson最大流算法
        
        参数:
            capacity: 容量矩阵
            source: 源点
            sink: 汇点
            
        返回:
            最大流结果
        """
        n = capacity.shape[0]
        residual = capacity.copy()
        flow = np.zeros_like(capacity)
        
        def bfs_path():
            visited = np.zeros(n, dtype=bool)
            parent = np.ones(n) * -1
            queue = [source]
            visited[source] = True
            
            while queue:
                u = queue.pop(0)
                for v in range(n):
                    if not visited[v] and residual[u, v] > 0:
                        visited[v] = True
                        parent[v] = u
                        if v == sink:
                            return parent
                        queue.append(v)
            return None
        
        max_flow = 0
        while True:
            parent = bfs_path()
            if parent is None:
                break
            
            # 找增广路径的最小容量
            path_flow = np.inf
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, residual[u, v])
                v = u
            
            # 更新残留网络
            v = sink
            while v != source:
                u = parent[v]
                residual[u, v] -= path_flow
                residual[v, u] += path_flow
                flow[u, v] += path_flow
                flow[v, u] -= path_flow
                v = u
            
            max_flow += path_flow
        
        return {
            "max_flow": max_flow,
            "flow_matrix": flow,
            "residual_matrix": residual
        }


class PageRank:
    """PageRank算法"""
    
    @staticmethod
    def compute(adjacency: np.ndarray, damping: float = 0.85,
                max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """
        计算PageRank
        
        参数:
            adjacency: 邻接矩阵
            damping: 阻尼系数
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        返回:
            PageRank向量
        """
        n = adjacency.shape[0]
        out_degree = adjacency.sum(axis=1)
        
        # 构建转移矩阵
        M = np.zeros((n, n))
        for i in range(n):
            if out_degree[i] > 0:
                for j in range(n):
                    if adjacency[j, i] > 0:
                        M[j, i] = adjacency[j, i] / out_degree[i]
        
        # 迭代计算
        rank = np.ones(n) / n
        
        for _ in range(max_iter):
            new_rank = (1 - damping) / n + damping * M @ rank
            
            if np.linalg.norm(new_rank - rank, 1) < tol:
                break
            rank = new_rank
        
        return rank