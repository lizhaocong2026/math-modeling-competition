"""
凸优化求解器
用于求解凸优化问题
"""
import numpy as np
from typing import Callable, List, Tuple, Dict, Any, Optional
from scipy.optimize import minimize


class ConvexOptimizer:
    """凸优化求解器"""
    
    def __init__(self, method: str = "interior-point"):
        """
        参数:
            method: 求解方法 ('interior-point', 'sqp', 'trust-constr')
        """
        self.method = method
        self.result = None
        
    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        constraints: List[Dict] = None,
        bounds: List[Tuple[float, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        最小化目标函数
        
        参数:
            objective: 目标函数 f(x)
            x0: 初始点
            constraints: 约束列表 [{'type': 'eq/ineq', 'fun': func}]
            bounds: 变量边界
            
        返回:
            优化结果
        """
        options = kwargs.get('options', {})
        options.update({
            'maxiter': kwargs.get('max_iter', 1000),
            'ftol': kwargs.get('tol', 1e-8)
        })
        
        try:
            self.result = minimize(
                objective,
                x0,
                method=self.method,
                bounds=bounds,
                constraints=constraints,
                options=options
            )
            
            return {
                "success": self.result.success,
                "optimal_value": float(self.result.fun),
                "optimal_solution": self.result.x.tolist(),
                "message": self.result.message,
                "iterations": self.result.nit
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def quadratic_programming(
        self,
        Q: np.ndarray,
        c: np.ndarray,
        A: np.ndarray = None,
        b: np.ndarray = None,
        bounds: List[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        二次规划求解
        min 0.5 * x^T Q x + c^T x
        s.t. Ax <= b
        """
        n = len(c)
        x0 = np.zeros(n)
        
        def objective(x):
            return 0.5 * x @ Q @ x + c @ x
        
        constraints = []
        if A is not None and b is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: b - A @ x
            })
        
        return self.minimize(objective, x0, constraints=constraints, bounds=bounds)
    
    def least_squares(
        self,
        A: np.ndarray,
        b: np.ndarray,
        bounds: List[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        最小二乘求解
        min ||Ax - b||^2
        """
        n = A.shape[1]
        x0 = np.linalg.lstsq(A, b, rcond=None)[0]
        
        def objective(x):
            return np.sum((A @ x - b) ** 2)
        
        return self.minimize(objective, x0, bounds=bounds)


class SequentialQuadraticProgramming:
    """序列二次规划（SQP）"""
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
    
    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        constraints: List[Dict] = None,
        bounds: List[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        使用SQP求解约束优化问题
        """
        from scipy.optimize import minimize
        
        # 使用scipy的SLSQP方法
        result = minimize(f, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': self.max_iter, 'ftol': self.tol})
        
        return {
            "success": result.success,
            "optimal_value": float(result.fun),
            "optimal_solution": result.x.tolist(),
            "iterations": result.nit,
            "message": result.message
        }


class InteriorPointMethod:
    """内点法求解器"""
    
    def __init__(self, mu: float = 0.1, max_iter: int = 100):
        """
        参数:
            mu: 中心性参数
            max_iter: 最大迭代次数
        """
        self.mu = mu
        self.max_iter = max_iter
    
    def optimize_linear(
        self,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> Dict[str, Any]:
        """
        线性规划内点法
        
        min c^T x
        s.t. Ax = b, x >= 0
        """
        n = len(c)
        m = len(b)
        
        # 初始化
        x = np.ones(n) * 10
        y = np.zeros(m)
        s = np.ones(n) * 10  # 松弛变量
        
        for k in range(self.max_iter):
            # 计算残差
            r_p = b - A @ x
            r_d = c - A.T @ y - s
            
            # 计算中心性参数
            mu_k = (np.dot(s, x) / n) ** 2
            
            # 求解KKT系统（简化版）
            try:
                # 简化更新
                X = np.diag(x)
                S = np.diag(s)
                
                # 牛顿方向
                dx = np.linalg.solve(A @ X @ S**-1 @ X @ A.T, 
                                    b - A @ x + A @ X @ S**-1 @ (s - mu_k / x))
                
                x_new = x - 0.5 * dx
                x_new = np.maximum(x_new, 1e-10)
                
                # 检查收敛
                if np.linalg.norm(r_p) < self.tol and np.linalg.norm(r_d) < self.tol:
                    return {
                        "success": True,
                        "optimal_value": float(c @ x_new),
                        "optimal_solution": x_new.tolist()
                    }
                
                x = x_new
            except np.linalg.LinAlgError:
                break
        
        return {
            "success": False,
            "message": "未收敛",
            "optimal_value": float(c @ x),
            "optimal_solution": x.tolist()
        }


class DynamicProgrammingSolver:
    """动态规划求解器"""
    
    def __init__(self):
        self.memo = {}
        self.decision_table = {}
    
    def solve_knapsack(
        self,
        weights: List[float],
        values: List[float],
        capacity: float
    ) -> Dict[str, Any]:
        """
        0-1背包问题动态规划求解
        """
        n = len(weights)
        # DP表
        dp = [[0] * (int(capacity) + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(int(capacity) + 1):
                dp[i][w] = dp[i-1][w]
                if weights[i-1] <= w:
                    dp[i][w] = max(dp[i][w], dp[i-1][w-int(weights[i-1])] + values[i-1])
        
        # 回溯
        selected = []
        w = int(capacity)
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i-1)
                w -= int(weights[i-1])
        
        return {
            "max_value": dp[n][int(capacity)],
            "selected_items": selected[::-1],
            "total_weight": sum(weights[i] for i in selected),
            "dp_table": dp
        }
    
    def solve_shortest_path(
        self,
        graph: Dict[int, Dict[int, float]],
        start: int,
        end: int
    ) -> Dict[str, Any]:
        """
        最短路径动态规划
        """
        # Dijkstra算法
        import heapq
        
        dist = {node: float('inf') for node in graph}
        dist[start] = 0
        prev = {node: None for node in graph}
        pq = [(0, start)]
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist[u]:
                continue
            
            for v, w in graph.get(u, {}).items():
                new_dist = dist[u] + w
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))
        
        # 重建路径
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        
        return {
            "distance": dist[end],
            "path": path,
            "success": dist[end] < float('inf')
        }