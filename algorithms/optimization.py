"""
线性规划求解器
支持标准形式: min c^T x, s.t. A_eq x = b_eq, A_ineq x <= b_ineq
使用 scipy.optimize.linprog
"""
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds, NonlinearConstraint
from typing import Optional, Tuple, Dict, Any
import warnings


class LinearProgramming:
    """线性规划求解器"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.result = None
        self.x_opt = None
        self.f_opt = None
        
    def solve(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[list] = None,
        method: str = "highs",
        max_iter: int = 10000
    ) -> Dict[str, Any]:
        """
        求解线性规划问题
        
        参数:
            c: 目标函数系数向量
            A_ub: 不等式约束矩阵 (m×n)
            b_ub: 不等式约束向量 (m,)
            A_eq: 等式约束矩阵 (p×n)
            b_eq: 等式约束向量 (p,)
            bounds: 变量边界列表 [(lower, upper), ...]
            method: 求解方法 ('highs', 'simplex', 'interior-point')
            max_iter: 最大迭代次数
            
        返回:
            包含最优解、最优值、状态信息的字典
        """
        c = np.asarray(c, dtype=float)
        n = len(c)
        
        # 设置默认边界
        if bounds is None:
            bounds = [(0, None)] * n
        
        # 构建bounds参数
        if isinstance(bounds[0], (int, float)):
            bounds = [(bounds[0], bounds[1])] * n
        elif len(bounds) == 1 and isinstance(bounds[0], tuple):
            bounds = [bounds[0]] * n
        
        # 调用求解器
        try:
            self.result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=method,
                options={
                    'maxiter': max_iter,
                    'disp': self.verbose
                }
            )
            
            if self.result.success:
                self.x_opt = self.result.x
                self.f_opt = self.result.fun
            else:
                raise ValueError(f"求解失败: {self.result.message}")
                
        except Exception as e:
            warnings.warn(f"高维线性规划求解失败，尝试fallback方法: {e}")
            self.result = self._simplex_fallback(c, A_ub, b_ub, A_eq, b_eq, bounds)
        
        return self._format_result()
    
    def _simplex_fallback(
        self, 
        c, A_ub, b_ub, A_eq, b_eq, bounds
    ) -> Any:
        """Simplex算法fallback实现"""
        return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='simplex')
    
    def _format_result(self) -> Dict[str, Any]:
        """格式化结果"""
        if self.result is None:
            return {"success": False, "message": "未求解"}
        
        return {
            "success": self.result.success,
            "optimal_value": float(self.f_opt) if self.f_opt is not None else None,
            "optimal_solution": self.x_opt.tolist() if self.x_opt is not None else None,
            "message": self.result.message if self.result else "未求解",
            "status": self.result.status if self.result else -1
        }
    
    def solve_multiple(self, problems: list) -> list:
        """批量求解多个线性规划问题"""
        results = []
        for prob in problems:
            result = self.solve(**prob)
            results.append(result)
        return results


class IntegerProgramming(LinearProgramming):
    """整数规划求解器"""
    
    def solve(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        integrality: Optional[np.ndarray] = None,
        bounds: Optional[list] = None,
        method: str = "highs"
    ) -> Dict[str, Any]:
        """
        求解整数规划问题
        
        参数:
            c: 目标函数系数向量
            A_ub: 不等式约束矩阵
            b_ub: 不等式约束向量
            A_eq: 等式约束矩阵
            b_eq: 等式约束向量
            integrality: 变量类型 (0=连续, 1=整数)
            bounds: 变量边界
            method: 求解方法
        """
        c = np.asarray(c, dtype=float)
        n = len(c)
        
        if integrality is None:
            integrality = np.ones(n)
        else:
            integrality = np.asarray(integrality, dtype=int)
        
        if bounds is None:
            bounds = [(0, None)] * n
        
        try:
            # 使用scipy的milp求解器
            self.result = milp(
                c=c,
                bounds=Bounds(
                    lb=np.array([b[0] for b in bounds] if isinstance(bounds[0], tuple) else [bounds[0]] * n),
                    ub=np.array([b[1] for b in bounds] if isinstance(bounds[0], tuple) else [bounds[1]] * n)
                ),
                constraints=LinearConstraint(A_ub or np.zeros((0, n)), -np.inf, b_ub or np.inf),
                integrality=integrality,
                method=method
            )
            
            if self.result.success:
                self.x_opt = self.result.x
                self.f_opt = self.result.fun
            else:
                # Fallback to linear programming + rounding
                self.x_opt = self._round_to_integer(
                    c, A_ub, b_ub, A_eq, b_eq, bounds
                )
                self.f_opt = c @ self.x_opt
                
        except Exception as e:
            warnings.warn(f"整数规划求解失败，使用LP relaxation + 取整: {e}")
            self.x_opt = self._round_to_integer(c, A_ub, b_ub, A_eq, b_eq, bounds)
            self.f_opt = c @ self.x_opt
        
        return self._format_result()
    
    def _round_to_integer(self, c, A_ub, b_ub, A_eq, b_eq, bounds):
        """LP relaxation + 取整"""
        lp_result = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='highs')
        if lp_result.success:
            return np.round(lp_result.x).astype(int)
        return np.zeros(len(c))
    
    def solve_assignment(self, cost_matrix: np.ndarray) -> Dict[str, Any]:
        """
        求解指派问题 (Assignment Problem)
        最小化总成本
        """
        n = cost_matrix.shape[0]
        c = cost_matrix.flatten()
        
        # 构建约束矩阵
        A_eq = np.zeros((2*n, n*n))
        for i in range(n):
            # 每个任务只能分配给一个人
            A_eq[i, i*n:(i+1)*n] = 1
            # 每个人只能接一个任务
            A_eq[n+i, i::n] = 1
        
        b_eq = np.ones(2*n)
        
        # 所有变量都是0-1整数
        bounds = [(0, 1)] * (n * n)
        integrality = np.ones(n * n)
        
        return self.solve(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, integrality=integrality)


class NonlinearProgramming:
    """非线性规划求解器"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.result = None
        self.x_opt = None
        self.f_opt = None
        
    def solve(
        self,
        fun,
        x0: np.ndarray,
        args: tuple = (),
        bounds: Optional[list] = None,
        constraints: Optional[list] = None,
        method: str = "SLSQP",
        options: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        求解非线性规划问题
        
        参数:
            fun: 目标函数 f(x)
            x0: 初始点
            args: 附加参数
            bounds: 变量边界
            constraints: 约束列表 [{'type': 'eq/ineq', 'fun': func}]
            method: 求解方法
            options: 求解选项
        """
        from scipy.optimize import minimize
        
        x0 = np.asarray(x0, dtype=float)
        
        default_options = {
            'maxiter': 1000,
            'ftol': 1e-12,
            'gtol': 1e-10,
            'disp': self.verbose
        }
        if options:
            default_options.update(options)
        
        try:
            self.result = minimize(
                fun,
                x0,
                args=args,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=default_options
            )
            
            if self.result.success:
                self.x_opt = self.result.x
                self.f_opt = float(self.result.fun)
            else:
                raise ValueError(f"求解失败: {self.result.message}")
                
        except Exception as e:
            warnings.warn(f"非线性规划求解失败: {e}")
            # 尝试多个初始点
            self.x_opt, self.f_opt = self._multi_start(fun, x0, bounds, constraints)
        
        return self._format_result()
    
    def _multi_start(self, fun, x0, bounds, constraints, n_starts: int = 10):
        """多起点优化"""
        best_x = x0.copy()
        best_f = fun(x0)
        
        for i in range(n_starts):
            x_start = best_x + np.random.randn(len(x0)) * 0.1
            if bounds:
                lb = [b[0] for b in bounds] if bounds else -np.inf
                ub = [b[1] for b in bounds] if bounds else np.inf
                x_start = np.clip(x_start, lb, ub)
            
            try:
                res = minimize(fun, x_start, bounds=bounds, constraints=constraints)
                if res.success and res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except:
                continue
        
        return best_x, best_f
    
    def _format_result(self) -> Dict[str, Any]:
        if self.result is None:
            return {"success": False, "message": "未求解"}
        
        return {
            "success": self.result.success if self.result else False,
            "optimal_value": float(self.f_opt) if self.f_opt is not None else None,
            "optimal_solution": self.x_opt.tolist() if self.x_opt is not None else None,
            "message": self.result.message if self.result else "未求解",
            "nit": self.result.nit if self.result else 0
        }
    
    def solve_constrained(
        self,
        fun,
        x0: np.ndarray,
        eq_constraints: list = None,
        ineq_constraints: list = None,
        bounds: list = None
    ) -> Dict[str, Any]:
        """
        求解带约束的非线性规划
        
        参数:
            fun: 目标函数
            x0: 初始点
            eq_constraints: 等式约束列表
            ineq_constraints: 不等式约束列表
            bounds: 变量边界
        """
        constraints = []
        if eq_constraints:
            for c in eq_constraints:
                constraints.append({'type': 'eq', 'fun': c})
        if ineq_constraints:
            for c in ineq_constraints:
                constraints.append({'type': 'ineq', 'fun': c})
        
        return self.solve(fun, x0, bounds=bounds, constraints=constraints)
