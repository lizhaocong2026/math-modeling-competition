"""
偏微分方程数值解
用于热传导、波动方程等
"""
import numpy as np
from typing import Callable, Tuple, Dict, Any


class PDESolver:
    """PDE求解器基类"""
    pass


class HeatEquationSolver(PDESolver):
    """一维热传导方程求解"""
    
    def __init__(self, alpha: float = 1.0):
        """
        参数:
            alpha: 热扩散系数
        """
        self.alpha = alpha
    
    def solve_finite_difference(
        self,
        L: float,           # 空间长度
        T: float,           # 时间长度
        nx: int,            # 空间网格数
        nt: int,            # 时间步数
        boundary_cond: Tuple[float, float] = (0, 0),
        initial_cond: Callable[[np.ndarray], np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        有限差分法求解
        
        参数:
            L: 空间域长度
            T: 时间域长度
            nx: 空间网格数
            nt: 时间步数
            boundary_cond: 边界条件 (T(0,t), T(L,t))
            initial_cond: 初始条件函数
            
        返回:
            温度场数据
        """
        dx = L / (nx - 1)
        dt = T / (nt - 1)
        r = self.alpha * dt / dx**2
        
        # 初始化
        x = np.linspace(0, L, nx)
        if initial_cond is None:
            u = np.zeros(nx)
        else:
            u = initial_cond(x)
        
        u[0] = boundary_cond[0]
        u[-1] = boundary_cond[1]
        
        # 时间推进
        u_history = [u.copy()]
        
        for n in range(nt - 1):
            u_new = u.copy()
            for i in range(1, nx - 1):
                u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
            u_new[0] = boundary_cond[0]
            u_new[-1] = boundary_cond[1]
            u = u_new
            u_history.append(u.copy())
            
            # 稳定性检查
            if r > 0.5:
                print(f"Warning: r={r} > 0.5, may be unstable")
        
        return {
            "x": x,
            "t": np.linspace(0, T, nt),
            "u": np.array(u_history),
            "dx": dx,
            "dt": dt,
            "r": r
        }
    
    def solve_crank_nicolson(
        self,
        L: float,
        T: float,
        nx: int,
        nt: int,
        boundary_cond: Tuple[float, float] = (0, 0)
    ) -> Dict[str, Any]:
        """
        Crank-Nicolson方法（无条件稳定）
        """
        dx = L / (nx - 1)
        dt = T / (nt - 1)
        r = self.alpha * dt / (2 * dx**2)
        
        x = np.linspace(0, L, nx)
        u = np.zeros(nx)
        u[0] = boundary_cond[0]
        u[-1] = boundary_cond[1]
        
        # 构建三对角矩阵
        diag = np.ones(nx - 2) * (1 + 2*r)
        off_diag = np.ones(nx - 3) * (-r)
        
        u_history = [u.copy()]
        
        for n in range(nt - 1):
            # 右端项
            b = np.zeros(nx - 2)
            for i in range(nx - 2):
                b[i] = r * u[i] + (1 - 2*r) * u[i+1] + r * u[i+2]
            
            # 修复边界
            b[0] += r * boundary_cond[0]
            b[-1] += r * boundary_cond[1]
            
            # Thomas算法求解
            u_new = self._thomas_algorithm(diag, off_diag, off_diag, b)
            u = np.concatenate([[boundary_cond[0]], u_new, [boundary_cond[1]]])
            u_history.append(u.copy())
        
        return {
            "x": x,
            "t": np.linspace(0, T, nt),
            "u": np.array(u_history),
            "method": "Crank-Nicolson"
        }
    
    def _thomas_algorithm(self, diag, lower, upper, rhs):
        """Thomas算法求解三对角方程组"""
        n = len(rhs)
        cprime = np.zeros(n-1)
        dprime = np.zeros(n)
        
        # 前向消元
        cprime[0] = upper[0] / diag[0]
        for i in range(1, n-1):
            cprime[i] = upper[i] / (diag[i] - lower[i-1] * cprime[i-1])
        
        dprime[0] = rhs[0] / diag[0]
        for i in range(1, n):
            dprime[i] = (rhs[i] - lower[i-1] * dprime[i-1]) / (diag[i] - lower[i-1] * cprime[i-1])
        
        # 回代
        x = np.zeros(n)
        x[-1] = dprime[-1]
        for i in range(n-2, -1, -1):
            x[i] = dprime[i] - cprime[i] * x[i+1]
        
        return x


class WaveEquationSolver(PDESolver):
    """一维波动方程求解"""
    
    def __init__(self, c: float = 1.0):
        """
        参数:
            c: 波速
        """
        self.c = c
    
    def solve_finite_difference(
        self,
        L: float,
        T: float,
        nx: int,
        nt: int,
        boundary_cond: Tuple[float, float] = (0, 0),
        initial_displacement: Callable = None,
        initial_velocity: Callable = None
    ) -> Dict[str, Any]:
        """有限差分法求解波动方程"""
        dx = L / (nx - 1)
        dt = T / (nt - 1)
        r = (self.c * dt / dx) ** 2
        
        x = np.linspace(0, L, nx)
        
        # 初始化
        if initial_displacement:
            u_prev = initial_displacement(x)
        else:
            u_prev = np.zeros(nx)
        
        if initial_velocity:
            v0 = initial_velocity(x)
        else:
            v0 = np.zeros(nx)
        
        u_curr = u_prev.copy()
        u_prev[0] = boundary_cond[0]
        u_prev[-1] = boundary_cond[1]
        u_curr[0] = boundary_cond[0]
        u_curr[-1] = boundary_cond[1]
        
        # 第一步
        for i in range(1, nx - 1):
            u_curr[i] = u_prev[i] + v0[i] * dt + 0.5 * r * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])
        
        u_history = [u_prev.copy(), u_curr.copy()]
        
        # 时间推进
        for n in range(1, nt - 1):
            u_next = np.zeros(nx)
            for i in range(1, nx - 1):
                u_next[i] = 2*u_curr[i] - u_prev[i] + r * (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
            u_next[0] = boundary_cond[0]
            u_next[-1] = boundary_cond[1]
            
            u_prev = u_curr
            u_curr = u_next
            u_history.append(u_curr.copy())
        
        return {
            "x": x,
            "t": np.linspace(0, T, nt),
            "u": np.array(u_history),
            "method": "WaveEquation-FD"
        }


class BlackScholesSolver(PDESolver):
    """Black-Scholes方程求解（金融数学）"""
    
    def __init__(self, r: float = 0.05, sigma: float = 0.2):
        """
        参数:
            r: 无风险利率
            sigma: 波动率
        """
        self.r = r
        self.sigma = sigma
    
    def solve_call_option(
        self,
        S_max: float = 200,
        T: float = 1.0,
        K: float = 100,
        nS: int = 200,
        nT: int = 200
    ) -> Dict[str, Any]:
        """
        求解看涨期权价格
        
        使用变换转换为热传导方程
        """
        # 变换变量
        x = np.log(S / K)
        tau = self.sigma**2 * (T - t) / 2
        
        # 这里用简化方法：直接计算解析解
        from scipy import stats
        
        def bs_price(S, K, r, sigma, T):
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            return S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)
        
        S = np.linspace(0, S_max, nS)
        prices = bs_price(S, K, self.r, self.sigma, T)
        
        return {
            "S": S,
            "price": prices,
            "method": "Black-Scholes-Analytical"
        }