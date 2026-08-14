"""
常微分方程数值解
用于建立动态模型
"""
import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any


class ODESolver:
    """ODE求解器基类"""
    
    def __init__(self):
        self.t_values = []
        self.y_values = []
        
    def solve(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000
    ) -> Dict[str, Any]:
        """求解ODE"""
        raise NotImplementedError


class EulerMethod(ODESolver):
    """欧拉方法 - 一阶精度"""
    
    def solve(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000
    ) -> Dict[str, Any]:
        t0, tf = t_span
        h = (tf - t0) / n_steps
        
        self.t_values = np.linspace(t0, tf, n_steps + 1)
        self.y_values = np.zeros((n_steps + 1, len(y0)))
        self.y_values[0] = y0
        
        for i in range(n_steps):
            self.y_values[i + 1] = self.y_values[i] + h * f(self.t_values[i], self.y_values[i])
        
        return {
            "t": self.t_values,
            "y": self.y_values,
            "method": "Euler",
            "order": 1
        }


class RK4(ODESolver):
    """四阶Runge-Kutta方法 - 最常用"""
    
    def solve(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000
    ) -> Dict[str, Any]:
        t0, tf = t_span
        h = (tf - t0) / n_steps
        
        self.t_values = np.linspace(t0, tf, n_steps + 1)
        self.y_values = np.zeros((n_steps + 1, len(y0)))
        self.y_values[0] = y0
        
        for i in range(n_steps):
            t = self.t_values[i]
            y = self.y_values[i]
            
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            
            self.y_values[i + 1] = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return {
            "t": self.t_values,
            "y": self.y_values,
            "method": "RK4",
            "order": 4
        }


class RungeKuttaFehlberg(ODESolver):
    """RKF45自适应步长方法"""
    
    def __init__(self, tol: float = 1e-6):
        super().__init__()
        self.tol = tol
        # RKF45系数
        self.a2, self.a3, self.a4, self.a5, self.a6 = 1/4, 3/8, 12/13, 1, 1/2
        self.b21 = 1/4
        self.b31, self.b32 = 3/32, 9/32
        self.b41, self.b42, self.b43 = 1932/2197, -7200/2197, 7296/2197
        self.b51, self.b52, self.b53, self.b54 = 439/216, -8, 3680/513, -845/4104
        self.b61, self.b62, self.b63, self.b64, self.b65 = -8/27, 2, -3544/2565, 1859/4104, -11/40
        self.c1, self.c3, self.c4, self.c5, self.c6 = 25/216, 1408/2565, 2197/4104, -1/5, 0
        self.d1, self.d3, self.d4, self.d5, self.d6 = 16/135, 6656/12825, 28561/56430, -9/50, 2/55


class AdamsBashforth(ODESolver):
    """Adams-Bashforth多步法"""
    
    def solve(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000,
        order: int = 4
    ) -> Dict[str, Any]:
        t0, tf = t_span
        h = (tf - t0) / n_steps
        
        self.t_values = np.linspace(t0, tf, n_steps + 1)
        self.y_values = np.zeros((n_steps + 1, len(y0)))
        self.y_values[0] = y0
        
        # 用RK4启动
        rk4 = RK4()
        start = min(4, n_steps)
        start_result = rk4.solve(f, y0, (t0, t0 + start*h), start)
        self.y_values[:start+1] = start_result["y"]
        
        # Adams-Bashforth公式
        ab_coeffs = {
            1: [1],
            2: [3/2, -1/2],
            3: [23/12, -16/12, 5/12],
            4: [55/24, -59/24, 37/24, -9/24]
        }
        coeffs = ab_coeffs[order]
        
        for i in range(start, n_steps):
            y = self.y_values[i]
            t = self.t_values[i]
            
            sum_val = np.zeros(len(y0))
            for j, c in enumerate(coeffs):
                sum_val += c * f(t - j*h, self.y_values[i-j])
            
            self.y_values[i+1] = y + h * sum_val
        
        return {
            "t": self.t_values,
            "y": self.y_values,
            "method": f"Adams-Bashforth-{order}",
            "order": order
        }


# 系统ODE示例
def lotka_volterra(t, y, alpha=1.5, beta=1.0, delta=1.0, gamma=1.0):
    """Lotka-Volterra捕食者-猎物模型"""
    prey, predator = y
    dydt = [alpha*prey - beta*prey*predator,
            delta*prey*predator - gamma*predator]
    return np.array(dydt)


def pendulum(t, y, g=9.8, L=1.0):
    """单摆方程"""
    theta, omega = y
    dydt = [omega, -(g/L)*np.sin(theta)]
    return np.array(dydt)


def logistic_growth(t, y, r=0.1, K=100):
    """Logistic增长模型"""
    N = y[0]
    dNdt = r * N * (1 - N/K)
    return np.array([dNdt])