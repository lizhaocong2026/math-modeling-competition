"""
时间序列高级模型
包括状态空间模型、卡尔曼滤波等
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple, List


class KalmanFilter:
    """卡尔曼滤波器"""
    
    def __init__(self, A: np.ndarray, B: np.ndarray, H: np.ndarray,
                 Q: np.ndarray, R: np.ndarray):
        """
        初始化卡尔曼滤波器
        
        参数:
            A: 状态转移矩阵
            B: 控制输入矩阵
            H: 观测矩阵
            Q: 过程噪声协方差
            R: 观测噪声协方差
        """
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = None  # 状态估计
        self.P = None  # 协方差
        
    def initialize(self, x0: np.ndarray, P0: np.ndarray):
        """初始化状态"""
        self.x = x0.copy()
        self.P = P0.copy()
        
    def predict(self, u: Optional[np.ndarray] = None):
        """预测步骤"""
        if u is not None:
            self.x = self.A @ self.x + self.B @ u
        else:
            self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        
    def update(self, z: np.ndarray):
        """更新步骤"""
        y = z - self.H @ self.x  # 创新
        S = self.H @ self.P @ self.H.T + self.R  # 创新协方差
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
    
    def filter(self, observations: np.ndarray, 
               controls: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        执行完整滤波
        
        参数:
            observations: 观测序列
            controls: 控制输入序列
            
        返回:
            滤波结果
        """
        n_steps = len(observations)
        filtered_states = []
        predicted_states = []
        innovations = []
        
        for t in range(n_steps):
            # 预测
            u = controls[t] if controls is not None else None
            self.predict(u)
            predicted_states.append(self.x.copy())
            
            # 更新
            self.update(observations[t])
            filtered_states.append(self.x.copy())
            
            # 计算创新
            innovation = observations[t] - self.H @ predicted_states[-1]
            innovations.append(innovation)
        
        return {
            "filtered_states": np.array(filtered_states),
            "predicted_states": np.array(predicted_states),
            "innovations": np.array(innovations),
            "final_state": self.x,
            "final_covariance": self.P
        }


class StateSpaceModel:
    """状态空间模型"""
    
    def __init__(self, n_states: int, n_obs: int):
        self.n_states = n_states
        self.n_obs = n_obs
        self.A = None
        self.B = None
        self.H = None
        self.Q = None
        self.R = None
        
    def fit(self, observations: np.ndarray, 
            method: str = "mle") -> Dict[str, Any]:
        """
        拟合状态空间模型
        
        参数:
            observations: 观测数据
            method: 拟合方法 ('mle', 'em')
            
        返回:
            拟合结果
        """
        n_t = len(observations)
        
        # 简化的参数估计
        # 实际应用中应使用EM算法或MLE
        self.A = np.eye(self.n_states) * 0.9
        self.H = np.eye(self.n_obs, self.n_states)
        self.Q = np.eye(self.n_states) * 0.01
        self.R = np.eye(self.n_obs) * 0.1
        self.B = np.zeros((self.n_states, 1))
        
        # 执行卡尔曼滤波
        kf = KalmanFilter(self.A, self.B, self.H, self.Q, self.R)
        x0 = np.zeros(self.n_states)
        P0 = np.eye(self.n_states) * 10
        kf.initialize(x0, P0)
        
        result = kf.filter(observations)
        
        return {
            **result,
            "parameters": {
                "A": self.A.tolist(),
                "H": self.H.tolist(),
                "Q": self.Q.tolist(),
                "R": self.R.tolist()
            }
        }
    
    def forecast(self, observations: np.ndarray, 
                 steps: int = 5) -> np.ndarray:
        """预测未来值"""
        # 先拟合
        self.fit(observations)
        
        # 获取最后状态
        kf = KalmanFilter(self.A, self.B, self.H, self.Q, self.R)
        kf.initialize(np.zeros(self.n_states), np.eye(self.n_states) * 10)
        
        # 滤波
        for obs in observations:
            kf.predict()
            kf.update(obs)
        
        # 预测
        forecasts = []
        for _ in range(steps):
            kf.predict()
            forecast = self.H @ kf.x
            forecasts.append(forecast)
        
        return np.array(forecasts)


class ETSModel:
    """ETS (Error-Trend-Seasonal) 模型"""
    
    def __init__(self, error: str = "add", trend: str = None, 
                 seasonal: str = None, m: int = 12):
        """
        初始化ETS模型
        
        参数:
            error: 误差项 ('add' 或 'mul')
            trend: 趋势项 ('add', 'mul', 或 None)
            seasonal: 季节项 ('add', 'mul', 或 None)
            m: 季节周期
        """
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.m = m
        
        # 参数
        self.alpha = 0.3  # 水平平滑参数
        self.beta = 0.1  # 趋势平滑参数
        self.gamma = 0.1  # 季节平滑参数
        
    def fit(self, data: np.ndarray) -> 'ETSModel':
        """拟合模型"""
        n = len(data)
        
        # 初始化状态
        self.level = np.mean(data[:self.m])
        self.trend = 0
        self.seasonal = np.zeros(self.m)
        
        # 简单初始化季节成分
        if self.seasonal is not None:
            for i in range(self.m):
                seasonal_values = data[i::self.m]
                self.seasonal[i] = np.mean(seasonal_values) - self.level
        
        # 迭代优化参数（简化版）
        for t in range(n):
            self._update(data[t])
        
        return self
    
    def _update(self, y: np.ndarray):
        """更新状态"""
        t_idx = len(self._history) if hasattr(self, '_history') else 0
        self._history.append(y)
        
        # ETS(A,A,N) 模型更新
        if self.seasonal is None:
            # 无季节成分
            self.level = self.alpha * y + (1 - self.alpha) * (self.level + self.trend)
            if self.trend is not None:
                self.trend = self.beta * (self.level - self._history[-2] if len(self._history) > 1 else self.level) + (1 - self.beta) * self.trend
        else:
            # 有季节成分
            s_idx = t_idx % self.m
            self.level = self.alpha * (y - self.seasonal[s_idx]) + (1 - self.alpha) * (self.level + self.trend)
            self.seasonal[s_idx] = self.gamma * (y - self.level) + (1 - self.gamma) * self.seasonal[s_idx]
            if self.trend is not None:
                self.trend = self.beta * (self.level - (self.level - self.trend)) + (1 - self.beta) * self.trend
    
    def forecast(self, steps: int = 1) -> np.ndarray:
        """预测"""
        forecasts = np.zeros(steps)
        n = len(self._history)
        
        for h in range(1, steps + 1):
            forecast = self.level + h * self.trend
            if self.seasonal is not None:
                s_idx = (n + h - 1) % self.m
                forecast += self.seasonal[s_idx]
            forecasts[h-1] = forecast
        
        return forecasts