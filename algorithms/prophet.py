"""
Prophet 时间序列分解 - 纯NumPy实现
用于趋势、季节性和残差分解
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy.signal import fftconvolve


class ProphetDecompose:
    """
    Prophet风格时间序列分解
    
    模型: y(t) = g(t) + s(t) + h(t) + e(t)
    其中:
        g(t): 趋势项 (分段线性或逻辑回归)
        s(t): 季节性项 (傅里叶级数)
        h(t): 节假日效应
        e(t): 误差项
    """
    
    def __init__(self, growth: str = 'linear', changepoint_prior_scale: float = 0.05,
                 seasonality_mode: str = 'additive', seasonality_width: float = 2*np.pi):
        """
        初始化Prophet分解器
        
        参数:
            growth: 趋势类型 ('linear', 'logistic', 'flat')
            changepoint_prior_scale: 变化点正则化强度
            seasonality_mode: 季节模式 ('additive', 'multiplicative')
            seasonality_width: 季节周期宽度 (弧度)
        """
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.seasonality_width = seasonality_width
        
        # 傅里叶阶数
        self.n_fourier = 10
        self.trend_coeffs = None
        self.seasonal_coeffs = None
        self.changepoints = None
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None,
            seasonality_period: int = 7) -> Dict[str, Any]:
        """
        拟合模型
        
        参数:
            y: 时间序列
            X: 外部变量 (可选)
            seasonality_period: 季节周期长度
        
        返回:
            拟合结果字典
        """
        n = len(y)
        t = np.arange(n)
        
        # 1. 趋势项拟合
        trend = self._fit_trend(t, y)
        
        # 2. 季节性项拟合 (傅里叶级数)
        seasonal = self._fit_seasonality(t, y - trend, period=seasonality_period)
        
        # 3. 残差
        residual = y - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'y_hat': trend + seasonal
        }
    
    def _fit_trend(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """拟合趋势项"""
        if self.growth == 'linear':
            # 线性趋势: y = a + b*t
            A = np.vstack([np.ones(len(t)), t]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            return coeffs[0] + coeffs[1] * t
        elif self.growth == 'flat':
            return np.ones(len(t)) * np.mean(y)
        else:
            # 默认线性
            A = np.vstack([np.ones(len(t)), t]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            return coeffs[0] + coeffs[1] * t
    
    def _fit_seasonality(self, t: np.ndarray, y: np.ndarray, 
                         period: int) -> np.ndarray:
        """使用傅里叶级数拟合季节性"""
        n = len(y)
        seasonal = np.zeros(n)
        
        for k in range(1, self.n_fourier + 1):
            # 傅里叶基函数
            x_k = 2 * np.pi * k * t / period
            
            # 正弦和余弦分量
            sin_k = np.sin(x_k)
            cos_k = np.cos(x_k)
            
            # 最小二乘拟合
            A = np.vstack([sin_k, cos_k]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            
            seasonal += coeffs[0] * sin_k + coeffs[1] * cos_k
        
        return seasonal
    
    def predict(self, n_future: int, X_new: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """预测未来值"""
        # 这里简化实现，实际应用中需要保存更多状态
        return {'lower': None, 'upper': None}
    
    def plot_components(self, result: Dict[str, np.ndarray], 
                        ax=None) -> 'matplotlib.axes.Axes':
        """
        绘制分解组件
        
        参数:
            result: fit()返回的结果
            ax: matplotlib轴
        
        返回:
            matplotlib轴对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("需要matplotlib库")
            return None
        
        if ax is None:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        else:
            axes = ax if isinstance(ax, list) else [ax]
        
        t = np.arange(len(result['trend']))
        
        # 趋势
        axes[0].plot(t, result['trend'], 'b-', label='Trend')
        axes[0].set_title('Trend Component')
        axes[0].legend()
        
        # 季节性
        axes[1].plot(t, result['seasonal'], 'g-', label='Seasonal')
        axes[1].set_title('Seasonal Component')
        axes[1].legend()
        
        # 残差
        axes[2].plot(t, result['residual'], 'r-', label='Residual')
        axes[2].set_title('Residual Component')
        axes[2].legend()
        
        # 原始数据 vs 拟合
        axes[3].plot(t, result['trend'] + result['seasonal'] + result['residual'], 
                     'b-', label='Original')
        axes[3].plot(t, result['y_hat'], 'r--', label='Fitted')
        axes[3].set_title('Original vs Fitted')
        axes[3].legend()
        
        plt.tight_layout()
        return axes[0] if not isinstance(ax, list) else ax


class HarmonicRegression:
    """
    谐波回归 - 用于捕获周期性和季节性
    """
    
    def __init__(self, n_harmonics: int = 5):
        self.n_harmonics = n_harmonics
        self.coeffs = None
    
    def fit(self, t: np.ndarray, y: np.ndarray, period: float) -> np.ndarray:
        """
        拟合谐波回归
        
        参数:
            t: 时间点
            y: 观测值
            period: 周期长度
        
        返回:
            拟合值
        """
        n = len(t)
        A = np.zeros((n, 2 * self.n_harmonics))
        
        for k in range(1, self.n_harmonics + 1):
            x_k = 2 * np.pi * k * t / period
            A[:, 2*k-2] = np.sin(x_k)
            A[:, 2*k-1] = np.cos(x_k)
        
        self.coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        
        return A @ self.coeffs
    
    def predict(self, t: np.ndarray, period: float) -> np.ndarray:
        """预测"""
        n = len(t)
        A = np.zeros((n, 2 * self.n_harmonics))
        
        for k in range(1, self.n_harmonics + 1):
            x_k = 2 * np.pi * k * t / period
            A[:, 2*k-2] = np.sin(x_k)
            A[:, 2*k-1] = np.cos(x_k)
        
        return A @ self.coeffs


if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成带季节性的时间序列
    T = 365
    t = np.arange(T)
    
    # 趋势 + 季节 + 噪声
    trend = 0.01 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 365)
    noise = np.random.randn(T) * 2
    y = trend + seasonal + noise
    
    # 拟合Prophet模型
    prophet = ProphetDecompose()
    result = prophet.fit(y, seasonality_period=365)
    
    print(f"Original shape: {y.shape}")
    print(f"Trend shape: {result['trend'].shape}")
    print(f"Seasonal shape: {result['seasonal'].shape}")
    print(f"Residual std: {np.std(result['residual']):.4f}")
    
    # 计算R²
    ss_res = np.sum((y - result['y_hat']) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"R² score: {r2:.4f}")
