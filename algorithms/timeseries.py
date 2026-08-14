"""
时间序列分解
将时间序列分解为趋势、季节和残差成分
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple
from scipy.signal import savgol_filter


class TimeSeriesDecomposition:
    """时间序列分解"""
    
    def __init__(self, period: Optional[int] = None):
        """
        初始化
        
        参数:
            period: 季节周期，None表示自动检测
        """
        self.period = period
        self.trend = None
        self.seasonal = None
        self.residual = None
        self.original = None
        
    def fit(self, data: np.ndarray) -> 'TimeSeriesDecomposition':
        """
        拟合分解模型
        
        参数:
            data: 时间序列数据
            
        返回:
            拟合后的实例
        """
        self.original = np.asarray(data, dtype=float)
        n = len(self.original)
        
        # 自动检测周期
        if self.period is None:
            self.period = self._detect_period()
        
        # 提取趋势（移动平均）
        self.trend = self._extract_trend(n)
        
        # 提取季节成分
        self.seasonal = self._extract_seasonal(n)
        
        # 残差
        self.residual = self.original - self.trend - self.seasonal
        
        return self
    
    def _detect_period(self) -> int:
        """自动检测周期"""
        data = self.original
        n = len(data)
        
        # 使用自相关检测周期
        data_centered = data - np.mean(data)
        autocorr = np.correlate(data_centered, data_centered, mode='full')
        autocorr = autocorr[n:]  # 取正半部分
        
        # 找到第一个显著峰值
        max_lag = min(n // 2, 50)
        for lag in range(2, max_lag):
            if autocorr[lag] > autocorr[lag-1] and autocorr[lag] > autocorr[lag+1]:
                if autocorr[lag] > 0.5 * autocorr[0]:
                    return lag
        
        return min(12, n // 3)  # 默认值
    
    def _extract_trend(self, n: int) -> np.ndarray:
        """提取趋势成分"""
        period = self.period
        
        # 中心化移动平均
        if period % 2 == 0:
            # 偶数周期：两次移动平均
            trend = np.convolve(self.original, np.ones(period)/period, mode='same')
            trend = np.convolve(trend, np.ones(2)/2, mode='same')
        else:
            # 奇数周期：单次移动平均
            trend = np.convolve(self.original, np.ones(period)/period, mode='same')
        
        # 处理边界
        trend[:period//2] = trend[period//2]
        trend[-period//2:] = trend[-period//2-1]
        
        return trend
    
    def _extract_seasonal(self, n: int) -> np.ndarray:
        """提取季节成分"""
        seasonal = np.zeros(n)
        
        for i in range(n):
            # 找到同季节的平均残差
            seasonal_indices = [j for j in range(n) if j % self.period == i % self.period]
            if seasonal_indices:
                seasonal[i] = np.mean(self.original[seasonal_indices] - self.trend[seasonal_indices])
        
        # 归一化
        seasonal = seasonal - np.mean(seasonal)
        
        return seasonal
    
    def decompose(self) -> Dict[str, Any]:
        """
        分解时间序列
        
        返回:
            包含各成分的字典
        """
        if self.original is None:
            raise RuntimeError("请先调用fit()方法")
        
        return {
            "original": self.original.tolist(),
            "trend": self.trend.tolist(),
            "seasonal": self.seasonal.tolist(),
            "residual": self.residual.tolist(),
            "period": self.period,
            "n_points": len(self.original)
        }
    
    def forecast(self, steps: int = 5) -> Dict[str, Any]:
        """
        基于分解模型进行预测
        
        参数:
            steps: 预测步数
            
        返回:
            预测结果
        """
        if self.original is None:
            raise RuntimeError("请先调用fit()方法")
        
        n = len(self.original)
        
        # 趋势外推（线性）
        x = np.arange(n)
        slope = (self.trend[-1] - self.trend[0]) / (n - 1) if n > 1 else 0
        trend_forecast = np.array([self.trend[-1] + slope * (i + 1) for i in range(steps)])
        
        # 季节成分循环
        seasonal_forecast = np.array([
            self.seasonal[n - steps + i] if n - steps + i >= 0 else 
            self.seasonal[i % self.period]
            for i in range(steps)
        ])
        
        # 预测值
        forecast = trend_forecast + seasonal_forecast
        
        return {
            "forecast": forecast.tolist(),
            "trend_forecast": trend_forecast.tolist(),
            "seasonal_forecast": seasonal_forecast.tolist(),
            "steps": steps
        }
