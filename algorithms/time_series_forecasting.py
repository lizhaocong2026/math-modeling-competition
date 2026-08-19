# Time Series Forecasting Algorithms
import numpy as np
from typing import List, Tuple, Dict, Any

class TimeSeriesForecasting:
    """时间序列预测算法集合"""
    
    def __init__(self):
        self.results = {}
    
    def moving_average(self, data, window=3):
        """移动平均法"""
        result = np.full(len(data), np.nan, dtype=float)
        for i in range(window-1, len(data)):
            result[i] = np.mean(data[i-window+1:i+1])
        return result
    
    def exponential_smoothing(self, data, alpha=0.3):
        """指数平滑法"""
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    def linear_trend_forecast(self, data, steps=5):
        """线性趋势外推"""
        n = len(data)
        x = np.arange(n)
        coeffs = np.polyfit(x, data, 1)
        future_x = np.arange(n, n + steps)
        return np.polyval(coeffs, future_x)
    
    def polynomial_trend_forecast(self, data, degree=2, steps=5):
        """多项式趋势外推"""
        n = len(data)
        x = np.arange(n)
        coeffs = np.polyfit(x, data, degree)
        future_x = np.arange(n, n + steps)
        return np.polyval(coeffs, future_x)
    
    def naive_forecast(self, data, steps=1):
        """朴素预测（最近值）"""
        return np.full(steps, data[-1])
    
    def seasonal_naive(self, data, seasonal_period=4, steps=5):
        """季节性朴素预测"""
        result = []
        n = len(data)
        for i in range(steps):
            idx = n - (steps - i) % seasonal_period
            if idx < 0:
                idx = 0
            result.append(data[idx] if idx < n else data[-1])
        return np.array(result)
    
    def compute_mape(self, actual, predicted):
        """计算MAPE误差"""
        mask = actual != 0
        if not np.any(mask):
            return float("inf")
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def compute_rmse(self, actual, predicted):
        """计算RMSE误差"""
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    def compare_methods(self, data, test_ratio=0.2):
        """比较多种预测方法"""
        split_idx = int(len(data) * (1 - test_ratio))
        train = data[:split_idx]
        test = data[split_idx:]
        
        results = {}
        ma_pred = self.moving_average(data, window=3)[-len(test):]
        results["Moving Average"] = {"mape": self.compute_mape(test, ma_pred), "rmse": self.compute_rmse(test, ma_pred)}
        
        es_pred = self.exponential_smoothing(data, alpha=0.3)[-len(test):]
        results["Exponential Smoothing"] = {"mape": self.compute_mape(test, es_pred), "rmse": self.compute_rmse(test, es_pred)}
        
        lt_pred = self.linear_trend_forecast(train, steps=len(test))
        results["Linear Trend"] = {"mape": self.compute_mape(test, lt_pred), "rmse": self.compute_rmse(test, lt_pred)}
        
        naive_pred = self.naive_forecast(train, steps=len(test))
        results["Naive"] = {"mape": self.compute_mape(test, naive_pred), "rmse": self.compute_rmse(test, naive_pred)}
        
        return results
