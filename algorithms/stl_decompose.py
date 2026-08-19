# STL Decomposition - Seasonal-Trend decomposition
import numpy as np
from typing import Dict, Any
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STATSModels = True
except ImportError:
    HAS_STATSModels = False
    STL = None

class STLDecomposer:
    """STL时间序列分解 - 趋势/季节/残差分离"""
    
    def __init__(self, period: int = 12, robust: bool = False):
        self.period = period
        self.robust = robust
        self.result = None
        
    def fit(self, series: np.ndarray) -> Dict[str, Any]:
        if not HAS_STATSModels:
            return {'status': 'error', 'message': '请安装statsmodels: pip install statsmodels'}
        if len(series) < self.period * 2:
            return {'status': 'error', 'message': '数据长度不足，需要至少2个周期'}
        self.result = STL(series, period=self.period, robust=self.robust).fit()
        return {'status': 'success', 'message': 'STL分解完成'}
        
    def get_trend(self) -> np.ndarray:
        return self.result.trend
        
    def get_seasonal(self) -> np.ndarray:
        return self.result.seasonal
        
    def get_resid(self) -> np.ndarray:
        return self.result.resid
        
    def summary(self) -> Dict[str, float]:
        return {
            'trend_var': float(np.var(self.result.trend)),
            'seasonal_var': float(np.var(self.result.seasonal)),
            'resid_var': float(np.var(self.result.resid))
        }

