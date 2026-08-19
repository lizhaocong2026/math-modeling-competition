# SARIMA - Seasonal ARIMA
import numpy as np
from typing import Dict, Any
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSModels = True
except ImportError:
    HAS_STATSModels = False
    SARIMAX = None

class SARIMAModel:
    """SARIMA季节模型 - 周期性强数据预测"""
    
    def __init__(self, order: tuple = (1,1,1), seasonal_order: tuple = (1,1,1,12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        
    def fit(self, series: np.ndarray) -> Dict[str, Any]:
        if not HAS_STATSModels:
            return {'status': 'error', 'message': '请安装statsmodels: pip install statsmodels'}
        self.model = SARIMAX(series, order=self.order, seasonal_order=self.seasonal_order)
        self.result = self.model.fit(disp=False)
        return {'status': 'success', 'message': 'SARIMA模型训练完成'}
        
    def predict(self, steps: int = 10) -> np.ndarray:
        forecast = self.result.get_forecast(steps=steps)
        return forecast.predicted_mean
        
    def summary(self) -> str:
        return self.result.summary().as_text()
        
    def get_params(self) -> Dict[str, Any]:
        return {'order': self.order, 'seasonal_order': self.seasonal_order}
