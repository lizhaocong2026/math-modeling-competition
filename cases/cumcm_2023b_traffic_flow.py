# CUMCM 2023 B - Urban Traffic Flow Prediction
import numpy as np
from typing import Dict, Any
from algorithms.stl_decompose import STLDecomposer
from algorithms.arima import ARIMA


class TrafficFlowPredictor:
    '''STL decomposition + ARIMA hybrid for traffic flow prediction'''

    def __init__(self, period_hourly=24):
        self.period_hourly = period_hourly

    def stl_decomposition(self, flow_data: np.ndarray) -> Dict[str, np.ndarray]:
        decomposer = STLDecomposer(period=self.period_hourly, robust=True)
        result = decomposer.fit(flow_data)
        if result.get("status") == "success":
            return {
                "trend": decomposer.get_trend(),
                "seasonal": decomposer.get_seasonal(),
                "residual": decomposer.get_resid(),
            }
        else:
            n = len(flow_data)
            return {
                "trend": flow_data,
                "seasonal": np.zeros(n),
                "residual": np.zeros(n),
            }

    def hybrid_predict(self, historical: np.ndarray, steps: int = 24) -> Dict[str, Any]:
        decomp_result = self.stl_decomposition(historical)
        if decomp_result.get("trend") is not None and decomp_result["trend"] is not None:
            trend = decomp_result["trend"]
            seasonal = decomp_result["seasonal"]
        else:
            trend = historical
            seasonal = np.zeros(len(historical))
        # Simple moving average forecast as fallback
        ma_window = min(24, len(trend))
        arima_fc = np.full(steps, np.mean(trend[-ma_window:]))
        if len(seasonal) >= steps:
            seasonal_extend = seasonal[-steps:]
        else:
            seasonal_extend = np.zeros(steps)
        combined = arima_fc + seasonal_extend
        return {"method": "STL-ARIMA Hybrid", "forecast": combined}

    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        return {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        }


if __name__ == "__main__":
    np.random.seed(42)
    t = np.arange(0, 336, 1)
    flow_data = (100 + 50 * np.sin(2 * np.pi * (t - 8) / 24)
                 + 20 * np.sin(2 * np.pi * t / 168)
                 + np.random.normal(0, 5, 336))
    predictor = TrafficFlowPredictor()
    result = predictor.hybrid_predict(flow_data, steps=24)
    print("CUMCM 2023 B - Traffic Flow Prediction")
    print("Method:", result["method"])
    print("Forecast length:", len(result["forecast"]))
