"""
Prophet-like time series forecasting model
Simplified additive model for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List


class ProphetModel:
    """
    Simplified Prophet model for time series forecasting
    
    Additive model: y(t) = g(t) + s(t) + h(t) + e(t)
    - g(t): trend component
    - s(t): seasonality component  
    - h(t): holiday effects
    - e(t): error term
    """
    
    def __init__(self, growth: str = "linear", 
                 changepoint_range: float = 0.8,
                 seasonality_mode: str = "additive"):
        self.growth = growth
        self.changepoint_range = changepoint_range
        self.seasonality_mode = seasonality_mode
        
        self.k = 0
        self.m = 0
        self.delta = None
        self.theta = {}
        self.seasonality_periods = []
        self.raw_X = None
        
    def _make_changepoints(self, n_changepoints: int, t: np.ndarray):
        n_points = int(len(t) * self.changepoint_range)
        changepoint_vals = np.linspace(t[0], t[-1], n_points + 1)[1:]
        return changepoint_vals[:n_changepoints]
    
    def _fourier_series(self, t: np.ndarray, period: int, 
                        fourier_order: int = 10) -> np.ndarray:
        result = np.zeros((len(t), 2 * fourier_order))
        for i in range(fourier_order):
            j = 2 * i
            result[:, j] = np.sin(2 * np.pi * (i + 1) * t / period)
            result[:, j + 1] = np.cos(2 * np.pi * (i + 1) * t / period)
        return result
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            n_changepoints: int = 25,
            seasonal_fourier_order: int = 10) -> Dict[str, Any]:
        X = X.flatten()
        y = y.flatten()
        self.raw_X = X.copy()
        
        X_scaled = (X - X.min()) / (X.max() - X.min() + 1e-8)
        
        coeffs = np.polyfit(X_scaled, y, 1)
        self.k = coeffs[0]
        self.m = coeffs[1]
        
        n_changepoints = min(n_changepoints, len(X) // 4)
        self.changepoints = self._make_changepoints(n_changepoints, X_scaled)
        
        self.seasonality_periods = [7, 365]
        self.theta = {}
        
        for period in self.seasonality_periods:
            if len(X) >= period:
                S = self._fourier_series(X_scaled, period, seasonal_fourier_order)
                beta, _, _, _ = np.linalg.lstsq(S, y - (self.k * X_scaled + self.m), rcond=None)
                self.theta[period] = beta
        
        self.fitted_values = self.k * X_scaled + self.m
        for period, beta in self.theta.items():
            S = self._fourier_series(X_scaled, period, seasonal_fourier_order)
            self.fitted_values += S @ beta
        
        self.residuals = y - self.fitted_values
        self.mse = np.mean(self.residuals ** 2)
        
        return {
            "status": "success",
            "mse": float(self.mse),
            "rmse": float(np.sqrt(self.mse)),
            "n_changepoints": n_changepoints,
            "seasonal_periods": self.seasonality_periods
        }
    
    def predict(self, X: np.ndarray, steps: int = None) -> np.ndarray:
        if steps is not None and X is None:
            X = np.arange(len(self.raw_X)) + steps
        
        X = X.flatten()
        X_scaled = (X - self.raw_X.min()) / (self.raw_X.max() - self.raw_X.min() + 1e-8)
        
        y_pred = self.k * X_scaled + self.m
        for period, beta in self.theta.items():
            S = self._fourier_series(X_scaled, period, len(beta) // 2)
            y_pred += S @ beta
        
        return y_pred
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "growth": self.growth,
            "k": float(self.k),
            "m": float(self.m),
            "theta_shapes": {p: v.shape for p, v in self.theta.items()}
        }


def ProphetDecompose(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Prophet-style decomposition
    """
    model = ProphetModel()
    result = model.fit(X, y)
    return {
        "trend": model.fitted_values,
        "seasonal": model.fitted_values - (model.k * ((X - X.min()) / (X.max() - X.min() + 1e-8)) + model.m),
        "residual": model.residuals,
        "metrics": {"rmse": result["rmse"]}
    }


def HarmonicRegression(X: np.ndarray, y: np.ndarray, n_harmonics: int = 5) -> Dict[str, Any]:
    """
    Harmonic regression for periodic data
    """
    X = X.flatten()
    X_scaled = (X - X.min()) / (X.max() - X.min() + 1e-8)
    
    n = len(X_scaled)
    A = np.column_stack([
        np.ones(n),
        np.sin(2 * np.pi * X_scaled),
        np.cos(2 * np.pi * X_scaled),
    ])
    
    for i in range(2, n_harmonics):
        A = np.column_stack([A, 
                            np.sin(2 * np.pi * (i + 1) * X_scaled),
                            np.cos(2 * np.pi * (i + 1) * X_scaled)])
    
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ beta
    residuals = y - y_pred
    
    return {
        "coefficients": beta,
        "predictions": y_pred,
        "residuals": residuals,
        "rmse": float(np.sqrt(np.mean(residuals ** 2)))
    }
