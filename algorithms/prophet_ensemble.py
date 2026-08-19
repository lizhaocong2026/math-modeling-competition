"""
Enhanced Prophet Model with decomposition and AutoML Pipeline
Advanced time series forecasting and automated model selection
"""
import numpy as np
from typing import Dict, Any, List, Optional
import math


class ProphetModel:
    """
    Enhanced Prophet-style decomposition model for time series forecasting
    
    Supports: trend (linear/logistic), seasonality (Fourier), changepoints
    Suitable for: 碳排放预测、负荷预测、销售预测等有趋势和季节性的数据
    """
    
    def __init__(self, n_changepoints=20, yearly_period=365.25, 
                 weekly_period=7, daily_period=1, Fourier_order=10):
        self.n_changepoints = n_changepoints
        self.yearly_period = yearly_period
        self.weekly_period = weekly_period
        self.daily_period = daily_period
        self.Fourier_order = Fourier_order
        
        # Trend parameters
        self.k = 0.0  # initial rate
        self.m = 0.0  # initial offset
        self.delta = None
        self.changepoints = []
        
        # Seasonal parameters
        self.yearly_coeffs = None
        self.weekly_coeffs = None
        self.daily_coeffs = None
        
        self.history = []
    
    def _fourier_series(self, t: np.ndarray, period: int) -> np.ndarray:
        """Compute Fourier components for a given period"""
        n = self.Fourier_order
        result = np.zeros((len(t), 2 * n))
        for i in range(n):
            result[:, 2*i] = np.sin(2 * math.pi * (i + 1) * t / period)
            result[:, 2*i + 1] = np.cos(2 * math.pi * (i + 1) * t / period)
        return result
    
    def _linear_trend(self, t: np.ndarray, k: float, m: float, delta: np.ndarray = None,
                       t_shifts: np.ndarray = None) -> np.ndarray:
        """Linear trend with changepoints"""
        if delta is None:
            return k * t + m
        # Apply changepoint adjustments
        trend = k * t + m
        if t_shifts is not None and len(t_shifts) > 0:
            for i, tp in enumerate(t_shifts):
                if tp < t[-1]:
                    trend += delta[i] * np.maximum(0, t - tp)
        return trend
    
    def fit(self, y: np.ndarray, dates: Optional[np.ndarray] = None,
            seasonality_mode: str = "additive") -> Dict[str, Any]:
        """Fit Prophet model to time series"""
        n = len(y)
        if dates is None:
            dates = np.arange(n)
        
        t = (dates - dates[0]) / max(1, n)  # normalized time [0, 1]
        
        # Fit trend using least squares
        T = n  # number of observations
        # Simple linear trend fit
        A = np.column_stack([t, np.ones(n)])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self.k = float(coeffs[0])
        self.m = float(coeffs[1])
        
        # Setup changepoints
        self.changepoints = np.linspace(0.05, 0.95, self.n_changepoints)
        self.delta = np.zeros(self.n_changepoints)
        
        # Fit seasonal components
        yearly_hat = self._fourier_series(t * self.yearly_period, self.yearly_period)
        weekly_hat = self._fourier_series(t * self.weekly_period, self.weekly_period)
        
        # Seasonal coefficients via least squares
        X_seasonal = np.column_stack([yearly_hat, weekly_hat])
        residuals = y - (self.k * t + self.m)
        seasonal_coeffs, _, _, _ = np.linalg.lstsq(X_seasonal, residuals, rcond=None)
        
        self.yearly_coeffs = seasonal_coeffs[:2 * self.Fourier_order]
        self.weekly_coeffs = seasonal_coeffs[2 * self.Fourier_order:]
        
        # Compute fit history
        trend = self._linear_trend(t, self.k, self.m, self.delta, 
                                    self.changepoints if self.n_changepoints > 0 else None)
        yearly_seasonal = X_seasonal @ seasonal_coeffs
        self.history = {"trend": trend, "seasonal": yearly_seasonal, 
                       "residual": y - trend - yearly_seasonal}
        
        rss = np.mean((y - trend - yearly_seasonal) ** 2)
        return {"status": "success", "final_rmse": float(np.sqrt(rss)), "n_changepoints": self.n_changepoints}
    
    def predict(self, future_dates: np.ndarray) -> Dict[str, np.ndarray]:
        """Forecast into the future"""
        t = (future_dates - future_dates[0]) / max(1, len(future_dates))
        
        trend = self._linear_trend(t, self.k, self.m, self.delta, None)
        yearly_hat = self._fourier_series(t * self.yearly_period, self.yearly_period)
        weekly_hat = self._fourier_series(t * self.weekly_period, self.weekly_period)
        X_seasonal = np.column_stack([yearly_hat, weekly_hat])
        
        seasonal = X_seasonal @ np.concatenate([self.yearly_coeffs, self.weekly_coeffs])
        forecast = trend + seasonal
        
        return {"trend": trend, "seasonal": seasonal, "forecast": forecast}
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_changepoints": self.n_changepoints,
            "Fourier_order": self.Fourier_order,
            "k": float(self.k),
            "m": float(self.m)
        }


class ProphetDecompose:
    """
    Prophet decomposition with STL-like separation
    
    Decomposes time series into trend, seasonal, and residual components
    """
    
    def __init__(self, period=24, n_harmonics=10):
        self.period = period
        self.n_harmonics = n_harmonics
    
    def decompose(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose time series using FFT-based approach"""
        n = len(y)
        # Detrend using moving average
        window = min(self.period, n // 3)
        trend = np.convolve(y, np.ones(window) / window, mode='same')
        trend = np.convolve(trend, np.ones(window) // 2, mode='same')
        
        # Detrended series
        detrended = y - trend
        
        # Extract seasonal component via FFT
        fft_coeffs = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(n, d=1.0)
        
        # Keep only the seasonal frequency components
        seasonal_freq_mask = np.abs(freqs - 1.0 / self.period) < 0.01
        seasonal_freq_mask[0] = True  # Keep DC component
        
        seasonal_fft = fft_coeffs.copy()
        seasonal_fft[~seasonal_freq_mask] = 0
        seasonal = np.fft.irfft(seasonal_fft, n=n)
        
        # Residual
        residual = y - trend - seasonal
        
        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "original": y
        }
    
    def fit_predict(self, y: np.ndarray, steps: int = 24) -> Dict[str, Any]:
        """Decompose and forecast"""
        decomp = self.decompose(y)
        
        # Forecast trend (linear extrapolation)
        trend_last = decomp["trend"][-1]
        trend_slope = np.mean(np.diff(decomp["trend"][-self.period:]))
        future_trend = np.array([trend_last + trend_slope * (i + 1) for i in range(steps)])
        
        # Forecast seasonal (repeat last period)
        future_seasonal = np.tile(decomp["seasonal"][-self.period:], (steps // self.period + 1))[:steps]
        
        forecast = future_trend + future_seasonal
        
        return {
            "forecast": forecast,
            "decomposition": decomp,
            "mape": float(np.mean(np.abs(decomp["residual"] / (np.abs(decomp["original"]) + 1e-8)))) * 100
        }
    
    def get_params(self) -> Dict[str, Any]:
        return {"period": self.period, "n_harmonics": self.n_harmonics}


class AutoMLPipeline:
    """
    Automated machine learning pipeline for regression tasks
    
    Automatically selects and tunes the best model from multiple candidates
    Suitable for: 快速建模、基准模型选择、超参数自动搜索
    """
    
    def __init__(self, max_models=5, cv_folds=3):
        self.max_models = max_models
        self.cv_folds = cv_folds
        self.model_scores = {}
        self.best_model_name = None
        self.best_model = None
    
    def _evaluate_model(self, model, X, y, fold_indices):
        """Cross-validation evaluation"""
        n = len(y)
        fold_size = n // self.cv_folds
        scores = []
        
        for fold in range(self.cv_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.cv_folds - 1 else n
            
            val_idx = list(range(val_start, val_end))
            train_idx = list(range(0, val_start)) + list(range(val_end, n))
            
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                mse = np.mean((y_val - pred) ** 2)
                scores.append(mse)
            except Exception:
                scores.append(float("inf"))
        
        return float(np.mean(scores)) if scores else float("inf")
    
    def fit(self, X, y) -> Dict[str, Any]:
        """Train and evaluate multiple models, select best"""
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Define candidate models
        models = [
            ("LinearRegression", LinearRegression()),
            ("Ridge", Ridge(alpha=1.0)),
            ("RandomForest", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ("SVR", SVR(kernel="rbf", C=1.0)),
            ("KNN", KNeighborsRegressor(n_neighbors=5)),
        ]
        
        # Create CV folds
        n = len(y)
        indices = np.arange(n)
        fold_size = n // self.cv_folds
        fold_indices = []
        for fold in range(self.cv_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.cv_folds - 1 else n
            fold_indices.append((list(range(0, val_start)) + list(range(val_end, n)),
                                list(range(val_start, val_end))))
        
        # Evaluate each model
        self.model_scores = {}
        for name, model in models[:self.max_models]:
            score = self._evaluate_model(model, X, y, fold_indices)
            self.model_scores[name] = score
            print(f"  {name}: CV MSE = {score:.6f}")
        
        # Select best model and train on full data
        self.best_model_name = min(self.model_scores, key=self.model_scores.get)
        for name, model in models:
            if name == self.best_model_name:
                self.best_model = model
                break
        self.best_model.fit(X, y)
        
        return {
            "status": "success",
            "best_model": self.best_model_name,
            "model_scores": {k: float(v) for k, v in self.model_scores.items()},
            "cv_folds": self.cv_folds
        }
    
    def predict(self, X) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.best_model.predict(X)
    
    def evaluate(self, X, y) -> Dict[str, float]:
        pred = self.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        r2 = float(1 - np.sum((y - pred) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-10))
        return {"MAE": mae, "RMSE": rmse, "R2": r2}
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "best_model": self.best_model_name,
            "model_scores": self.model_scores,
            "max_models": self.max_models
        }
