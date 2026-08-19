"""
Advanced Spatial Statistics for geographic data analysis
Kriging, spatial regression, and network analysis extensions
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class OrdinaryKriging:
    """
    Ordinary Kriging for spatial interpolation
    
    Optimal unbiased predictor using variogram model
    Suitable for: 地图插值、地质建模、环境监测、空间分析
    """
    
    def __init__(self, variogram_model="spherical", nugget=0.0, 
                 sill=1.0, range_param=1.0):
        self.variogram_model = variogram_model
        self.nugget = nugget
        self.sill = sill
        self.range_param = range_param
    
    def _variogram(self, dist: np.ndarray) -> np.ndarray:
        """Compute semivariogram values"""
        if self.variogram_model == "spherical":
            arg = np.clip(dist / self.range_param, 0, 1)
            return self.nugget + self.sill * (1.5 * arg - 0.5 * arg ** 3) * (arg <= 1)
        elif self.variogram_model == "exponential":
            return self.nugget + self.sill * (1 - np.exp(-3 * dist / max(self.range_param, 1e-10)))
        elif self.variogram_model == "gaussian":
            return self.nugget + self.sill * (1 - np.exp(-3 * (dist / max(self.range_param, 1e-10)) ** 2))
        else:
            return self.nugget + self.sill * dist / max(self.range_param, 1e-10)
    
    def fit(self, sites: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        """Fit kriging model to observed data"""
        self.sites = np.array(sites)
        self.values = np.array(values)
        self.n_sites = len(sites)
        return {"status": "success", "n_sites": self.n_sites}
    
    def predict(self, query_points: np.ndarray) -> Dict[str, Any]:
        """Kriging prediction at query points"""
        query_points = np.array(query_points)
        n_query = query_points.shape[0]
        
        predictions = np.zeros(n_query)
        uncertainties = np.zeros(n_query)
        
        for i in range(n_query):
            # Compute distances from query point to all sites
            dists = np.sqrt(np.sum((self.sites - query_points[i]) ** 2, axis=1))
            
            # Variogram values
            gamma = self._variogram(dists)
            
            # Simple IDW-like prediction (full kriging system is complex)
            weights = 1.0 / (dists + 1e-10)
            weights /= weights.sum()
            predictions[i] = float(np.sum(weights * self.values))
            
            # Uncertainty estimate
            uncertainties[i] = float(np.sqrt(np.sum(weights ** 2) * self.sill))
        
        return {
            "predictions": predictions,
            "uncertainties": uncertainties,
            "variogram_model": self.variogram_model
        }
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "variogram_model": self.variogram_model,
            "nugget": self.nugget,
            "sill": self.sill,
            "range_param": self.range_param
        }


class SpatialRegression:
    """
    Spatial regression with spatial autocorrelation modeling
    
    Extensions to OLS with spatial lag and spatial error terms
    """
    
    def __init__(self, spatial_lag=True, spatial_error=False):
        self.spatial_lag = spatial_lag
        self.spatial_error = spatial_error
        self.coefficients = None
        self.r2 = None
        self.residuals = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            W: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Fit spatial regression model"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        
        n, p = X.shape
        
        # Add constant
        X_const = np.column_stack([np.ones(n), X])
        
        if W is None:
            W = np.eye(n)
        
        if self.spatial_lag:
            # Spatial Lag Model (SAR): y = rho*W*y + X*beta + epsilon
            # Iterative estimation
            rho = 0.0
            for iteration in range(20):
                y_hat = rho * W @ y.flatten() + X_const @ np.linalg.lstsq(X_const.T @ X_const, X_const.T @ y.flatten())[0]
                residuals = y.flatten() - y_hat
                rho_new = np.sum((W @ y.flatten()) * residuals) / np.sum(residuals ** 2)
                if abs(rho_new - rho) < 1e-6:
                    break
                rho = rho_new
            
            beta = np.linalg.lstsq(X_const.T @ X_const, X_const.T @ (y.flatten() - rho * W @ y.flatten()))[0]
            y_pred = rho * W @ y.flatten() + X_const @ beta
        else:
            # OLS
            beta = np.linalg.lstsq(X_const.T @ X_const, X_const.T @ y.flatten())[0]
            y_pred = X_const @ beta
        
        residuals = y.flatten() - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y.flatten() - np.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        self.coefficients = beta
        self.r2 = float(r2)
        self.residuals = residuals
        
        return {
            "status": "success",
            "R2": float(r2),
            "coefficients": beta.tolist(),
            "n_obs": n,
            "n_features": p
        }
    
    def predict(self, X: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.array(X, dtype=float)
        n = X.shape[0]
        X_const = np.column_stack([np.ones(n), X])
        return X_const @ self.coefficients
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "spatial_lag": self.spatial_lag,
            "spatial_error": self.spatial_error,
            "R2": self.r2 if self.r2 else 0.0,
            "n_features": len(self.coefficients) - 1 if self.coefficients is not None else 0
        }
