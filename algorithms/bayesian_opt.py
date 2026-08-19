"""
Bayesian Optimization and Stacking Ensemble methods
Advanced optimization and ensemble techniques for math modeling
"""
import numpy as np
from typing import Callable, Dict, Any, List, Tuple
import math


class BayesianOptimization:
    """
    Bayesian Optimization with Gaussian Process surrogate
    
    Efficient global optimization for expensive black-box functions
    Suitable for: 超参数调优、实验设计优化、 costly function optimization
    """
    
    def __init__(self, bounds: List[Tuple], n_init=10, n_iter=30, 
                 alpha=1e-6, nu=1.5, acquisition="ei", verbose=False):
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.n_init = n_init
        self.n_iter = n_iter
        self.alpha = alpha
        self.nu = nu
        self.acquisition = acquisition
        self.verbose = verbose
        
        self.X_train = []
        self.y_train = []
        self.best_x = None
        self.best_y = None
        
    def _covariance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """RBF covariance with Mahalanobis distance"""
        diff = x1 - x2
        dist = np.sqrt(np.sum(diff ** 2))
        return math.exp(-0.5 * dist ** 2)
    
    def _gaussian_process_predict(self, x_test: np.ndarray) -> Tuple[float, float]:
        """GP prediction at test point"""
        if len(self.X_train) < 2:
            return 0.0, 1.0
        
        n = len(self.X_train)
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        
        # Build kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._covariance(X[i], X[j])
        K += self.alpha * np.eye(n)
        
        # Predict at test point
        k_star = np.array([self._covariance(x_test, X[i]) for i in range(n)])
        K_inv = np.linalg.inv(K)
        
        mu = k_star @ K_inv @ y
        var = self._covariance(x_test, x_test) - k_star @ K_inv @ k_star
        var = max(var, 1e-10)
        
        return float(mu), float(var)
    
    def _expected_improvement(self, x_test: np.ndarray) -> float:
        """Expected Improvement acquisition function"""
        mu, sigma = self._gaussian_process_predict(x_test)
        sigma = max(sigma, 1e-8)
        
        if self.best_y is None:
            return 0.0
        
        improvement = mu - self.best_y
        z = improvement / sigma
        ei = improvement * self._phi(z) + sigma * self._phi_std(z)
        return float(ei)
    
    def _phi(self, x: float) -> float:
        """Standard normal CDF approximation"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
    
    def _phi_std(self, x: float) -> float:
        """Standard normal PDF"""
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
    
    def optimize(self, objective_func: Callable, is_maximization: bool = True) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        results = []
        
        # Initial design (Latin Hypercube-like)
        for i in range(self.n_init):
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            y = objective_func(x)
            self.X_train.append(x.tolist())
            self.y_train.append(y)
            
            if self.best_y is None or (is_maximization and y > self.best_y) or \
               (not is_maximization and y < self.best_y):
                self.best_y = y
                self.best_x = x.copy()
        
        # Iterative optimization
        for iteration in range(self.n_iter):
            # Find next point via acquisition function
            best_acq = -np.inf if is_maximization else np.inf
            x_next = None
            
            for _ in range(50):  # Random search for next point
                x_candidate = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                acq = self._expected_improvement(x_candidate)
                
                if is_maximization and acq > best_acq:
                    best_acq = acq
                    x_next = x_candidate
                elif not is_maximization and acq < best_acq:
                    best_acq = acq
                    x_next = x_candidate
            
            if x_next is None:
                x_next = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            
            y_next = objective_func(x_next)
            self.X_train.append(x_next.tolist())
            self.y_train.append(y_next)
            
            if (is_maximization and y_next > self.best_y) or \
               (not is_maximization and y_next < self.best_y):
                self.best_y = y_next
                self.best_x = x_next.copy()
            
            results.append({"iteration": iteration, "x": x_next.tolist(), "y": y_next})
            
            if self.verbose and iteration % 5 == 0:
                print(f"  Iter {iteration}: best_y={self.best_y:.6f}")
        
        return {
            "status": "success",
            "optimal_solution": self.best_x.tolist() if self.best_x is not None else [],
            "optimal_value": float(self.best_y) if self.best_y is not None else 0.0,
            "n_evaluations": len(self.X_train),
            "best_history": results[:10]
        }
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_dims": self.n_dims,
            "n_init": self.n_init,
            "n_iter": self.n_iter,
            "acquisition": self.acquisition
        }


class StackingEnsemble:
    """
    Stacking ensemble: meta-learner combines base model predictions
    
    Suitable for: 提高预测精度、模型融合、回归/分类任务
    """
    
    def __init__(self, base_models=None, meta_model=None):
        self.base_models = base_models or [
            ("linear", None),
            ("rf", None),
            ("svm", None)
        ]
        self.meta_model = meta_model
        self.base_fits = {}
        self.meta_weights = None
        self.n_valid_models = 0
        self.meta_bias = 0.0
        
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        """Fit base models and learn meta-weights"""
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        
        # Define base models
        model_specs = {
            "linear": LinearRegression(),
            "rf": RandomForestRegressor(n_estimators=50, random_state=42),
            "svm": SVR(kernel="rbf"),
            "gb": None  # Will use gradient boosting if available
        }
        
        # Fit base models
        oof_pred = np.zeros((len(X_train), len(model_specs)))
        n_valid = 0
        
        for i, (name, _) in enumerate(model_specs.items()):
            try:
                model = model_specs[name].fit(X_train, y_train)
                self.base_fits[name] = model
                preds = model.predict(X_train)
                if len(preds) == len(X_train):
                    oof_pred[:, i] = preds
                    n_valid += 1
            except Exception:
                pass
        
        # Train meta-learner only on valid columns
        valid_cols = np.all(np.isfinite(oof_pred[:, :n_valid]), axis=1) if n_valid > 0 else np.ones(len(X_train), dtype=bool)
        if n_valid > 0 and valid_cols.sum() > 10:
            from sklearn.linear_model import LinearRegression
            meta = LinearRegression()
            meta.fit(oof_pred[valid_cols, :n_valid], y_train[valid_cols])
            self.meta_weights = meta.coef_
            self.meta_bias = meta.intercept_
            self.n_valid_models = n_valid
        else:
            self.n_valid_models = 0
        
        # Train meta-learner (simple linear regression on OOF predictions)
        valid_cols = np.all(np.isfinite(oof_pred), axis=1)
        if valid_cols.sum() > 10:
            meta = LinearRegression()
            meta.fit(oof_pred[valid_cols], y_train[valid_cols])
            self.meta_weights = meta.coef_
            self.meta_bias = meta.intercept_
        
        return {
            "status": "success",
            "base_models": list(self.base_fits.keys()),
            "meta_weights": self.meta_weights.tolist() if self.meta_weights is not None else None
        }
    
    def predict(self, X) -> np.ndarray:
        """Stacked prediction"""
        if self.n_valid_models == 0 or not self.base_fits:
            return np.mean(X, axis=1)
        
        base_preds = []
        for name, model in list(self.base_fits.items())[:self.n_valid_models]:
            try:
                base_preds.append(model.predict(X))
            except Exception:
                pass
        
        if not base_preds:
            return np.mean(X, axis=1)
        
        base_matrix = np.column_stack(base_preds)
        
        if self.meta_weights is not None and len(self.meta_weights) == base_matrix.shape[1]:
            return base_matrix @ self.meta_weights + self.meta_bias
        else:
            return base_matrix.mean(axis=1)
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate stacked model"""
        y_pred = self.predict(X_test)
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / (np.sum((y_test - np.mean(y_test)) ** 2) + 1e-10)
        return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "base_models": list(self.base_fits.keys()),
            "n_base_models": self.n_valid_models if hasattr(self, "n_valid_models") else len(self.base_fits)
        }
