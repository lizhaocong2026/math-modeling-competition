"""
AutoML Pipeline for automated model selection and hyperparameter tuning
Simplified AutoML for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class AutoMLPipeline:
    """
    Automated Machine Learning pipeline
    
    Suitable for: 模型自动选择、超参数调优、基准模型对比
    """
    
    def __init__(self, models: List[str] = None, max_time: int = 300, n_folds: int = 5):
        self.max_time = max_time
        self.n_folds = n_folds
        self.selected_models = models or ["linear_regression", "random_forest", "svm"]
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = None
        self.best_model_name = None
        
    def _linear_regression(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
        model = LinearRegression()
        score = np.mean(cross_val_score(model, X, y, cv=self.n_folds, scoring='r2'))
        return model, score
    
    def _random_forest_reg(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        score = np.mean(cross_val_score(model, X, y, cv=self.n_folds, scoring='r2'))
        return model, score
    
    def _svm_reg(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        score = np.mean(cross_val_score(model, X, y, cv=self.n_folds, scoring='r2'))
        return model, score
    
    def _random_forest_cls(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        score = np.mean(cross_val_score(model, X, y, cv=self.n_folds, scoring='accuracy'))
        return model, score
    
    def _svm_cls(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
        model = SVC(kernel='rbf', C=1.0, probability=True)
        score = np.mean(cross_val_score(model, X, y, cv=self.n_folds, scoring='accuracy'))
        return model, score
    
    def _logistic_regression(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
        model = LogisticRegression(max_iter=1000)
        score = np.mean(cross_val_score(model, X, y, cv=self.n_folds, scoring='accuracy'))
        return model, score
    
    def evaluate_all(self, X: np.ndarray, y: np.ndarray, cv: int = None,
                     problem_type: str = "auto") -> Dict[str, Any]:
        if cv is not None:
            self.n_folds = cv
        if problem_type == "auto":
            problem_type = "classification" if len(np.unique(y)) < 10 else "regression"
        
        model_mapping = {
            "linear_regression": self._linear_regression,
            "random_forest": self._random_forest_reg if problem_type == "regression" else self._random_forest_cls,
            "svm": self._svm_reg if problem_type == "regression" else self._svm_cls,
            "logistic_regression": self._logistic_regression
        }
        
        # Store results with model names as keys (nested structure for test compatibility)
        for name in self.selected_models:
            if name in model_mapping:
                try:
                    model, score = model_mapping[name](X, y)
                    self.models[name] = model
                    self.results[name] = {"mean_r2": float(score), "model": model}
                except Exception as e:
                    self.results[name] = {"mean_r2": -np.inf, "error": str(e)}
        
        # Select best
        valid_results = {k: v["mean_r2"] for k, v in self.results.items() if v["mean_r2"] != -np.inf}
        if valid_results:
            self.best_model_name = max(valid_results, key=valid_results.get)
            self.best_model = self.models[self.best_model_name]
            self.best_score = valid_results[self.best_model_name]
        
        return {
            "status": "success",
            "problem_type": problem_type,
            **self.results,
            "best_model": self.best_model_name,
            "best_score": float(self.best_score) if self.best_score else None
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            problem_type: str = "auto") -> Dict[str, Any]:
        return self.evaluate_all(X, y, problem_type=problem_type)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.best_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(X)
        return self.predict(X)
    
    def get_best_model(self) -> str:
        return self.best_model_name
    
    def get_summary(self) -> str:
        lines = ["AutoML Pipeline Summary", "=" * 30]
        for name, result in self.results.items():
            lines.append(f"{name}: score={result.get('mean_r2', 'N/A')}")
        lines.append(f"Best model: {self.best_model_name}")
        return "\\n".join(lines)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "selected_models": self.selected_models,
            "best_model": self.best_model_name,
            "best_score": float(self.best_score) if self.best_score else None
        }


def auto_select_models(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       problem_type: str = "auto") -> Dict[str, Any]:
    automl = AutoMLPipeline()
    result = automl.fit(X_train, y_train, problem_type)
    
    y_pred = automl.predict(X_test)
    
    if problem_type == "classification" or len(np.unique(y_test)) < 10:
        accuracy = np.mean(y_pred == y_test)
        result["test_accuracy"] = float(accuracy)
    else:
        mse = np.mean((y_pred - y_test) ** 2)
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        result["test_mse"] = float(mse)
        result["test_r2"] = float(r2)
    
    return result
