# RF Regression - Random Forest Regressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any

class RandomForestRegressorModel:
    """随机森林回归器 - 非线性关系建模"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 min_samples_split: int = 2, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        return {'status': 'success', 'message': '随机森林回归器训练完成'}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_
        
    def get_params(self) -> Dict[str, Any]:
        return {'n_estimators': self.n_estimators, 'max_depth': self.max_depth}
