# SVR - Support Vector Regression
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple

class SVRRegressor:
    """SVM回归预测器 - 适用于小样本非线性回归"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, **kwargs)
        self.model.fit(X_scaled, y_scaled)
        return {'status': 'success', 'message': 'SVR模型训练完成'}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        return {
            'mse': float(mean_squared_error(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'mae': float(mean_absolute_error(y, y_pred)),
            'r2': float(r2_score(y, y_pred))
        }
        
    def get_params(self) -> Dict[str, Any]:
        return {'kernel': self.kernel, 'C': self.C, 'epsilon': self.epsilon}
