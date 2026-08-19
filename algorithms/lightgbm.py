# LightGBM Regressor
import numpy as np
from typing import Dict, Any
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    lgb = None

class LightGBMRegressor:
    """LightGBM梯度提升回归器 - 更快的梯度提升"""
    
    def __init__(self, num_leaves: int = 31, learning_rate: float = 0.1,
                 n_estimators: int = 100, min_child_samples: int = 20):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        if not HAS_LGB:
            return {'status': 'error', 'message': '请安装lightgbm: pip install lightgbm'}
        params = {
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'min_child_samples': self.min_child_samples,
            'objective': 'regression'
        }
        params.update(kwargs)
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, train_data)
        return {'status': 'success', 'message': 'LightGBM模型训练完成'}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError('模型未训练')
        return self.model.predict(X)
        
    def get_params(self) -> Dict[str, Any]:
        return {'num_leaves': self.num_leaves, 'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators, 'min_child_samples': self.min_child_samples}
