# XGBoost Regressor
import numpy as np
from typing import Dict, Any
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None

class XGBoostRegressor:
    """XGBoost梯度提升回归器 - 大数据集预测神器"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, subsample: float = 0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        if not HAS_XGB:
            return {'status': 'error', 'message': '请安装xgboost: pip install xgboost'}
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'objective': 'reg:squarederror'
        }
        params.update(kwargs)
        self.model = xgb.train(params, dtrain, self.n_estimators)
        return {'status': 'success', 'message': 'XGBoost模型训练完成'}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError('模型未训练')
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
        
    def feature_importance(self) -> np.ndarray:
        if self.model is None:
            raise ValueError('模型未训练')
        return self.model.get_fscore()
        
    def get_params(self) -> Dict[str, Any]:
        return {'n_estimators': self.n_estimators, 'max_depth': self.max_depth,
                'learning_rate': self.learning_rate, 'subsample': self.subsample}
