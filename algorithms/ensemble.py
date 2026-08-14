"""
回归集成方法
结合多种回归模型提高预测精度
"""
import numpy as np
from typing import Dict, Any, List, Optional
import sys
sys.path.insert(0, '..')

from algorithms.linear_regression import LinearRegression
from algorithms.polynomial_regression import PolynomialRegression
from algorithms.grey_model import GM11


class RegressionEnsemble:
    """回归模型集成"""
    
    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = weights
        self.models = []
        self.model_names = []
        self.metric_history = {}
        
    def add_model(self, model, name: str = None):
        """添加模型到集成"""
        self.models.append(model)
        self.model_names.append(name or f"model_{len(self.models)}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """拟合所有模型"""
        for model in self.models:
            if hasattr(model, 'fit'):
                model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """加权集成预测"""
        predictions = np.array([model.predict(X) for model in self.models])
        
        if self.weights is None:
            # 等权重
            return np.mean(predictions, axis=0)
        
        # 加权平均
        weighted_sum = sum(w * p for w, p in zip(self.weights, predictions))
        return weighted_sum / np.sum(self.weights)
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray, method: str = "stacking") -> Dict[str, Any]:
        """拟合并预测"""
        self.fit(X, y)
        y_pred = self.predict(X)
        
        # 计算各项指标
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        
        # 单模型对比
        individual_results = {}
        for model, name in zip(self.models, self.model_names):
            pred = model.predict(X)
            m = np.mean((y - pred)**2)
            individual_results[name] = {"MSE": float(m), "predictions": pred.tolist()}
        
        return {
            "ensemble_predictions": y_pred.tolist(),
            "MSE": float(mse),
            "MAE": float(mae),
            "R2": float(r2),
            "individual_models": individual_results,
            "method": method
        }
    
    def stacking(self, X: np.ndarray, y: np.ndarray, test_X: np.ndarray = None) -> Dict[str, Any]:
        """Stacking集成"""
        # 使用交叉验证生成元特征
        n = len(y)
        meta_features = np.zeros((n, len(self.models)))
        
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            # 简单留一法
            preds = np.zeros(n)
            for train_idx in range(n):
                X_train = np.delete(X, train_idx, axis=0)
                y_train = np.delete(y, train_idx)
                X_val = X[train_idx:train_idx+1]
                
                model_copy = type(model)()
                model_copy.fit(X_train, y_train)
                preds[train_idx] = model_copy.predict(X_val)[0]
            
            meta_features[:, i] = preds
        
        # 用元特征训练最终模型（线性回归）
        final_model = LinearRegression()
        final_model.fit(meta_features, y)
        
        # 预测
        if test_X is not None:
            test_meta = np.zeros((len(test_X), len(self.models)))
            for i, model in enumerate(self.models):
                test_meta[:, i] = model.predict(test_X)
            y_test_pred = final_model.predict(test_meta)
        else:
            y_test_pred = final_model.predict(meta_features)
        
        return {
            "predictions": y_test_pred.tolist(),
            "meta_features": meta_features.tolist()
        }


def auto_select_models(data: np.ndarray, max_degree: int = 3) -> RegressionEnsemble:
    """
    自动选择回归模型
    
    参数:
        data: 时间序列数据
        max_degree: 多项式最高阶数
        
    返回:
        包含多个模型的集成
    """
    n = len(data)
    X = np.arange(n).reshape(-1, 1)
    
    ensemble = RegressionEnsemble()
    
    # 添加不同模型
    ensemble.add_model(GM11(), "GM11")
    ensemble.add_model(LinearRegression(), "Linear")
    
    for degree in range(2, min(max_degree + 1, n)):
        pr = PolynomialRegression(degree=degree)
        pr.fit(X, data)
        ensemble.add_model(pr, f"Poly{degree}")
    
    return ensemble
