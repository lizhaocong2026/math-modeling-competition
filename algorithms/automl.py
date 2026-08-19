# AutoML Pipeline - Automated Machine Learning
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

class AutoMLPipeline:
    """自动机器学习流水线 - 超参搜索与模型选择"""
    
    def __init__(self, models: List[str] = None):
        self.models = models or ['linear_regression', 'random_forest', 'svm', 'xgboost']
        self.results = {}
        
    def evaluate_all(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Dict]:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        
        model_map = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'svm': SVR(kernel='rbf'),
            'knn': KNeighborsRegressor(n_neighbors=5)
        }
        
        for name, model in model_map.items():
            if name in self.models:
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                self.results[name] = {
                    'mean_r2': float(np.mean(scores)),
                    'std_r2': float(np.std(scores)),
                    'best_score': float(np.max(scores))
                }
                
        return self.results
        
    def get_best_model(self) -> str:
        if not self.results:
            raise ValueError('请先运行evaluate_all')
        return max(self.results.items(), key=lambda x: x[1]['mean_r2'])[0]
        
    def get_summary(self) -> str:
        if not self.results:
            return 'No results yet'
        lines = ['Model Performance (R2 score):']
        for name, metrics in sorted(self.results.items(), key=lambda x: -x[1]['mean_r2']):
            lines.append(f'  {name}: {metrics["mean_r2"]:.4f} +/- {metrics["std_r2"]:.4f}')
        return chr(10).join(lines)
