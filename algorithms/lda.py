# LDA - Linear Discriminant Analysis
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDAClassifier:
    """线性判别分析 - 分类降维利器"""
    
    def __init__(self, n_components: int = None, shrinkage: str = None):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        self.model = LinearDiscriminantAnalysis(n_components=self.n_components, 
                                                shrinkage=self.shrinkage)
        self.model.fit(X, y)
        return {'status': 'success', 'message': 'LDA模型训练完成'}
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.model.score(X, y)
        
    def get_params(self) -> Dict[str, Any]:
        return {'n_components': self.n_components, 'shrinkage': self.shrinkage}
