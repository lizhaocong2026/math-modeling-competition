"""
随机森林回归与分类
集成学习方法，抗过拟合能力强
"""
import numpy as np
from typing import Optional, Dict, Any, List
import random


class DecisionTree:
    """决策树节点"""
    
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  # 叶子节点的值
        
    def fit(self, X: np.ndarray, y: np.ndarray, max_depth: int = 10, 
            min_samples_split: int = 2):
        """构建决策树"""
        self._build_tree(X, y, max_depth, min_samples_split)
    
    def _build_tree(self, X, y, max_depth, min_samples_split):
        n_samples, n_features = X.shape
        
        # 终止条件
        if (max_depth <= 0 or n_samples < min_samples_split or 
            len(np.unique(y)) == 1):
            self.value = np.mean(y) if len(y) > 0 else 0
            return
        
        # 选择最佳分裂点
        best_feature, best_threshold, best_loss = self._best_split(X, y)
        
        if best_loss == float('inf'):
            self.value = np.mean(y)
            return
        
        self.feature = best_feature
        self.threshold = best_threshold
        
        # 分裂数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # 递归构建子树
        self.left = DecisionTree()
        self.left._build_tree(X[left_mask], y[left_mask], max_depth - 1, min_samples_split)
        
        self.right = DecisionTree()
        self.right._build_tree(X[right_mask], y[right_mask], max_depth - 1, min_samples_split)
    
    def _best_split(self, X, y):
        """寻找最佳分裂点"""
        n_samples, n_features = X.shape
        best_feature, best_threshold, best_loss = None, None, float('inf')
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feature], np.linspace(5, 95, 20))
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # 计算基尼不纯度
                loss = self._gini_loss(y[left_mask], y[right_mask], n_samples)
                
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_loss
    
    def _gini_loss(self, y_left, y_right, n_total):
        """计算基尼不纯度损失"""
        n_left, n_right = len(y_left), len(y_right)
        if n_left == 0 or n_right == 0:
            return float('inf')
        
        # 方差损失（回归）
        loss = (n_left * np.var(y_left) + n_right * np.var(y_right)) / n_total
        return loss
    
    def predict_one(self, x):
        # x is a 1D array for single prediction
        """预测单个样本"""
        if self.value is not None:
            return self.value
        
        if x[self.feature] <= self.threshold:
            return self.left.predict_one(x)
        else:
            return self.right.predict_one(x)


class RandomForest:
    """随机森林"""
    
    def __init__(self, n_trees: int = 100, max_depth: int = 10,
                 max_features: Optional[int] = None, random_state: int = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练随机森林"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        if self.max_features is None:
            self.max_features = n_features
        
        self.trees = []
        
        for i in range(self.n_trees):
            # Bootstrap采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # 随机选择特征子集
            feature_subset = np.random.choice(n_features, 
                                              min(self.max_features, n_features), 
                                              replace=False)
            X_boot = X_boot[:, feature_subset]
            
            # 构建决策树
            tree = DecisionTree()
            tree.fit(X_boot, y_boot, self.max_depth)
            
            self.trees.append((tree, feature_subset))
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        predictions = np.array([tree.predict_one(X[i, tree_features])
                               for i in range(len(X))
                               for tree, tree_features in self.trees]).reshape(-1, len(self.trees))
        return np.mean(predictions, axis=1)
        return np.mean(predictions, axis=1)
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """拟合并返回结果"""
        self.fit(X, y)
        y_pred = self.predict(X)
        
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        
        return {
            "predictions": y_pred.tolist(),
            "MSE": float(mse),
            "R2": float(r2),
            "n_trees": self.n_trees
        }
