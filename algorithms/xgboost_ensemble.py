"""
XGBoost Ensemble for regression and classification
Simplified XGBoost implementation for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List


class XGBoostTree:
    """Single decision tree for XGBoost"""
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def _gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        p1 = np.sum(y == 1) / len(y)
        p0 = 1 - p1
        return 1 - p0**2 - p1**2
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, 
                    gradients: np.ndarray, hessians: np.ndarray) -> Dict:
        """Find best split using second-order Taylor approximation"""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feature], np.linspace(0, 100, 11))
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or \
                   np.sum(right_mask) < self.min_samples_split:
                    continue
                
                # Calculate gain using second-order information
                g_left = np.sum(gradients[left_mask])
                h_left = np.sum(hessians[left_mask])
                g_right = np.sum(gradients[right_mask])
                h_right = np.sum(hessians[right_mask])
                
                gain = (g_left**2 / (h_left + 0.01) + 
                        g_right**2 / (h_right + 0.01) -
                        (g_left + g_right)**2 / (h_left + h_right + 0.01))
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return {"feature": best_feature, "threshold": best_threshold, "gain": best_gain}
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray,
                    gradients: np.ndarray, hessians: np.ndarray,
                    depth: int = 0) -> Dict:
        """Recursively build tree"""
        node = {"leaf": False}
        
        # Check stopping conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            node["leaf"] = True
            node["value"] = np.sum(gradients) / (np.sum(hessians) + 0.01)
            return node
        
        # Find best split
        split = self._best_split(X, y, gradients, hessians)
        
        if split["feature"] is None or split["gain"] <= 0:
            node["leaf"] = True
            node["value"] = np.sum(gradients) / (np.sum(hessians) + 0.01)
            return node
        
        # Split data
        left_mask = X[:, split["feature"]] <= split["threshold"]
        right_mask = ~left_mask
        
        node["feature"] = split["feature"]
        node["threshold"] = split["threshold"]
        node["gain"] = split["gain"]
        
        # Build subtrees
        node["left"] = self._build_tree(X[left_mask], y[left_mask],
                                        gradients[left_mask], hessians[left_mask],
                                        depth + 1)
        node["right"] = self._build_tree(X[right_mask], y[right_mask],
                                         gradients[right_mask], hessians[right_mask],
                                         depth + 1)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            gradients: np.ndarray, hessians: np.ndarray):
        """Build decision tree"""
        self.tree = self._build_tree(X, y, gradients, hessians)
    
    def predict_one(self, x: np.ndarray) -> float:
        """Predict single sample"""
        node = self.tree
        while not node["leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(x) for x in X])


class XGBoostEnsemble:
    """
    XGBoost Ensemble for gradient boosting
    
    Suitable for: 回归预测、分类问题、特征重要性分析
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5,
                 learning_rate: float = 0.1, subsample: float = 1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.trees = []
        self.base_score = None
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            objective: str = "regression") -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples,)
            objective: "regression" or "classification"
        """
        self.trees = []
        self.base_score = np.mean(y)
        
        current_pred = np.full(len(y), self.base_score)
        
        for i in range(self.n_estimators):
            # Compute gradients and hessians
            if objective == "regression":
                gradients = current_pred - y
                hessians = np.ones(len(y))
            else:  # classification
                probs = self._sigmoid(current_pred)
                gradients = probs - y
                hessians = probs * (1 - probs)
            
            # Subsample
            if self.subsample < 1.0:
                n_samples = int(len(y) * self.subsample)
                indices = np.random.choice(len(y), n_samples, replace=False)
                X_sub = X[indices]
                g_sub = gradients[indices]
                h_sub = hessians[indices]
            else:
                X_sub = X
                g_sub = gradients
                h_sub = hessians
            
            # Build tree
            tree = XGBoostTree(max_depth=self.max_depth)
            tree.fit(X_sub, y, g_sub, h_sub)
            self.trees.append(tree)
            
            # Update predictions
            tree_pred = tree.predict(X)
            current_pred += self.learning_rate * tree_pred
        
        return {
            "status": "success",
            "n_estimators": len(self.trees),
            "base_score": float(self.base_score)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble"""
        pred = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        
        # For classification, apply sigmoid
        return self._sigmoid(pred)
    
    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance from trees"""
        importance = {}
        for tree in self.trees:
            self._extract_importance(tree.tree, importance)
        return importance
    
    def _extract_importance(self, node: Dict, importance: Dict):
        if node.get("leaf"):
            return
        feature = node.get("feature")
        if feature is not None:
            importance[feature] = importance.get(feature, 0) + node.get("gain", 0)
        self._extract_importance(node.get("left", {}), importance)
        self._extract_importance(node.get("right", {}), importance)
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            for k in importance:
                importance[k] /= total
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample
        }
