"""
Improved XGBoost Ensemble and Gradient Boosting for regression
Advanced tree-based ensemble methods for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List, Optional


class XGBoostRegressor:
    """
    Simplified XGBoost regressor with regularization and gradient boosting
    
    Suitable for: 回归预测、特征重要性分析、结构化数据建模
    """
    
    def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        
        self.trees = []
        self.init_pred = 0.0
        self.loss_history = []
    
    def _build_tree(self, X, residuals, depth=0):
        """Build a decision tree leaf"""
        if depth >= self.max_depth or len(residuals) < 2:
            return {"leaf": True, "value": float(np.mean(residuals))}
        
        n_features = X.shape[1]
        best_feat = np.random.randint(n_features)
        best_thresh = np.percentile(X[:, best_feat], 50)
        
        left_mask = X[:, best_feat] < best_thresh
        right_mask = ~left_mask
        
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {"leaf": True, "value": float(np.mean(residuals))}
        
        return {
            "leaf": False,
            "feature": int(best_feat),
            "threshold": float(best_thresh),
            "left": self._build_tree(X[left_mask], residuals[left_mask], depth+1),
            "right": self._build_tree(X[right_mask], residuals[right_mask], depth+1)
        }
    
    def _predict_tree(self, tree, x):
        if tree["leaf"]:
            return tree["value"]
        if x[tree["feature"]] < tree["threshold"]:
            return self._predict_tree(tree["left"], x)
        else:
            return self._predict_tree(tree["right"], x)
    
    def fit(self, X, y, eval_set=None) -> Dict[str, Any]:
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        self.init_pred = float(np.mean(y))
        predictions = np.full(len(y), self.init_pred)
        
        n_samples = X.shape[0]
        
        for i in range(self.n_estimators):
            # Random subsampling
            if self.subsample < 1.0:
                idx = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                X_sub = X[idx]
                y_sub = y[idx]
            else:
                X_sub, y_sub, idx = X, y, np.arange(n_samples)
            
            # Compute residuals (negative gradient of squared loss)
            residuals = y_sub - predictions[idx]
            
            # Build tree on residuals
            tree = self._build_tree(X_sub, residuals)
            self.trees.append(tree)
            
            # Update predictions
            tree_preds = np.array([self._predict_tree(tree, X_sub[j]) for j in range(len(X_sub))])
            predictions[idx] += self.learning_rate * tree_preds
            
            # Track loss
            loss = float(np.mean((y - predictions) ** 2))
            self.loss_history.append(loss)
            
            if (i + 1) % 20 == 0:
                print(f"  Boost {i+1}/{self.n_estimators}, Loss: {loss:.6f}")
        
        return {"status": "success", "final_loss": float(self.loss_history[-1]), "n_trees": len(self.trees)}
    
    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        predictions = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            tree_preds = np.array([self._predict_tree(tree, X[j]) for j in range(X.shape[0])])
            predictions += self.learning_rate * tree_preds
        return predictions
    
    def feature_importance(self, X, y) -> np.ndarray:
        """Approximate feature importance based on tree splits"""
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        for tree in self.trees:
            self._accumulate_importance(tree, importance)
        return importance / (importance.sum() + 1e-10)
    
    def _accumulate_importance(self, tree, importance):
        if not tree["leaf"]:
            importance[tree["feature"]] += 1
            self._accumulate_importance(tree["left"], importance)
            self._accumulate_importance(tree["right"], importance)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda
        }


class GradientBoostingEnsemble:
    """
    Gradient Boosting with custom loss functions
    
    Supports: squared error, absolute error, Huber loss
    """
    
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.05,
                 loss="squared", huber_delta=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.loss = loss
        self.huber_delta = huber_delta
        self.trees = []
        self.init_pred = 0.0
        self.loss_history = []
    
    def _build_tree(self, X, residuals, depth=0):
        if depth >= self.max_depth or len(residuals) < 3:
            return {"leaf": True, "value": float(np.median(residuals))}
        
        n_features = min(X.shape[1], 10)
        features = np.random.choice(X.shape[1], n_features, replace=False)
        best_feat, best_thresh, best_score = None, None, np.inf
        
        for feat in features:
            thresholds = np.percentile(X[:, feat], np.linspace(20, 80, 5))
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                if left_mask.sum() < 2 or (~left_mask).sum() < 2:
                    continue
                left_var = np.var(residuals[left_mask]) * left_mask.sum()
                right_var = np.var(residuals[~left_mask]) * (~left_mask).sum()
                score = left_var + right_var
                if score < best_score:
                    best_score = score
                    best_feat = feat
                    best_thresh = thresh
        
        if best_feat is None:
            return {"leaf": True, "value": float(np.median(residuals))}
        
        left_mask = X[:, best_feat] <= best_thresh
        return {
            "leaf": False, "feature": int(best_feat), "threshold": float(best_thresh),
            "left": self._build_tree(X[left_mask], residuals[left_mask], depth+1),
            "right": self._build_tree(X[~left_mask], residuals[~left_mask], depth+1)
        }
    
    def _predict_tree(self, tree, x):
        if tree["leaf"]:
            return tree["value"]
        return self._predict_tree(tree["left"] if x[tree["feature"]] <= tree["threshold"] else tree["right"], x)
    
    def _negative_gradient(self, y, predictions):
        if self.loss == "squared":
            return y - predictions
        elif self.loss == "absolute":
            return np.sign(y - predictions)
        elif self.loss == "huber":
            residuals = y - predictions
            delta = self.huber_delta
            mask = np.abs(residuals) <= delta
            grad = np.where(mask, residuals, delta * np.sign(residuals))
            return grad
        return y - predictions
    
    def fit(self, X, y) -> Dict[str, Any]:
        X, y = np.array(X), np.array(y)
        self.init_pred = float(np.median(y))
        predictions = np.full(len(y), self.init_pred)
        
        for i in range(self.n_estimators):
            residuals = self._negative_gradient(y, predictions)
            tree = self._build_tree(X, residuals)
            self.trees.append(tree)
            tree_preds = np.array([self._predict_tree(tree, X[j]) for j in range(len(X))])
            predictions += self.learning_rate * tree_preds
            loss = float(np.mean((y - predictions) ** 2))
            self.loss_history.append(loss)
            if (i+1) % 20 == 0:
                print(f"  GB {i+1}/{self.n_estimators}, Loss: {loss:.6f}")
        
        return {"status": "success", "final_loss": float(self.loss_history[-1])}
    
    def predict(self, X):
        X = np.array(X)
        preds = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            tpreds = np.array([self._predict_tree(tree, X[j]) for j in range(X.shape[0])])
            preds += self.learning_rate * tpreds
        return preds
    
    def get_params(self) -> Dict[str, Any]:
        return {"n_estimators": self.n_estimators, "loss": self.loss, "learning_rate": self.learning_rate}
