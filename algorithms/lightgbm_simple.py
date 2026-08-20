import numpy as np
from typing import Dict, Any


class LightGBMSimple:
    def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.05, min_samples_split=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.trees = []
        self.init_pred = 0.0
        self.history = []

    def _build_tree(self, X, residuals, depth=0):
        if depth >= self.max_depth or len(residuals) < self.min_samples_split:
            return dict(leaf=True, value=float(np.mean(residuals)))
        best_feat, best_thresh, best_score = 0, 0.0, np.inf
        n_feats = X.shape[1]
        for f in range(n_feats):
            vals = np.sort(np.unique(X[:, f]))
            if len(vals) < 2: continue
            thresholds = (vals[:-1] + vals[1:]) / 2
            for thresh in thresholds[::max(1, len(thresholds)//10)]:
                lm = X[:, f] <= thresh
                rm = ~lm
                if np.sum(lm) < 2 or np.sum(rm) < 2: continue
                score = np.var(residuals[lm]) * np.sum(lm) + np.var(residuals[rm]) * np.sum(rm)
                if score < best_score:
                    best_score = score
                    best_feat = f
                    best_thresh = thresh
        lm = X[:, best_feat] <= best_thresh
        rm = ~lm
        return dict(leaf=False, feat=best_feat, thresh=best_thresh,
                   left=self._build_tree(X[lm], residuals[lm], depth+1),
                   right=self._build_tree(X[rm], residuals[rm], depth+1))

    def _predict_tree(self, tree, x):
        if tree["leaf"]: return tree["value"]
        if x[tree["feat"]] <= tree["thresh"]:
            return self._predict_tree(tree["left"], x)
        return self._predict_tree(tree["right"], x)

    def forward(self, X):
        pred = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            pred += self.learning_rate * np.array([self._predict_tree(tree, x) for x in X])
        return pred

    def fit(self, X, y, verbose=False):
        self.init_pred = float(np.mean(y))
        pred = np.full(y.shape[0], self.init_pred)
        for i in range(self.n_estimators):
            residuals = y - pred
            tree = self._build_tree(X, residuals)
            self.trees.append(tree)
            pred += self.learning_rate * np.array([self._predict_tree(tree, x) for x in X])
            loss = np.mean((y - pred)**2)
            self.history.append(loss)
            if verbose and (i % 20 == 0 or i == self.n_estimators-1):
                print("Tree %d/%d MSE=%.6f", i, self.n_estimators, loss)
        return dict(status="success", final_mse=self.history[-1] if self.history else None)

    def predict(self, X): return self.forward(X)