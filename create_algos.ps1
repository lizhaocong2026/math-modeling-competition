import os
base = r'D:\本地的知识库构建\math-modeling-competition\algorithms'

# dea.py
with open(os.path.join(base, 'dea.py'), 'w', encoding='utf-8') as f:
    f.write("""# DEA Data Envelopment Analysis
import numpy as np
from scipy.optimize import linprog

class DEA:
    def __init__(self):
        self.results = None
    
    def evaluate(self, X, Y, target_idx=None):
        m, n = X.shape
        if target_idx is None:
            target_idx = list(range(n))
        efficiencies = {}
        for idx in target_idx:
            c = np.zeros(m + 1)
            A_ub = -X.T
            b_ub = np.zeros(n)
            A_eq = np.zeros((1, m))
            A_eq[0] = X[:, idx]
            b_eq = np.array([1.0])
            bounds = [(0, None)] * m
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if result.success:
                efficiencies[idx] = float(result.fun)
            else:
                efficiencies[idx] = None
        self.results = efficiencies
        return efficiencies
    
    def classify_efficiency(self, efficiencies, threshold=0.8):
        classification = {}
        for idx, eff in efficiencies.items():
            if eff is None:
                classification[idx] = "failed"
            elif eff >= 1.0:
                classification[idx] = "effective"
            elif eff >= threshold:
                classification[idx] = "near_effective"
            else:
                classification[idx] = "ineffective"
        return classification
""")

# kmeans.py
with open(os.path.join(base, 'kmeans.py'), 'w', encoding='utf-8') as f:
    f.write("""# KMeans Clustering Algorithm
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.centers = None
        self.labels = None
        self.inertia = None
    
    def _init_centers(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _assign_clusters(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centers[k]) ** 2, axis=1)
        return np.argmin(distances, axis=1)
    
    def _update_centers(self, X, labels):
        new_centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) > 0:
                new_centers[k] = members.mean(axis=0)
            else:
                new_centers[k] = self.centers[k]
        return new_centers
    
    def fit(self, X):
        best_labels = None
        best_inertia = float("inf")
        for init_idx in range(self.n_init):
            self.centers = self._init_centers(X)
            for iteration in range(self.max_iter):
                labels = self._assign_clusters(X)
                new_centers = self._update_centers(X, labels)
                if np.allclose(self.centers, new_centers):
                    break
                self.centers = new_centers
            inertia = self._compute_inertia(X, labels)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
        self.labels = best_labels
        self.inertia = best_inertia
        return self
    
    def _compute_inertia(self, X, labels):
        inertia = 0.0
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) > 0:
                inertia += np.sum((members - self.centers[k]) ** 2)
        return inertia
""")

print("Created dea.py and kmeans.py")
