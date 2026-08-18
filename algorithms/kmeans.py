# KMeans Clustering
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
    def predict(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centers[k]) ** 2, axis=1)
        return np.argmin(distances, axis=1)

