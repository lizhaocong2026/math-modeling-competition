# DBSCAN Clustering Algorithm
import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def _distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _region_query(self, data, point_idx):
        neighbors = []
        for i in range(len(data)):
            if self._distance(data[point_idx], data[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def fit(self, X):
        n = len(X)
        self.labels = np.array([-1] * n)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                self._expand_cluster(X, neighbors, cluster_id, visited)
                cluster_id += 1
        return self

    def _expand_cluster(self, data, neighbors, cluster_id, visited):
        self.labels[neighbors] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._region_query(data, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    for n in new_neighbors:
                        if n not in neighbors:
                            neighbors.append(n)
            i += 1

    def get_n_clusters(self):
        return len(set(self.labels)) - (1 if -1 in self.labels else 0)
