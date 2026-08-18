import os
base = r"algorithms"

with open(os.path.join(base, "kmeans.py"), "r", encoding="utf-8") as f:
    content = f.read()

# Add predict method before the closing of the class
predict_method = '''
    def predict(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centers[k]) ** 2, axis=1)
        return np.argmin(distances, axis=1)
'''

# Insert before the last line (which is the end of class)
content = content.rstrip() + predict_method + "\n"

with open(os.path.join(base, "kmeans.py"), "w", encoding="utf-8") as f:
    f.write(content)
print("Added predict method to KMeans")
