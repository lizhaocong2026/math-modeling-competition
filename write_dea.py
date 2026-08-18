import os
base = r"algorithms"

dea_content = """# DEA Data Envelopment Analysis
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
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
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
"""

with open(os.path.join(base, "dea.py"), "w", encoding="utf-8") as f:
    f.write(dea_content)
print("Created dea.py")
