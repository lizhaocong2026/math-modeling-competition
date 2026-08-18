import os
base = r"algorithms"

# Fix dea.py - the CCR model needs proper formulation
# For each DMU d: maximize u^T y_d s.t. v^T x_d = 1, u^T Y <= v^T X
with open(os.path.join(base, "dea.py"), "r", encoding="utf-8") as f:
    content = f.read()

new_content = '''# DEA Data Envelopment Analysis
import numpy as np
from scipy.optimize import linprog

class DEA:
    def __init__(self):
        self.results = None
    
    def evaluate(self, X, Y, target_idx=None):
        """
        CCR model DEA evaluation
        X: input matrix (m, n) - m inputs, n DMUs
        Y: output matrix (s, n) - s outputs, n DMUs
        """
        m, n = X.shape
        s = Y.shape[0]
        if target_idx is None:
            target_idx = list(range(n))
        efficiencies = {}
        for idx in target_idx:
            # Variables: v (m inputs), u (s outputs)
            # Maximize u^T y_d  s.t. v^T x_d = 1, u^T Y <= v^T X, v,u >= 0
            # Convert to min: min -u^T y_d
            c = np.zeros(m + s)
            c[m:] = -Y[:, idx]
            
            # Equality: v^T x_d = 1
            A_eq = np.zeros((1, m + s))
            A_eq[0, :m] = X[:, idx]
            b_eq = np.array([1.0])
            
            # Inequalities: u^T Y_j - v^T x_j <= 0 for all j
            A_ub_list = []
            b_ub_list = []
            for j in range(n):
                row = np.zeros(m + s)
                row[:m] = -X[:, j]    # -v^T x_j
                row[m:] = Y[:, j]     # +u^T Y_j
                A_ub_list.append(row)
                b_ub_list.append(0.0)
            A_ub = np.array(A_ub_list)
            b_ub = np.array(b_ub_list)
            
            bounds = [(0, None)] * (m + s)
            
            try:
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
                if result.success:
                    efficiencies[idx] = float(-result.fun)
                else:
                    efficiencies[idx] = None
            except Exception:
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
'''

with open(os.path.join(base, "dea.py"), "w", encoding="utf-8") as f:
    f.write(new_content)
print("Fixed dea.py")
