import os
base = r"algorithms"

with open(os.path.join(base, "anomaly_detection.py"), "w", encoding="utf-8") as f:
    f.write("""# Anomaly Detection Algorithms
import numpy as np
from typing import Tuple, Dict, Any

class AnomalyDetection:
    \"\"\"异常检测算法集合\"\"\"
    
    def __init__(self):
        self.results = {}
    
    def z_score_method(self, data, threshold=3.0):
        \"\"\"Z-score异常检测\"\"\"
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        z_scores = np.abs((data - mean) / std)
        anomalies = z_scores > threshold
        return anomalies, z_scores
    
    def iqr_method(self, data):
        \"\"\"IQR异常检测\"\"\"
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies = (data < lower_bound) | (data > upper_bound)
        scores = np.where(data < lower_bound, lower_bound - data, 
                         np.where(data > upper_bound, data - upper_bound, 0))
        return anomalies, scores
    
    def modified_z_score(self, data, threshold=3.5):
        \"\"\"修正Z-score（使用中位数更稳健）\"\"\"
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            mad = 1.0
        modified_z = 0.6745 * np.abs(data - median) / mad
        anomalies = modified_z > threshold
        return anomalies, modified_z
    
    def isolation_anomaly_score(self, data, n_samples=100, seed=42):
        \"\"\"基于孤立森林思想的异常得分\"\"\"
        np.random.seed(seed)
        n = len(data)
        path_lengths = np.zeros(n)
        
        for _ in range(n_samples):
            thresholds = np.sort(np.random.choice(data, size=min(n-1, 50), replace=False))
            for i in range(n):
                val = data[i]
                depth = 0
                current_thresh = thresholds
                while len(current_thresh) > 0:
                    depth += 1
                    split = current_thresh[len(current_thresh) // 2]
                    if val < split:
                        current_thresh = current_thresh[:len(current_thresh) // 2]
                    else:
                        current_thresh = current_thresh[len(current_thresh) // 2 + 1:]
                path_lengths[i] += depth
        
        avg_path = path_lengths / n_samples
        max_path = np.log2(n) if n > 1 else 1
        anomaly_scores = 2 ** (-avg_path / max_path)
        return anomaly_scores
    
    def sliding_window_anomaly(self, data, window=10, threshold=2.0):
        \"\"\"滑动窗口异常检测\"\"\"
        n = len(data)
        anomalies = np.zeros(n, dtype=bool)
        scores = np.zeros(n)
        
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            window_data = data[start:end]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std > 0:
                scores[i] = abs(data[i] - mean) / std
                anomalies[i] = scores[i] > threshold
        
        return anomalies, scores
    
    def detect_all(self, data):
        \"\"\"综合检测\"\"\"
        results = {
            "z_score": self.z_score_method(data),
            "iqr": self.iqr_method(data),
            "modified_z": self.modified_z_score(data),
            "isolation": self.isolation_anomaly_score(data),
            "sliding_window": self.sliding_window_anomaly(data)
        }
        return results
""")
print("Created anomaly_detection.py")
