import os
base = r"algorithms"

# Fix time_series_forecasting.py - use float array
with open(os.path.join(base, "time_series_forecasting.py"), "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("result = np.full_like(data, np.nan)", "result = np.full(len(data), np.nan, dtype=float)")
with open(os.path.join(base, "time_series_forecasting.py"), "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed time_series_forecasting.py")

# Fix anomaly_detection.py - use float array
with open(os.path.join(base, "anomaly_detection.py"), "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("anomalies = np.zeros(n, dtype=bool)", "anomalies = np.zeros(n, dtype=bool)")
content = content.replace("scores = np.zeros(n)", "scores = np.zeros(n, dtype=float)")
with open(os.path.join(base, "anomaly_detection.py"), "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed anomaly_detection.py")
