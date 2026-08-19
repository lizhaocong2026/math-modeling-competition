import os
base = "tests"

with open(os.path.join(base, "test_extended_algorithms.py"), "w", encoding="utf-8") as f:
    f.write("""# Tests for extended algorithms: Ensemble, TimeSeries, AnomalyDetection
import numpy as np
import pytest
from algorithms.ensemble_methods import EnsembleMethods
from algorithms.time_series_forecasting import TimeSeriesForecasting
from algorithms.anomaly_detection import AnomalyDetection


class TestEnsembleMethods:
    def test_random_forest_classifier(self):
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        em = EnsembleMethods()
        result = em.random_forest_classifier(X, y, n_estimators=10, random_state=42)
        assert result["accuracy"] > 0
        assert len(result["predictions"]) == 50

    def test_gradient_boosting_classifier(self):
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        em = EnsembleMethods()
        result = em.gradient_boosting_classifier(X, y, n_estimators=10)
        assert result["accuracy"] > 0

    def test_adaboost_classifier(self):
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        em = EnsembleMethods()
        result = em.adaboost_classifier(X, y, n_estimators=10)
        assert result["accuracy"] > 0

    def test_compare_methods(self):
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        em = EnsembleMethods()
        results = em.compare_ensemble_methods(X, y)
        assert len(results) >= 3


class TestTimeSeriesForecasting:
    def test_moving_average(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ts = TimeSeriesForecasting()
        result = ts.moving_average(data, window=3)
        assert len(result) == 10
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0

    def test_exponential_smoothing(self):
        data = np.array([1, 2, 3, 4, 5])
        ts = TimeSeriesForecasting()
        result = ts.exponential_smoothing(data, alpha=0.5)
        assert len(result) == 5
        assert result[0] == 1.0

    def test_linear_trend_forecast(self):
        data = np.array([10, 12, 14, 16, 18])
        ts = TimeSeriesForecasting()
        result = ts.linear_trend_forecast(data, steps=3)
        assert len(result) == 3
        assert result[0] > 18

    def test_naive_forecast(self):
        data = np.array([1, 2, 3, 4, 5])
        ts = TimeSeriesForecasting()
        result = ts.naive_forecast(data, steps=3)
        assert len(result) == 3
        assert all(r == 5 for r in result)

    def test_compute_mape(self):
        actual = np.array([10, 20, 30])
        predicted = np.array([11, 19, 32])
        ts = TimeSeriesForecasting()
        mape = ts.compute_mape(actual, predicted)
        assert 0 < mape < 100

    def test_compute_rmse(self):
        actual = np.array([10, 20, 30])
        predicted = np.array([11, 19, 32])
        ts = TimeSeriesForecasting()
        rmse = ts.compute_rmse(actual, predicted)
        assert rmse > 0

    def test_compare_methods(self):
        np.random.seed(42)
        data = 50 + np.cumsum(np.random.randn(100) * 2)
        ts = TimeSeriesForecasting()
        results = ts.compare_methods(data, test_ratio=0.2)
        assert len(results) >= 3
        for name, metrics in results.items():
            assert "mape" in metrics
            assert "rmse" in metrics


class TestAnomalyDetection:
    def test_z_score_method(self):
        data = np.array([1, 2, 3, 4, 5, 100, 7, 8, 9, 10])
        ad = AnomalyDetection()
        anomalies, scores = ad.z_score_method(data, threshold=2.0)
        assert anomalies[5] == True
        assert len(scores) == 10

    def test_iqr_method(self):
        data = np.array([1, 2, 3, 4, 5, 100, 7, 8, 9, 10])
        ad = AnomalyDetection()
        anomalies, scores = ad.iqr_method(data)
        assert anomalies[5] == True

    def test_modified_z_score(self):
        data = np.array([1, 2, 3, 4, 5, 100, 7, 8, 9, 10])
        ad = AnomalyDetection()
        anomalies, scores = ad.modified_z_score(data)
        assert len(anomalies) == 10

    def test_sliding_window_anomaly(self):
        data = np.array([1, 2, 3, 4, 5, 100, 7, 8, 9, 10])
        ad = AnomalyDetection()
        anomalies, scores = ad.sliding_window_anomaly(data, window=3, threshold=1.5)
        assert anomalies[5] == True

    def test_isolation_anomaly_score(self):
        data = np.array([1, 2, 3, 4, 5, 100, 7, 8, 9, 10], dtype=float)
        ad = AnomalyDetection()
        scores = ad.isolation_anomaly_score(data, n_samples=20)
        assert len(scores) == 10
        assert scores[5] == max(scores)
""")
print("Created test_extended_algorithms.py")
