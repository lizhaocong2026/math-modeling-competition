"""
Tests for Prophet Ensemble and AutoML Pipeline
"""
import unittest
import numpy as np


class TestProphetModel(unittest.TestCase):
    def test_init(self):
        from algorithms.prophet_ensemble import ProphetModel
        p = ProphetModel(n_changepoints=10, Fourier_order=5)
        self.assertEqual(p.n_changepoints, 10)
        self.assertEqual(p.Fourier_order, 5)

    def test_fit_predict(self):
        from algorithms.prophet_ensemble import ProphetModel
        np.random.seed(42)
        p = ProphetModel(n_changepoints=5, Fourier_order=3)
        t = np.arange(0, 365)
        y = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 365) + np.random.randn(365) * 5
        result = p.fit(y, dates=t)
        self.assertEqual(result["status"], "success")
        future = np.arange(365, 390)
        forecast = p.predict(future)
        self.assertEqual(len(forecast["forecast"]), 25)

    def test_get_params(self):
        from algorithms.prophet_ensemble import ProphetModel
        p = ProphetModel(n_changepoints=15, Fourier_order=8)
        params = p.get_params()
        self.assertEqual(params["n_changepoints"], 15)
        self.assertEqual(params["Fourier_order"], 8)


class TestProphetDecompose(unittest.TestCase):
    def test_init(self):
        from algorithms.prophet_ensemble import ProphetDecompose
        d = ProphetDecompose(period=24, n_harmonics=10)
        self.assertEqual(d.period, 24)

    def test_decompose(self):
        from algorithms.prophet_ensemble import ProphetDecompose
        np.random.seed(42)
        d = ProphetDecompose(period=24)
        t = np.arange(0, 240)
        y = 50 + 10 * np.sin(2 * np.pi * t / 24) + np.random.randn(240) * 2
        result = d.decompose(y)
        self.assertEqual(len(result["trend"]), 240)
        self.assertEqual(len(result["seasonal"]), 240)
        self.assertEqual(len(result["residual"]), 240)

    def test_fit_predict(self):
        from algorithms.prophet_ensemble import ProphetDecompose
        np.random.seed(42)
        d = ProphetDecompose(period=24)
        t = np.arange(0, 240)
        y = 50 + 10 * np.sin(2 * np.pi * t / 24) + np.random.randn(240) * 2
        result = d.fit_predict(y, steps=24)
        self.assertEqual(len(result["forecast"]), 24)
        self.assertIn("mape", result)

    def test_get_params(self):
        from algorithms.prophet_ensemble import ProphetDecompose
        d = ProphetDecompose(period=48, n_harmonics=5)
        params = d.get_params()
        self.assertEqual(params["period"], 48)


class TestAutoMLPipeline(unittest.TestCase):
    def test_init(self):
        from algorithms.prophet_ensemble import AutoMLPipeline
        p = AutoMLPipeline(max_models=3, cv_folds=3)
        self.assertEqual(p.max_models, 3)

    def test_fit(self):
        from algorithms.prophet_ensemble import AutoMLPipeline
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = 2 * X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(50) * 0.1
        p = AutoMLPipeline(max_models=3, cv_folds=2)
        result = p.fit(X, y)
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["best_model"])

    def test_predict(self):
        from algorithms.prophet_ensemble import AutoMLPipeline
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X[:, 0] + X[:, 1] + np.random.randn(50) * 0.05
        p = AutoMLPipeline(max_models=2, cv_folds=2)
        p.fit(X, y)
        pred = p.predict(X)
        self.assertEqual(len(pred), 50)

    def test_evaluate(self):
        from algorithms.prophet_ensemble import AutoMLPipeline
        np.random.seed(42)
        X = np.random.randn(60, 3)
        y = X[:, 0] + X[:, 1] + np.random.randn(60) * 0.05
        p = AutoMLPipeline(max_models=2, cv_folds=2)
        p.fit(X[:50], y[:50])
        metrics = p.evaluate(X[50:], y[50:])
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("R2", metrics)
        self.assertGreater(metrics["R2"], -1)

    def test_get_params(self):
        from algorithms.prophet_ensemble import AutoMLPipeline
        p = AutoMLPipeline(max_models=4)
        params = p.get_params()
        self.assertEqual(params["max_models"], 4)
        self.assertIn("best_model", params)


if __name__ == "__main__":
    unittest.main()
