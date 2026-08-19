"""
Tests for Particle Filter, EKF, Spatial Kriging, and Spatial Regression
"""
import unittest
import numpy as np


class TestExtendedKalmanFilter(unittest.TestCase):
    def test_init(self):
        from algorithms.particle_filter import ExtendedKalmanFilter
        ekf = ExtendedKalmanFilter(dim_state=3, dim_measure=2)
        self.assertEqual(ekf.dim_state, 3)
        self.assertEqual(ekf.dim_measure, 2)

    def test_predict_update(self):
        from algorithms.particle_filter import ExtendedKalmanFilter
        ekf = ExtendedKalmanFilter(dim_state=2, dim_measure=1, measure_noise=0.01)
        ekf.H = np.array([[1.0, 0.0]])
        
        result = ekf.predict(0.0)
        self.assertEqual(len(result["state"]), 2)
        
        result = ekf.update(np.array([1.5]))
        self.assertIn("residual", result)

    def test_filter(self):
        from algorithms.particle_filter import ExtendedKalmanFilter
        np.random.seed(42)
        ekf = ExtendedKalmanFilter(dim_state=2, dim_measure=1, process_noise=0.001, measure_noise=0.01)
        ekf.H = np.array([[1.0, 0.0]])
        measurements = np.random.randn(20, 1) * 2 + 1.0
        result = ekf.filter(measurements)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["states"]), 20)

    def test_get_params(self):
        from algorithms.particle_filter import ExtendedKalmanFilter
        ekf = ExtendedKalmanFilter(dim_state=4, dim_measure=2, process_noise=0.05)
        params = ekf.get_params()
        self.assertEqual(params["dim_state"], 4)


class TestParticleFilter(unittest.TestCase):
    def test_init(self):
        from algorithms.particle_filter import ParticleFilter
        pf = ParticleFilter(n_particles=500, dim_state=2)
        self.assertEqual(pf.n_particles, 500)

    def test_filter(self):
        from algorithms.particle_filter import ParticleFilter
        np.random.seed(42)
        pf = ParticleFilter(n_particles=300, dim_state=2, process_std=0.5, measure_std=1.0)
        measurements = np.random.randn(15, 1) * 3 + 2.0
        result = pf.filter(measurements)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["trajectory"]), 15)

    def test_estimate(self):
        from algorithms.particle_filter import ParticleFilter
        pf = ParticleFilter(n_particles=200, dim_state=2)
        est, cov = pf.estimate()
        self.assertEqual(len(est), 2)
        self.assertEqual(cov.shape, (2, 2))

    def test_get_params(self):
        from algorithms.particle_filter import ParticleFilter
        pf = ParticleFilter(n_particles=1000, dim_state=3)
        params = pf.get_params()
        self.assertEqual(params["n_particles"], 1000)


class TestOrdinaryKriging(unittest.TestCase):
    def test_init(self):
        from algorithms.spatial_advanced import OrdinaryKriging
        krig = OrdinaryKriging(variogram_model="spherical")
        self.assertEqual(krig.variogram_model, "spherical")

    def test_fit_predict(self):
        from algorithms.spatial_advanced import OrdinaryKriging
        np.random.seed(42)
        sites = np.random.randn(20, 2) * 10
        values = np.sin(sites[:, 0]) + np.cos(sites[:, 1]) + np.random.randn(20) * 0.1
        krig = OrdinaryKriging(nugget=0.01, sill=1.0, range_param=5.0)
        result = krig.fit(sites, values)
        self.assertEqual(result["status"], "success")
        
        query = np.random.randn(5, 2) * 10
        pred = krig.predict(query)
        self.assertEqual(len(pred["predictions"]), 5)

    def test_get_params(self):
        from algorithms.spatial_advanced import OrdinaryKriging
        krig = OrdinaryKriging(variogram_model="gaussian", sill=2.0)
        params = krig.get_params()
        self.assertEqual(params["sill"], 2.0)


class TestSpatialRegression(unittest.TestCase):
    def test_init(self):
        from algorithms.spatial_advanced import SpatialRegression
        sr = SpatialRegression(spatial_lag=True)
        self.assertTrue(sr.spatial_lag)

    def test_fit(self):
        from algorithms.spatial_advanced import SpatialRegression
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(30) * 0.5
        sr = SpatialRegression(spatial_lag=False)
        result = sr.fit(X, y)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["R2"], 0.5)

    def test_predict(self):
        from algorithms.spatial_advanced import SpatialRegression
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] + X[:, 1] + np.random.randn(30) * 0.1
        sr = SpatialRegression(spatial_lag=False)
        sr.fit(X, y)
        pred = sr.predict(X)
        self.assertEqual(len(pred), 30)

    def test_get_params(self):
        from algorithms.spatial_advanced import SpatialRegression
        sr = SpatialRegression(spatial_lag=True)
        params = sr.get_params()
        self.assertTrue(params["spatial_lag"])


if __name__ == "__main__":
    unittest.main()
