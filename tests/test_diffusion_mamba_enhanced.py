
import os
import unittest
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')


class TestDiffusionTS(unittest.TestCase):
    def test_diffusion_ts_init(self):
        from algorithms.diffusion_ts import SimpleDiffusion
        model = SimpleDiffusion(n_features=1, T=100)
        self.assertEqual(model.T, 100)
    
    def test_diffusion_ts_forecast(self):
        from algorithms.diffusion_ts import SimpleDiffusion
        model = SimpleDiffusion(n_features=1, T=50)
        X = np.random.rand(30, 1)
        result = model.fit(X, epochs=10)
        self.assertEqual(result['status'], 'success')
        samples = model.generate(10)
        self.assertEqual(len(samples), 10)


class TestSelectiveSSM(unittest.TestCase):
    def test_selective_ssm_init(self):
        from algorithms.mamba_enhanced import SimpleSSM
        model = SimpleSSM(input_dim=1, state_dim=64, output_dim=1)
        self.assertEqual(model.input_dim, 1)
    
    def test_selective_ssm_predict(self):
        from algorithms.mamba_enhanced import SimpleSSM
        model = SimpleSSM(input_dim=1, state_dim=32, output_dim=1)
        X = np.random.rand(20, 12, 1).astype(np.float32)
        y = np.random.rand(20, 12, 1).astype(np.float32)
        result = model.fit(X, y, epochs=5)
        self.assertEqual(result['status'], 'success')
        pred = model.predict(X)
        self.assertGreater(len(pred), 0)


if __name__ == '__main__':
    unittest.main()
