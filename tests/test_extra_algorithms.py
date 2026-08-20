import unittest
import numpy as np
import sys
sys.path.insert(0, ".")

class TestNeuralODE(unittest.TestCase):
    def test_neuralode_init(self):
        from algorithms.neural_ode import NeuralODE
        m = NeuralODE(input_dim=2)
        self.assertEqual(m.input_dim, 2)
    def test_neuralode_forward(self):
        from algorithms.neural_ode import NeuralODE
        m = NeuralODE(input_dim=2)
        X = np.random.rand(10, 2)
        pred = m.forward(X)
        self.assertEqual(pred.shape, (10, 1))
    def test_neuralode_fit(self):
        from algorithms.neural_ode import NeuralODE
        m = NeuralODE(input_dim=2)
        X = np.random.rand(20, 2)
        y = np.random.rand(20, 1)
        result = m.fit(X, y)
        self.assertEqual(result["status"], "success")

class TestVAE(unittest.TestCase):
    def test_vae_init(self):
        from algorithms.vae_enhanced import VAE
        m = VAE(input_dim=32, latent_dim=8)
        self.assertEqual(m.latent_dim, 8)
    def test_vae_forward(self):
        from algorithms.vae_enhanced import VAE
        m = VAE(input_dim=32, latent_dim=8)
        X = np.random.rand(10, 32)
        recon, mu, logvar = m.forward(X)
        self.assertEqual(recon.shape, (10, 32))
    def test_vae_fit(self):
        from algorithms.vae_enhanced import VAE
        m = VAE(input_dim=32, latent_dim=8)
        X = np.random.rand(20, 32)
        result = m.fit(X, epochs=5)
        self.assertEqual(result["status"], "success")

class TestMultiHeadAttention(unittest.TestCase):
    def test_mha_init(self):
        from algorithms.multi_head_attention import MultiHeadAttention
        m = MultiHeadAttention(d_model=64, nhead=4)
        self.assertEqual(m.nhead, 4)
    def test_mha_forward(self):
        from algorithms.multi_head_attention import MultiHeadAttention
        m = MultiHeadAttention(d_model=64, nhead=4)
        X = np.random.rand(10, 16, 64)
        out, weights = m.forward(X)
        self.assertEqual(out.shape, (10, 16, 64))

class TestLightGBM(unittest.TestCase):
    def test_lgbm_init(self):
        from algorithms.lightgbm_simple import LightGBMSimple
        m = LightGBMSimple(n_estimators=10)
        self.assertEqual(m.n_estimators, 10)
    def test_lgbm_forward(self):
        from algorithms.lightgbm_simple import LightGBMSimple
        m = LightGBMSimple(n_estimators=10)
        X = np.random.rand(20, 5)
        pred = m.forward(X)
        self.assertEqual(pred.shape, (20,))
    def test_lgbm_fit(self):
        from algorithms.lightgbm_simple import LightGBMSimple
        m = LightGBMSimple(n_estimators=10)
        X = np.random.rand(30, 5)
        y = np.random.rand(30)
        result = m.fit(X, y)
        self.assertEqual(result["status"], "success")

if __name__ == "__main__":
    unittest.main()