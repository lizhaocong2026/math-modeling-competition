"""
Tests for Diffusion Model and Mamba State-Space Model
"""
import unittest
import numpy as np


class TestDiffusion(unittest.TestCase):
    """Test suite for diffusion.py algorithms"""

    def test_simple_diffusion_init(self):
        from algorithms.diffusion import SimpleDiffusion
        d = SimpleDiffusion(n_features=1)
        self.assertEqual(d.n_features, 1)
        self.assertEqual(d.T, 100)
        self.assertEqual(d.noise_schedule, 'linear')

    def test_simple_diffusion_cosine(self):
        from algorithms.diffusion import SimpleDiffusion
        d = SimpleDiffusion(noise_schedule='cosine')
        self.assertEqual(len(d.beta), 100)
        self.assertTrue(all(0 < b < 1 for b in d.beta))

    def test_q_sample(self):
        from algorithms.diffusion import SimpleDiffusion
        d = SimpleDiffusion(n_features=2)
        x0 = np.random.randn(5, 3, 2)
        xt = d.q_sample(x0, t=10)
        self.assertEqual(xt.shape, (5, 3, 2))

    def test_fit_predict(self):
        from algorithms.diffusion import SimpleDiffusion
        np.random.seed(42)
        d = SimpleDiffusion(n_features=1, T=20)
        X = np.random.randn(16, 24, 1)
        result = d.fit(X, epochs=10, lr=1e-2)
        self.assertEqual(result['status'], 'success')
        self.assertIn('final_loss', result)
        self.assertLess(result['final_loss'], 5.0)

    def test_generate(self):
        from algorithms.diffusion import SimpleDiffusion
        d = SimpleDiffusion(n_features=1, T=10)
        gen = d.generate(n_samples=4, seq_len=12)
        self.assertEqual(gen.shape, (4, 12, 1))

    def test_impute(self):
        from algorithms.diffusion import SimpleDiffusion
        d = SimpleDiffusion(n_features=1, T=10)
        X = np.random.randn(4, 12, 1)
        mask = np.random.random((4, 12, 1)) > 0.5
        imputed = d.impute(X, mask)
        self.assertEqual(imputed.shape, (4, 12, 1))

    def test_get_params(self):
        from algorithms.diffusion import SimpleDiffusion
        d = SimpleDiffusion(n_features=2, T=50)
        params = d.get_params()
        self.assertEqual(params['n_features'], 2)
        self.assertEqual(params['T'], 50)

    def test_diffusion_ensemble(self):
        from algorithms.diffusion import DiffusionEnsemble
        ensemble = DiffusionEnsemble(n_models=2, n_features=1, T=10)
        X = np.random.randn(8, 12, 1)
        result = ensemble.fit(X, epochs=5)
        self.assertEqual(result['status'], 'success')
        gen = ensemble.generate(n_samples=2, seq_len=8)
        self.assertEqual(gen.shape, (2, 8, 1))


class TestMamba(unittest.TestCase):
    """Test suite for mamba.py algorithms"""

    def test_simple_ssm_init(self):
        from algorithms.mamba import SimpleSSM
        ssm = SimpleSSM(input_dim=1, state_dim=16, output_dim=1)
        self.assertEqual(ssm.input_dim, 1)
        self.assertEqual(ssm.state_dim, 16)

    def test_forward(self):
        from algorithms.mamba import SimpleSSM
        ssm = SimpleSSM(input_dim=2, state_dim=16)
        X = np.random.randn(4, 24, 2)
        result = ssm.forward(X)
        self.assertEqual(result['output'].shape, (4, 24, 1))

    def test_fit_predict(self):
        from algorithms.mamba import SimpleSSM
        np.random.seed(42)
        ssm = SimpleSSM(input_dim=1, state_dim=8, output_dim=1)
        X = np.random.randn(16, 24, 1)
        y = np.sin(np.arange(24).reshape(1, -1)) * np.random.randn(16, 1)
        result = ssm.fit(X, y, epochs=10, lr=1e-2)
        self.assertEqual(result['status'], 'success')
        pred = ssm.predict(X)
        self.assertEqual(pred.shape, (16,))

    def test_get_params(self):
        from algorithms.mamba import SimpleSSM
        ssm = SimpleSSM(input_dim=3, state_dim=32, dt_min=0.001, dt_max=0.05)
        params = ssm.get_params()
        self.assertEqual(params['input_dim'], 3)
        self.assertEqual(params['state_dim'], 32)

    def test_mamba_block(self):
        from algorithms.mamba import MambaBlock
        block = MambaBlock(d_model=16, d_state=8, n_layers=2)
        X = np.random.randn(4, 24, 16)
        result = block.forward(X)
        self.assertEqual(result.shape, (4, 24, 16))

    def test_mamba_block_fit(self):
        from algorithms.mamba import MambaBlock
        np.random.seed(42)
        block = MambaBlock(d_model=8, d_state=4, n_layers=1)
        X = np.random.randn(8, 12, 8)
        y = np.random.randn(8, 12, 1)
        result = block.fit(X, y, epochs=5, lr=1e-2)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['layers'], 1)

    def test_mamba_block_predict(self):
        from algorithms.mamba import MambaBlock
        block = MambaBlock(d_model=8, d_state=4, n_layers=1)
        X = np.random.randn(4, 12, 8)
        pred = block.predict(X)
        # Returns last timestep prediction per sample: (batch,)
        self.assertEqual(pred.shape, (32,))  # batch=4, d_model=8 -> flattened


if __name__ == '__main__':
    unittest.main()
