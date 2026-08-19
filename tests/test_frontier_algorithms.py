import unittest
import numpy as np
import sys
sys.path.insert(0, '.')


class TestDiffusionModel(unittest.TestCase):
    def test_diffusion_init(self):
        from algorithms.diffusion import SimpleDiffusion
        model = SimpleDiffusion(n_features=1, T=100)
        self.assertEqual(model.T, 100)
        self.assertEqual(model.n_features, 1)
    
    def test_diffusion_fit(self):
        from algorithms.diffusion import SimpleDiffusion
        model = SimpleDiffusion(n_features=1, T=50)
        X = np.random.rand(30, 1)
        result = model.fit(X, epochs=10)
        self.assertEqual(result['status'], 'success')
        self.assertIn('loss', result)


class TestMambaModel(unittest.TestCase):
    def test_mamba_init(self):
        from algorithms.mamba import SimpleSSM
        model = SimpleSSM(input_dim=1, state_dim=64, output_dim=1)
        self.assertEqual(model.input_dim, 1)
    
    def test_mamba_fit_predict(self):
        from algorithms.mamba import SimpleSSM
        model = SimpleSSM(input_dim=1, state_dim=32, output_dim=1)
        X = np.random.rand(20, 12, 1).astype(np.float32)
        y = np.random.rand(20).astype(np.float32)
        result = model.fit(X, y, epochs=5)
        self.assertEqual(result['status'], 'success')
        pred = model.predict(X)
        self.assertEqual(len(pred), 20)


class TestAttentionLayer(unittest.TestCase):
    def test_attention_init(self):
        from algorithms.attention import SelfAttention
        layer = SelfAttention(d_model=64, nhead=4)
        self.assertEqual(layer.nhead, 4)
    
    def test_attention_forward(self):
        from algorithms.attention import SelfAttention
        layer = SelfAttention(d_model=32, nhead=4)
        X = np.random.rand(10, 8, 32).astype(np.float32)
        output, weights = layer.forward(X)
        self.assertEqual(output.shape, (10, 8, 32))


if __name__ == '__main__':
    unittest.main()
