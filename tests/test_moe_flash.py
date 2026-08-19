import unittest
import numpy as np
import sys
sys.path.insert(0, '.')


class TestMoE(unittest.TestCase):
    def test_moe_init(self):
        from algorithms.moe import MoEModel
        model = MoEModel(input_dim=64, num_experts=4, expert_dim=32)
        self.assertEqual(model.input_dim, 64)
    
    def test_moe_forward(self):
        from algorithms.moe import MoEModel
        model = MoEModel(input_dim=64, num_experts=4, expert_dim=32)
        X = np.random.rand(50, 64)
        pred = model.forward(X)
        self.assertEqual(pred.shape, (50, 1))
    
    def test_moe_fit(self):
        from algorithms.moe import MoEModel
        model = MoEModel(input_dim=64, num_experts=4, expert_dim=32)
        X = np.random.rand(50, 64)
        y = np.random.rand(50, 1)
        result = model.fit(X, y, epochs=10)
        self.assertEqual(result['status'], 'success')


class TestFlashAttention(unittest.TestCase):
    def test_flash_init(self):
        from algorithms.flash_attention import FlashAttention
        model = FlashAttention(d_model=64, nhead=4)
        self.assertEqual(model.nhead, 4)
    
    def test_flash_forward(self):
        from algorithms.flash_attention import FlashAttention
        model = FlashAttention(d_model=64, nhead=4)
        X = np.random.rand(10, 16, 64)
        output, weights = model.forward(X)
        self.assertEqual(output.shape, (10, 16, 64))
    
    def test_flash_fit(self):
        from algorithms.flash_attention import FlashAttention
        model = FlashAttention(d_model=64, nhead=4)
        X = np.random.rand(10, 16, 64)
        y = np.random.rand(10, 16, 64)
        result = model.fit(X, y, epochs=5)
        self.assertEqual(result['status'], 'success')


if __name__ == '__main__':
    unittest.main()
