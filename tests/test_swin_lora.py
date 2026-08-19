import unittest
import numpy as np
import sys
sys.path.insert(0, '.')


class TestSwinTransformer(unittest.TestCase):
    def test_swin_init(self):
        from algorithms.swin_transformer import SelfAttention
        layer = SelfAttention(d_model=64, nhead=4)
        self.assertEqual(layer.nhead, 4)
    
    def test_swin_forward(self):
        from algorithms.swin_transformer import SelfAttention
        layer = SelfAttention(d_model=32, nhead=4)
        X = np.random.rand(10, 8, 32).astype(np.float32)
        output, weights = layer.forward(X)
        self.assertEqual(output.shape, (10, 8, 32))


class TestNeuralNetwork(unittest.TestCase):
    def test_lora_init(self):
        from algorithms.lora import NeuralNetwork
        model = NeuralNetwork([64, 32, 1])
        self.assertIsNotNone(model)
    
    def test_lora_forward(self):
        from algorithms.lora import NeuralNetwork
        model = NeuralNetwork([64, 32, 1])
        X = np.random.rand(50, 64)
        pred = model.forward(X)
        self.assertEqual(pred.shape, (50, 1))


if __name__ == '__main__':
    unittest.main()
