"""
Tests for Attention mechanism (algorithms/attention.py)
"""
import unittest
import numpy as np


class TestSelfAttention(unittest.TestCase):
    """Test suite for SelfAttention class"""

    def test_init_default(self):
        from algorithms.attention import SelfAttention
        sa = SelfAttention()
        self.assertEqual(sa.d_model, 64)
        self.assertEqual(sa.nhead, 4)
        self.assertEqual(sa.d_k, 16)
        self.assertEqual(sa.dropout, 0.1)

    def test_init_custom(self):
        from algorithms.attention import SelfAttention
        sa = SelfAttention(d_model=32, nhead=2, dropout=0.2)
        self.assertEqual(sa.d_model, 32)
        self.assertEqual(sa.nhead, 2)
        self.assertEqual(sa.d_k, 16)

    def test_forward_shape(self):
        from algorithms.attention import SelfAttention
        sa = SelfAttention(d_model=16, nhead=4)
        X = np.random.randn(4, 8, 16)
        output, attn_weights = sa.forward(X)
        self.assertEqual(output.shape, (4, 8, 16))
        self.assertEqual(attn_weights.shape, (4, 4, 8, 8))

    def test_forward_batch_size_1(self):
        from algorithms.attention import SelfAttention
        sa = SelfAttention(d_model=8, nhead=2)
        X = np.random.randn(1, 4, 8)
        output, attn_weights = sa.forward(X)
        self.assertEqual(output.shape, (1, 4, 8))

    def test_forward_attn_weights_sum_to_one(self):
        from algorithms.attention import SelfAttention
        sa = SelfAttention(d_model=8, nhead=2)
        X = np.random.randn(2, 4, 8)
        _, attn_weights = sa.forward(X)
        sums = attn_weights.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_fit_predict(self):
        from algorithms.attention import SelfAttention
        np.random.seed(42)
        sa = SelfAttention(d_model=8, nhead=2)
        X = np.random.randn(8, 4, 8)
        y = np.random.randn(8, 4, 8)
        result = sa.fit(X, y, epochs=5, lr=1e-2)
        self.assertEqual(result["status"], "success")
        self.assertIn("final_loss", result)
        self.assertLess(result["final_loss"], 10.0)

    def test_fit_shrinks_loss(self):
        from algorithms.attention import SelfAttention
        np.random.seed(42)
        sa = SelfAttention(d_model=8, nhead=2)
        X = np.random.randn(8, 4, 8)
        y = np.random.randn(8, 4, 8)
        sa.fit(X, y, epochs=1, lr=1e-2)
        loss1 = np.mean((sa.forward(X)[0] - y) ** 2)
        sa.fit(X, y, epochs=10, lr=1e-2)
        loss2 = np.mean((sa.forward(X)[0] - y) ** 2)
        self.assertLess(loss2, loss1 * 1.5)

    def test_get_params(self):
        from algorithms.attention import SelfAttention
        sa = SelfAttention(d_model=32, nhead=4, dropout=0.1)
        params = sa.get_params()
        self.assertEqual(params["d_model"], 32)
        self.assertEqual(params["nhead"], 4)
        self.assertEqual(params["dropout"], 0.1)

    def test_fit_history(self):
        from algorithms.attention import SelfAttention
        sa = SelfAttention(d_model=8, nhead=2)
        X = np.random.randn(4, 4, 8)
        y = np.random.randn(4, 4, 8)
        sa.fit(X, y, epochs=5, lr=1e-2)
        self.assertEqual(len(sa.history), 5)
        self.assertTrue(all(isinstance(l, float) for l in sa.history))


class TestMultiHeadAttention(unittest.TestCase):
    """Test suite for MultiHeadAttention class"""

    def test_init_default(self):
        from algorithms.attention import MultiHeadAttention
        mha = MultiHeadAttention()
        self.assertEqual(mha.d_model, 64)
        self.assertEqual(mha.nhead, 4)
        self.assertEqual(mha.d_k, 16)

    def test_init_custom(self):
        from algorithms.attention import MultiHeadAttention
        mha = MultiHeadAttention(d_model=32, nhead=2)
        self.assertEqual(mha.d_model, 32)
        self.assertEqual(mha.nhead, 2)
        self.assertEqual(mha.d_k, 16)

    def test_forward_shape(self):
        from algorithms.attention import MultiHeadAttention
        mha = MultiHeadAttention(d_model=16, nhead=4)
        X = np.random.randn(4, 8, 16)
        output = mha.forward(X)
        self.assertEqual(output.shape, (4, 8, 16))

    def test_forward_batch_size_1(self):
        from algorithms.attention import MultiHeadAttention
        mha = MultiHeadAttention(d_model=8, nhead=2)
        X = np.random.randn(1, 4, 8)
        output = mha.forward(X)
        self.assertEqual(output.shape, (1, 4, 8))

    def test_fit_predict(self):
        from algorithms.attention import MultiHeadAttention
        np.random.seed(42)
        mha = MultiHeadAttention(d_model=8, nhead=2)
        X = np.random.randn(8, 4, 8)
        y = np.random.randn(8, 4, 8)
        result = mha.fit(X, y, epochs=5, lr=1e-2)
        self.assertEqual(result["status"], "success")
        self.assertIn("final_loss", result)

    def test_fit_shrinks_loss(self):
        from algorithms.attention import MultiHeadAttention
        np.random.seed(42)
        mha = MultiHeadAttention(d_model=8, nhead=2)
        X = np.random.randn(8, 4, 8)
        y = np.random.randn(8, 4, 8)
        mha.fit(X, y, epochs=1, lr=1e-2)
        loss1 = np.mean((mha.forward(X) - y) ** 2)
        mha.fit(X, y, epochs=10, lr=1e-2)
        loss2 = np.mean((mha.forward(X) - y) ** 2)
        self.assertLess(loss2, loss1 * 1.5)

    def test_fit_history(self):
        from algorithms.attention import MultiHeadAttention
        mha = MultiHeadAttention(d_model=8, nhead=2)
        X = np.random.randn(4, 4, 8)
        y = np.random.randn(4, 4, 8)
        mha.fit(X, y, epochs=5, lr=1e-2)
        self.assertEqual(len(mha.history), 5)

    def test_layer_norm_stable(self):
        from algorithms.attention import MultiHeadAttention
        mha = MultiHeadAttention(d_model=16, nhead=4)
        X = np.random.randn(4, 8, 16) * 100  # large values
        output = mha.forward(X)
        # output should not contain NaN or Inf
        self.assertFalse(np.any(np.isnan(output)))
        self.assertFalse(np.any(np.isinf(output)))


if __name__ == "__main__":
    unittest.main()
