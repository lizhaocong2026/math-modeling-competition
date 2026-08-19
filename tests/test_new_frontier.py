import unittest
import numpy as np
import sys
sys.path.insert(0, ".")

class TestPINN(unittest.TestCase):
    def test_pinn_init(self):
        from algorithms.pinn import PINN
        m = PINN(input_dim=2)
        self.assertEqual(m.input_dim, 2)
    def test_pinn_forward(self):
        from algorithms.pinn import PINN
        m = PINN(input_dim=2)
        X = np.random.rand(10, 2)
        pred = m.forward(X)
        self.assertEqual(pred.shape, (10, 1))
    def test_pinn_fit(self):
        from algorithms.pinn import PINN
        m = PINN(input_dim=2)
        X = np.random.rand(20, 2)
        y = np.random.rand(20, 1)
        result = m.fit(X, y, epochs=5)
        self.assertEqual(result["status"], "success")

class TestContrastive(unittest.TestCase):
    def test_contrastive_encode(self):
        from algorithms.contrastive import ContrastiveEncoder
        m = ContrastiveEncoder(input_dim=10, embed_dim=5)
        X = np.random.rand(20, 10)
        emb = m.encode(X)
        self.assertEqual(emb.shape, (20, 5))
    def test_contrastive_loss(self):
        from algorithms.contrastive import ContrastiveLoss
        fn = ContrastiveLoss()
        a = np.random.rand(10, 8)
        p = np.random.rand(10, 8)
        n = np.random.rand(10, 8)
        loss, sim = fn.forward(a, p, n)
        self.assertTrue(loss > 0)

class TestDiffusionLM(unittest.TestCase):
    def test_diffusion_init(self):
        from algorithms.diffusion_lm import DiffusionLM
        m = DiffusionLM(seq_len=8, latent_dim=16)
        self.assertEqual(m.seq_len, 8)
    def test_diffusion_generate(self):
        from algorithms.diffusion_lm import DiffusionLM
        m = DiffusionLM(seq_len=8, latent_dim=16, num_steps=10)
        out = m.generate(shape=(2, 8, 16))
        self.assertEqual(out.shape, (2, 8, 16))
    def test_diffusion_fit(self):
        from algorithms.diffusion_lm import DiffusionLM
        m = DiffusionLM(seq_len=8, latent_dim=16, num_steps=10)
        X = np.random.rand(10, 8, 16)
        result = m.fit(X, epochs=5)
        self.assertEqual(result["status"], "success")

class TestRLHF(unittest.TestCase):
    def test_rlhf_init(self):
        from algorithms.rlhf import RLHFTrainer
        m = RLHFTrainer(num_actions=5)
        self.assertEqual(m.num_actions, 5)
    def test_rlhf_train_step(self):
        from algorithms.rlhf import RLHFTrainer
        m = RLHFTrainer(num_actions=5)
        loss = m.train_step(0, 1, 2)
        self.assertIsInstance(loss, float)
    def test_rlhf_fit(self):
        from algorithms.rlhf import RLHFTrainer
        m = RLHFTrainer(num_actions=5)
        prefs = [(0,1,2), (1,2,0), (2,0,1)]
        result = m.fit(prefs, epochs=5)
        self.assertEqual(result["status"], "success")

if __name__ == "__main__":
    unittest.main()