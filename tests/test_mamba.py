"""Tests for Mamba SSM and SimpleSSM"""
import sys
sys.path.insert(0, r"D:\本地的知识库构建\math-modeling-competition")
import pytest
import numpy as np


class TestSimpleSSM:
    def test_init(self):
        from algorithms.mamba import SimpleSSM
        ssm = SimpleSSM(input_dim=2, state_dim=16, output_dim=1)
        assert ssm.input_dim == 2
        assert ssm.state_dim == 16

    def test_forward(self):
        from algorithms.mamba import SimpleSSM
        ssm = SimpleSSM(input_dim=1, state_dim=8, output_dim=1)
        X = np.random.randn(4, 10, 1)
        result = ssm.forward(X)
        assert result["output"].shape == (4, 10, 1)

    def test_fit_predict(self):
        from algorithms.mamba import SimpleSSM
        np.random.seed(42)
        ssm = SimpleSSM(input_dim=1, state_dim=16, output_dim=1)
        X = np.random.randn(20, 8, 1)
        y = np.sin(np.arange(8)).reshape(1, -1, 1) * np.random.randn(20, 1, 1)
        result = ssm.fit(X, y, epochs=5, lr=1e-2)
        assert result["status"] == "success"
        pred = ssm.predict(X)
        assert len(pred) == 20

    def test_get_params(self):
        from algorithms.mamba import SimpleSSM
        ssm = SimpleSSM(dt_min=0.001, dt_max=0.1)
        params = ssm.get_params()
        assert params["dt_range"] == [0.001, 0.1]


class TestMambaBlock:
    def test_init(self):
        from algorithms.mamba import MambaBlock
        block = MambaBlock(d_model=32, d_state=8, n_layers=2)
        assert block.d_model == 32
        assert block.n_layers == 2

    def test_forward(self):
        from algorithms.mamba import MambaBlock
        block = MambaBlock(d_model=16, d_state=4, n_layers=1)
        X = np.random.randn(3, 8, 16)
        out = block.forward(X)
        assert out.shape == (3, 8, 16)

    def test_fit_predict(self):
        from algorithms.mamba import MambaBlock
        np.random.seed(42)
        block = MambaBlock(d_model=16, d_state=8, n_layers=2)
        X = np.random.randn(10, 8, 16)
        y = np.random.randn(10, 1)
        result = block.fit(X, y, epochs=5, lr=1e-2)
        assert result["status"] == "success"
        pred = block.predict(X)
        assert pred.shape[0] >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
