import sys
"""
Tests for XGBoost Ensemble and GNN Improved algorithms
"""
import os
sys.path.insert(0, r'D:\\本地的知识库构建\\math-modeling-competition')
import unittest
import numpy as np


class TestXGBoost(unittest.TestCase):
    def test_init(self):
        from algorithms.xgboost_ensemble import XGBoostRegressor
        xgb = XGBoostRegressor(n_estimators=50, max_depth=4)
        self.assertEqual(xgb.n_estimators, 50)

    def test_fit_predict(self):
        from algorithms.xgboost_ensemble import XGBoostRegressor
        np.random.seed(42)
        xgb = XGBoostRegressor(n_estimators=20, max_depth=3, learning_rate=0.1)
        X = np.random.randn(30, 3)
        y = 2 * X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(30) * 0.1
        result = xgb.fit(X, y)
        self.assertEqual(result["status"], "success")
        pred = xgb.predict(X)
        self.assertEqual(len(pred), 30)

    def test_feature_importance(self):
        from algorithms.xgboost_ensemble import XGBoostRegressor
        np.random.seed(42)
        xgb = XGBoostRegressor(n_estimators=10, max_depth=3)
        X = np.random.randn(20, 3)
        y = X[:, 0] + np.random.randn(20) * 0.1
        xgb.fit(X, y)
        imp = xgb.feature_importance(X, y)
        self.assertEqual(len(imp), 3)
        self.assertAlmostEqual(imp.sum(), 1.0, places=5)

    def test_gradient_boosting(self):
        from algorithms.xgboost_ensemble import GradientBoostingEnsemble
        np.random.seed(42)
        gb = GradientBoostingEnsemble(n_estimators=15, max_depth=2)
        X = np.random.randn(20, 2)
        y = X[:, 0] + X[:, 1] + np.random.randn(20) * 0.05
        result = gb.fit(X, y)
        self.assertEqual(result["status"], "success")
        pred = gb.predict(X)
        self.assertEqual(len(pred), 20)

    def test_get_params(self):
        from algorithms.xgboost_ensemble import XGBoostRegressor
        xgb = XGBoostRegressor(reg_alpha=0.5, reg_lambda=2.0)
        params = xgb.get_params()
        self.assertEqual(params["reg_alpha"], 0.5)


class TestImprovedGCN(unittest.TestCase):
    def test_init(self):
        from algorithms.gnn_improved import ImprovedGCN
        gcn = ImprovedGCN(n_features=4, n_hidden=16, n_classes=1)
        self.assertEqual(gcn.n_features, 4)

    def test_forward(self):
        from algorithms.gnn_improved import ImprovedGCN
        gcn = ImprovedGCN(n_features=3, n_hidden=8, n_classes=1)
        X = np.random.randn(5, 3)
        A = np.random.rand(5, 5)
        A = (A + A.T) / 2
        A[A > 0.5] = 1
        A[A <= 0.5] = 0
        np.fill_diagonal(A, 1)
        pred = gcn.forward(X, A)
        self.assertEqual(len(pred), 5)

    def test_fit_predict(self):
        from algorithms.gnn_improved import ImprovedGCN
        np.random.seed(42)
        gcn = ImprovedGCN(n_features=3, n_hidden=8, n_classes=1, dropout=0.0)
        X = np.random.randn(6, 3)
        A = np.eye(6) + np.random.rand(6, 6) * 0.3
        A = (A + A.T) / 2
        A[A > 0.5] = 1
        A[A <= 0.5] = 0
        np.fill_diagonal(A, 1)
        y = np.random.randn(6, 1)
        result = gcn.fit(X, A, y, epochs=5, lr=1e-2)
        self.assertEqual(result["status"], "success")
        pred = gcn.predict(X, A)
        self.assertEqual(len(pred), 6)


class TestSpatialTemporalGNN(unittest.TestCase):
    def test_init(self):
        from algorithms.gnn_improved import SpatialTemporalGNN
        stgnn = SpatialTemporalGNN(n_nodes=5, n_features=2, seq_len=12)
        self.assertEqual(stgnn.n_nodes, 5)

    def test_forward(self):
        from algorithms.gnn_improved import SpatialTemporalGNN
        stgnn = SpatialTemporalGNN(n_nodes=4, n_features=2, seq_len=6, hidden_dim=8)
        X = np.random.randn(3, 6, 4, 2)
        pred = stgnn.forward(X)
        self.assertEqual(len(pred), 3)

    def test_fit_predict(self):
        from algorithms.gnn_improved import SpatialTemporalGNN
        np.random.seed(42)
        stgnn = SpatialTemporalGNN(n_nodes=4, n_features=2, seq_len=6, hidden_dim=8)
        X = np.random.randn(10, 6, 4, 2)
        y = np.random.randn(10, 1)
        result = stgnn.fit(X, y, epochs=5, lr=1e-2)
        self.assertEqual(result["status"], "success")
        pred = stgnn.predict(X)
        self.assertEqual(len(pred), 10)


if __name__ == "__main__":
    unittest.main()
