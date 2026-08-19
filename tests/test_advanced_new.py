"""
Tests for GA improved, SA improved, CNN-LSTM, TCN, Bayesian Optimization
"""
import unittest
import numpy as np


class TestAdaptiveGA(unittest.TestCase):
    def test_init(self):
        from algorithms.ga_improved import AdaptiveGeneticAlgorithm
        ga = AdaptiveGeneticAlgorithm(pop_size=50, max_gen=100)
        self.assertEqual(ga.pop_size, 50)

    def test_optimize_single_obj(self):
        from algorithms.ga_improved import AdaptiveGeneticAlgorithm
        np.random.seed(42)
        ga = AdaptiveGeneticAlgorithm(pop_size=30, max_gen=50)
        def obj(x):
            return -(x[0]**2 + x[1]**2)
        result = ga.optimize(obj, bounds=[(-5, 5), (-5, 5)])
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["optimal_solution"])

    def test_get_params(self):
        from algorithms.ga_improved import AdaptiveGeneticAlgorithm
        ga = AdaptiveGeneticAlgorithm(crossover_rates=(0.7, 0.9))
        params = ga.get_params()
        self.assertEqual(params["crossover_range"], [0.7, 0.9])


class TestNSGA2Improved(unittest.TestCase):
    def test_init(self):
        from algorithms.ga_improved import NSGA2Improved
        nsga = NSGA2Improved(n_objectives=2, pop_size=50, max_gen=30)
        self.assertEqual(nsga.n_objectives, 2)

    def test_optimize_biobj(self):
        from algorithms.ga_improved import NSGA2Improved
        nsga = NSGA2Improved(n_objectives=2, pop_size=20, max_gen=15, n_vars=3)
        def f1(x):
            return x[0]**2 + x[1]**2
        def f2(x):
            return (x[0] - 2)**2 + (x[1] - 2)**2
        result = nsga.optimize([f1, f2])
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["n_solutions"], 0)

    def test_get_params(self):
        from algorithms.ga_improved import NSGA2Improved
        nsga = NSGA2Improved(n_objectives=3, n_vars=5)
        params = nsga.get_params()
        self.assertEqual(params["n_objectives"], 3)


class TestSimulatedAnnealing(unittest.TestCase):
    def test_init(self):
        from algorithms.sa_improved import SimulatedAnnealing
        sa = SimulatedAnnealing(initial_temp=100, cooling_rate=0.9)
        self.assertEqual(sa.initial_temp, 100)

    def test_optimize(self):
        from algorithms.sa_improved import SimulatedAnnealing
        sa = SimulatedAnnealing(initial_temp=100, min_temp=1e-4, cooling_rate=0.95, max_iter=500, restarts=2)
        def obj(x):
            return -(x[0]**2 + x[1]**2)
        result = sa.optimize(obj, bounds=[(-5, 5), (-5, 5)], is_maximization=False)
        self.assertEqual(result["status"], "success")

    def test_tabu_search(self):
        from algorithms.sa_improved import TabuSearch
        ts = TabuSearch(max_iter=200, tabu_size=20)
        def obj(x):
            return -(x[0]**2 + x[1]**2)
        result = ts.optimize(obj, bounds=[(-3, 3), (-3, 3)])
        self.assertEqual(result["status"], "success")

    def test_get_params(self):
        from algorithms.sa_improved import SimulatedAnnealing
        sa = SimulatedAnnealing(cooling_schedule="linear")
        params = sa.get_params()
        self.assertEqual(params["cooling_schedule"], "linear")


class TestCNNLSTM(unittest.TestCase):
    def test_init(self):
        from algorithms.cnn_lstm import CNNLSTM
        model = CNNLSTM(seq_len=24, n_features=2, cnn_filters=16, lstm_units=32)
        self.assertEqual(model.seq_len, 24)

    def test_forward(self):
        from algorithms.cnn_lstm import CNNLSTM
        model = CNNLSTM(seq_len=12, n_features=1, cnn_filters=8, lstm_units=16)
        X = np.random.randn(4, 12, 1)
        pred = model.forward(X)
        self.assertEqual(pred.shape, (4,))

    def test_fit_predict(self):
        from algorithms.cnn_lstm import CNNLSTM
        np.random.seed(42)
        model = CNNLSTM(seq_len=12, n_features=1, cnn_filters=8, lstm_units=16)
        X = np.random.randn(16, 12, 1)
        y = np.sin(np.arange(12)).reshape(1, -1) * np.random.randn(16, 1)
        result = model.fit(X, y, epochs=5, lr=1e-2)
        self.assertEqual(result["status"], "success")
        pred = model.predict(X)
        self.assertEqual(pred.shape, (16,))

    def test_get_params(self):
        from algorithms.cnn_lstm import CNNLSTM
        model = CNNLSTM(cnn_filters=64, dense_units=128)
        params = model.get_params()
        self.assertEqual(params["cnn_filters"], 64)


class TestTCN(unittest.TestCase):
    def test_init(self):
        from algorithms.cnn_lstm import TCN
        tcn = TCN(seq_len=24, n_features=1, n_filters=16, n_layers=2)
        self.assertEqual(tcn.n_layers, 2)

    def test_forward(self):
        from algorithms.cnn_lstm import TCN
        tcn = TCN(seq_len=12, n_features=2, n_filters=8, n_layers=2)
        X = np.random.randn(4, 12, 2)
        pred = tcn.forward(X)
        self.assertEqual(pred.shape, (4,))

    def test_fit_predict(self):
        from algorithms.cnn_lstm import TCN
        np.random.seed(42)
        tcn = TCN(seq_len=12, n_features=1, n_filters=8, n_layers=2)
        X = np.random.randn(16, 12, 1)
        y = np.random.randn(16, 1)
        result = tcn.fit(X, y, epochs=5, lr=1e-2)
        self.assertEqual(result["status"], "success")
        pred = tcn.predict(X)
        self.assertEqual(pred.shape, (16,))

    def test_get_params(self):
        from algorithms.cnn_lstm import TCN
        tcn = TCN(n_filters=64, n_layers=4)
        params = tcn.get_params()
        self.assertEqual(params["n_layers"], 4)


class TestBayesianOptimization(unittest.TestCase):
    def test_init(self):
        from algorithms.bayesian_opt import BayesianOptimization
        bo = BayesianOptimization(bounds=[(-5, 5), (-5, 5)], n_init=5, n_iter=10)
        self.assertEqual(bo.n_dims, 2)

    def test_optimize(self):
        from algorithms.bayesian_opt import BayesianOptimization
        bo = BayesianOptimization(bounds=[(-3, 3)], n_init=5, n_iter=10)
        def obj(x):
            return -(x[0] - 1)**2 + 2
        result = bo.optimize(obj, is_maximization=True)
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["optimal_solution"])

    def test_get_params(self):
        from algorithms.bayesian_opt import BayesianOptimization
        bo = BayesianOptimization(bounds=[(0, 10)], acquisition="ei")
        params = bo.get_params()
        self.assertEqual(params["acquisition"], "ei")


class TestStackingEnsemble(unittest.TestCase):
    def test_init(self):
        from algorithms.bayesian_opt import StackingEnsemble
        ens = StackingEnsemble()
        self.assertTrue(hasattr(ens, "base_fits"))
        self.assertTrue(isinstance(ens.base_fits, dict))
        self.assertEqual(ens.n_valid_models, 0)

    def test_fit_predict(self):
        from algorithms.bayesian_opt import StackingEnsemble
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(50) * 0.1
        ens = StackingEnsemble()
        result = ens.fit(X, y)
        self.assertEqual(result["status"], "success")
        pred = ens.predict(X)
        self.assertEqual(len(pred), 50)

    def test_evaluate(self):
        from algorithms.bayesian_opt import StackingEnsemble
        np.random.seed(42)
        X = np.random.randn(60, 3)
        y = X[:, 0] + X[:, 1] + np.random.randn(60) * 0.05
        ens = StackingEnsemble()
        ens.fit(X[:50], y[:50])
        metrics = ens.evaluate(X[50:], y[50:])
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("R2", metrics)
        self.assertGreater(metrics["R2"], -1)

    def test_get_params(self):
        from algorithms.bayesian_opt import StackingEnsemble
        ens = StackingEnsemble()
        params = ens.get_params()
        self.assertIsInstance(params, dict)
        self.assertIn("base_models", params)


if __name__ == "__main__":
    unittest.main()
