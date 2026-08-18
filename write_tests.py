import os
base = "tests"

content = """# Tests for new algorithms: DEA, MarkovChain, KMeans, DBSCAN
import numpy as np
import pytest
from algorithms.dea import DEA
from algorithms.markov_chain import MarkovChain
from algorithms.kmeans import KMeans
from algorithms.db_scan import DBSCAN


class TestDEA:
    def test_basic_evaluation(self):
        X = np.array([[2, 3, 4, 5],
                      [1, 2, 3, 4]])
        Y = np.array([[4, 5, 6, 7],
                      [2, 3, 4, 5]])
        dea = DEA()
        results = dea.evaluate(X, Y)
        assert len(results) == 4
        for idx, eff in results.items():
            assert eff is not None
            assert 0 <= eff <= 1

    def test_classification(self):
        X = np.array([[2, 3, 4, 5],
                      [1, 2, 3, 4]])
        Y = np.array([[4, 5, 6, 7],
                      [2, 3, 4, 5]])
        dea = DEA()
        results = dea.evaluate(X, Y)
        classification = dea.classify_efficiency(results)
        for idx, cls in classification.items():
            assert cls in ["effective", "near_effective", "ineffective", "failed"]


class TestMarkovChain:
    def test_fit_and_predict(self):
        sequence = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        mc = MarkovChain()
        mc.fit(sequence)
        pred = mc.predict(steps=1)
        assert len(pred) == 2
        assert abs(pred.sum() - 1.0) < 1e-6

    def test_predict_sequence(self):
        sequence = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        mc = MarkovChain()
        mc.fit(sequence)
        seq = mc.predict_sequence(steps=10, seed=42)
        assert len(seq) == 10
        assert all(s in [0, 1] for s in seq)

    def test_stationary_distribution(self):
        sequence = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        mc = MarkovChain()
        mc.fit(sequence)
        stationary = mc.get_stationary_distribution()
        assert len(stationary) == 2
        assert abs(stationary.sum() - 1.0) < 1e-6


class TestKMeans:
    def test_fit(self):
        np.random.seed(42)
        X = np.random.rand(100, 2)
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(X)
        assert km.labels is not None
        assert len(km.labels) == 100
        assert km.inertia is not None
        assert km.inertia >= 0

    def test_predict(self):
        np.random.seed(42)
        X = np.random.rand(100, 2)
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(X)
        predictions = km.predict(X)
        assert len(predictions) == 100


class TestDBSCAN:
    def test_fit(self):
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + [0, 0],
            np.random.randn(50, 2) + [5, 5],
            np.random.randn(20, 2) + [10, 0]
        ])
        dbscan = DBSCAN(eps=1.0, min_samples=5)
        dbscan.fit(X)
        assert dbscan.labels is not None
        assert len(dbscan.labels) == 120
        n_clusters = dbscan.get_n_clusters()
        assert n_clusters >= 1
"""

with open(os.path.join(base, "test_new_algorithms_extended.py"), "w", encoding="utf-8") as f:
    f.write(content)
print("Created test_new_algorithms_extended.py")
