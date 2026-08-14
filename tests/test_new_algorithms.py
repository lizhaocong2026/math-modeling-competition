"""
新增算法测试
"""
import unittest
import numpy as np
import sys
sys.path.insert(0, '..')


class TestRandomForest(unittest.TestCase):
    """测试随机森林"""
    
    def test_regression(self):
        from algorithms.random_forest import RandomForest
        
        X = np.random.randn(50, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(50) * 0.1
        
        rf = RandomForest(n_trees=10, max_depth=5)
        result = rf.fit_predict(X, y)
        
        self.assertIn('R2', result)
        self.assertGreater(result['R2'], 0.8)


class TestSVM(unittest.TestCase):
    """测试SVM"""
    
    def test_classification(self):
        from algorithms.svm import SVM
        
        X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        svm = SVM()
        result = svm.fit_predict(X, y)
        
        self.assertIn('accuracy', result)
        self.assertGreaterEqual(result['accuracy'], 0.5)


class TestBayesian(unittest.TestCase):
    """测试贝叶斯推断"""
    
    def test_conjugate_normal(self):
        from algorithms.bayesian import ConjugateBayes
        
        data = np.array([1.2, 1.5, 1.3, 1.6, 1.4])
        result = ConjugateBayes.normal_normal(data)
        
        self.assertIn('posterior_mean', result)
        self.assertIn('credible_interval_95', result)
    
    def test_beta_binomial(self):
        from algorithms.bayesian import ConjugateBayes
        
        result = ConjugateBayes.beta_binomial(successes=7, trials=10)
        
        self.assertIn('posterior_alpha', result)
        self.assertAlmostEqual(result['mean'], 0.7, places=1)


class TestGraph(unittest.TestCase):
    """测试图算法"""
    
    def test_dijkstra(self):
        from algorithms.graph import Graph
        
        g = Graph(4)
        g.add_edge(0, 1, 1)
        g.add_edge(0, 2, 4)
        g.add_edge(1, 2, 2)
        g.add_edge(1, 3, 5)
        g.add_edge(2, 3, 1)
        
        dist, prev = g.dijkstra(0)
        
        self.assertEqual(dist[0], 0)
        self.assertLess(dist[1], np.inf)
        self.assertLess(dist[3], np.inf)


class TestKalmanFilter(unittest.TestCase):
    """测试卡尔曼滤波"""
    
    def test_filtering(self):
        from algorithms.state_space import KalmanFilter
        
        # 简单状态空间模型
        A = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        
        kf = KalmanFilter(A, np.zeros((2, 1)), H, Q, R)
        kf.initialize(np.zeros(2), np.eye(2) * 10)
        
        observations = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = kf.filter(observations)
        
        self.assertEqual(result['filtered_states'].shape[0], 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)