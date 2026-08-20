"""
测试扩展模块
"""
import os
import unittest
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')


class TestDE(unittest.TestCase):
    """测试差分进化算法"""
    
    def test_sphere_function(self):
        from algorithms.de import DifferentialEvolution
        
        def sphere(x):
            return np.sum(x ** 2)
        
        de = DifferentialEvolution(pop_size=30, max_gen=100)
        result = de.optimize(sphere, [(-5, 5)] * 3, is_maximization=False)
        
        self.assertTrue(result['success'])
        self.assertLess(result['optimal_value'], 0.1)


class TestACO(unittest.TestCase):
    """测试蚁群算法"""
    
    def test_tsp_small(self):
        from algorithms.aco import AntColony
        
        # 简单4城市TSP
        dist = np.array([
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ])
        
        aco = AntColony(n_ants=20, max_iter=50)
        result = aco.solve_tsp(dist)
        
        self.assertTrue(result['success'])
        self.assertIn('optimal_path', result)
        self.assertIn('optimal_length', result)


class TestMonteCarlo(unittest.TestCase):
    """测试蒙特卡洛模拟"""
    
    def test_pi_estimate(self):
        from algorithms.monte_carlo import MonteCarlo
        
        mc = MonteCarlo(seed=42)
        result = mc.estimate_pi(n_samples=100000)
        
        self.assertTrue(abs(result['pi_estimate'] - 3.14159) < 0.01)
    
    def test_integral(self):
        from algorithms.monte_carlo import MonteCarlo
        
        mc = MonteCarlo(seed=42)
        
        def f(x, y):
            return x**2 + y**2
        
        result = mc.estimate_integral(f, (0, 1, 0, 1), n_samples=50000)
        
        # 理论值: ∫∫(x²+y²)dxdy = 2/3 ≈ 0.667
        self.assertAlmostEqual(result['integral'], 0.667, places=1)


class TestCurveFitting(unittest.TestCase):
    """测试曲线拟合"""
    
    def test_linear_fit(self):
        from algorithms.curve_fitting import CurveFitting
        
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
        
        result = CurveFitting.linear_fit(x, y)
        
        self.assertIn('slope', result)
        self.assertIn('intercept', result)
        self.assertGreater(result['r_squared'], 0.9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
