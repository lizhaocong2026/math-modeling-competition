"""
测试模块 - 验证算法正确性
"""
import unittest
import numpy as np
import sys
sys.path.insert(0, '..')


class TestOptimization(unittest.TestCase):
    """优化算法测试"""
    
    def test_linear_programming(self):
        """测试线性规划"""
        from algorithms.optimization import LinearProgramming
        
        lp = LinearProgramming()
        c = np.array([-3, -2])
        A_ub = np.array([[2, 1], [1, 3]])
        b_ub = np.array([20, 30])
        bounds = [(0, None), (0, None)]
        
        result = lp.solve(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        
        self.assertTrue(result['success'])
        self.assertLessEqual(result['optimal_value'], 0)  # 应为负值（最小化负利润）
    
    def test_genetic_algorithm(self):
        """测试遗传算法"""
        from algorithms.ga import GeneticAlgorithm
        
        def sphere(x):
            return np.sum(x ** 2)
        
        ga = GeneticAlgorithm(pop_size=50, max_gen=100)
        result = ga.optimize(sphere, [(-5, 5)] * 3, is_maximization=False)
        
        self.assertTrue(result['success'])
        self.assertLess(result['optimal_value'], 1.0)  # 应接近0


class TestPrediction(unittest.TestCase):
    """预测算法测试"""
    
    def test_grey_model(self):
        """测试灰色预测"""
        from algorithms.grey_model import GM11
        
        data = np.array([4.87, 5.38, 5.94, 6.54, 7.05])
        gm = GM11()
        result = gm.fit_predict(data, steps=2)
        
        self.assertIn('fitted_values', result)
        self.assertIn('predicted_values', result)
        self.assertEqual(len(result['predicted_values']), 2)
    
    def test_linear_regression(self):
        """测试线性回归"""
        from algorithms.linear_regression import LinearRegression
        
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 5, 4, 5])
        
        lr = LinearRegression()
        result = lr.fit_predict(X, y)
        
        self.assertIn('coefficients', result)
        self.assertIn('r_squared', result)
        self.assertGreaterEqual(result['r_squared'], 0)


class TestEvaluation(unittest.TestCase):
    """评价算法测试"""
    
    def test_topsis(self):
        """测试TOPSIS"""
        from algorithms.topsis import TOPSIS
        
        data = np.array([
            [85, 90, 88],
            [92, 85, 90],
            [78, 82, 75]
        ])
        
        topsis = TOPSIS()
        result = topsis.evaluate(data)
        
        self.assertIn('scores', result)
        self.assertIn('rankings', result)
        self.assertEqual(len(result['scores']), 3)
    
    def test_ahp(self):
        """测试AHP"""
        from algorithms.ahp import AHP
        
        matrix = np.array([
            [1, 3, 5],
            [1/3, 1, 2],
            [1/5, 1/2, 1]
        ])
        
        ahp = AHP()
        result = ahp.compare(matrix)
        
        self.assertIn('weights', result)
        self.assertIn('consistency_ratio', result)
        self.assertLess(result['consistency_ratio'], 0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
