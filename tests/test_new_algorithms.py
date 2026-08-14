"""
扩展算法测试
"""
import unittest
import numpy as np
import sys
sys.path.insert(0, '..')


class TestImageProcessing(unittest.TestCase):
    """测试图像处理算法"""
    
    def test_grayscale(self):
        from algorithms.image import ImageProcessing
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        gray = ImageProcessing.grayscale(img)
        self.assertEqual(gray.shape, (50, 50))
    
    def test_sobel_edge_detection(self):
        from algorithms.image import EdgeDetection
        img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        gx, gy, magnitude = EdgeDetection.sobel(img)
        self.assertEqual(gx.shape, (50, 50))
        self.assertEqual(magnitude.shape, (50, 50))


class TestEnsemble(unittest.TestCase):
    """测试回归集成"""
    
    def test_ensemble_predict(self):
        from algorithms.ensemble import RegressionEnsemble
        from algorithms.linear_regression import LinearRegression
        
        X = np.arange(20).reshape(-1, 1)
        y = 2 * X.flatten() + np.random.randn(20) * 0.1
        
        ensemble = RegressionEnsemble()
        ensemble.add_model(LinearRegression())
        ensemble.fit(X, y)
        
        pred = ensemble.predict(X[:5])
        self.assertEqual(len(pred), 5)


class TestTimeSeries(unittest.TestCase):
    """测试时间序列分解"""
    
    def test_decomposition(self):
        from algorithms.timeseries import TimeSeriesDecomposition
        
        # 生成带趋势和季节性的数据
        t = np.arange(120)
        data = 0.1 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 0.5
        
        decomposer = TimeSeriesDecomposition(period=12)
        result = decomposer.fit(data).decompose()
        
        self.assertIn('trend', result)
        self.assertIn('seasonal', result)
        self.assertIn('residual', result)
        self.assertEqual(len(result['trend']), 120)


if __name__ == '__main__':
    unittest.main(verbosity=2)
