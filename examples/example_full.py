# -*- coding: utf-8 -*-
"""完整示例：数学建模竞赛典型问题求解"""
import numpy as np
import sys
sys.path.insert(0, '..')

from algorithms.optimization import LinearProgramming
from algorithms.ga import GeneticAlgorithm
from algorithms.grey_model import GM11
from algorithms.topsis import TOPSIS
from algorithms.entropy_weight import EntropyWeight
from algorithms.pca import PCA
from algorithms.ensemble import RegressionEnsemble, auto_select_models
from algorithms.timeseries import TimeSeriesDecomposition
from algorithms.image import ImageProcessing, EdgeDetection, ImageSegmentation


def example_optimization():
    """优化问题示例"""
    print("\n" + "="*60)
    print("示例1: 资源分配优化")
    print("="*60)
    
    # 线性规划
    c = np.array([-3, -2])
    A_ub = np.array([[2, 1], [1, 3]])
    b_ub = np.array([20, 30])
    
    lp = LinearProgramming()
    result = lp.solve(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)])
    print(f"最优解: x={result['optimal_solution']}, 最优值: {result['optimal_value']}")
    
    # 遗传算法
    def sphere(x):
        return np.sum(x**2)
    
    ga = GeneticAlgorithm(pop_size=50, max_gen=100)
    result = ga.optimize(sphere, [(-5, 5)]*3, is_maximization=False)
    print(f"GA最优值: {result['optimal_value']:.6f}")


def example_prediction():
    """预测问题示例"""
    print("\n" + "="*60)
    print("示例2: 时间序列预测")
    print("="*60)
    
    # GM(1,1)预测
    data = np.array([4.87, 5.38, 5.94, 6.54, 7.05, 7.62, 8.18, 8.72])
    gm = GM11()
    result = gm.fit_predict(data, steps=3)
    print(f"预测值: {[round(x, 3) for x in result['predicted_values']]}")
    print(f"精度: {result['accuracy']['等级']}")
    
    # 时间序列分解
    t = np.arange(120)
    series = 0.1 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 0.5
    
    decomposer = TimeSeriesDecomposition(period=12)
    decomp = decomposer.fit(series).decompose()
    
    forecast = decomposer.forecast(steps=12)
    print(f"未来12期预测范围: [{forecast['forecast'][0]:.2f}, {forecast['forecast'][-1]:.2f}]")


def example_evaluation():
    """评价问题示例"""
    print("\n" + "="*60)
    print("示例3: 多方案综合评价")
    print("="*60)
    
    # 评价数据
    data = np.array([
        [95, 88, 92, 85, 78],
        [82, 75, 80, 90, 85],
        [78, 82, 75, 70, 92],
        [88, 90, 88, 82, 75],
        [75, 70, 78, 88, 88],
        [92, 85, 90, 78, 80]
    ])
    
    # 熵权法
    entropy = EntropyWeight()
    ew_result = entropy.evaluate(data)
    weights = np.array(ew_result['weights'])
    print(f"权重: {[round(w, 4) for w in weights]}")
    
    # TOPSIS
    topsis = TOPSIS(weights=weights)
    tp_result = topsis.evaluate(data, types=['benefit']*5)
    
    print("\n排名结果:")
    for scheme_idx, score, rank in tp_result['score_ranking']:
        print(f"  方案{scheme_idx}: 得分={score:.4f}, 排名={rank}")
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    print(f"PCA降维后形状: {pca_result.shape}")


def example_image():
    """图像处理示例"""
    print("\n" + "="*60)
    print("示例4: 图像处理")
    print("="*60)
    
    # 生成测试图像
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # 灰度化
    gray = ImageProcessing.grayscale(img)
    print(f"灰度图形状: {gray.shape}")
    
    # 边缘检测
    gx, gy, magnitude = EdgeDetection.sobel(gray)
    print(f"Sobel边缘检测完成，梯度幅值形状: {magnitude.shape}")
    
    # 阈值分割
    seg = ImageSegmentation.threshold_segmentation(gray, method='otsu')
    print(f"Otsu分割完成，二值图形状: {seg.shape}")


def example_ensemble():
    """集成预测示例"""
    print("\n" + "="*60)
    print("示例5: 回归集成预测")
    print("="*60)
    
    # 生成数据
    np.random.seed(42)
    X = np.arange(30).reshape(-1, 1)
    y = 2 * X.flatten() + 5 + np.random.randn(30) * 2
    
    # 自动选择模型
    ensemble = auto_select_models(y, max_degree=2)
    result = ensemble.fit_predict(X, y)
    
    print(f"集成预测MSE: {result['MSE']:.4f}")
    print(f"集成预测R²: {result['R2']:.4f}")
    print(f"包含模型: {ensemble.model_names}")


if __name__ == "__main__":
    print("="*60)
    print("Mathematical Modeling Competition - Complete Examples")
    print("="*60)
    
    example_optimization()
    example_prediction()
    example_evaluation()
    example_image()
    example_ensemble()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
