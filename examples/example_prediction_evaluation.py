"""
示例2: 预测与评价问题 (CUMCM常见B题、C题类型)
结合灰色预测、回归分析和综合评价
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from algorithms.grey_model import GM11
from algorithms.linear_regression import LinearRegression
from algorithms.polynomial_regression import PolynomialRegression
from algorithms.pca import PCA
from algorithms.ahp import AHP
from algorithms.topcis import TOPSIS
from algorithms.entropy_weight import EntropyWeight
from templates.prediction_template import PredictionProblemTemplate
from templates.evaluation_template import EvaluationProblemTemplate


def example_grey_prediction():
    """示例：GM(1,1)灰色预测"""
    print("\n" + "="*60)
    print("示例1: GM(1,1)灰色预测")
    print("="*60)
    
    # 模拟GDP数据（万亿元）
    gdp_data = np.array([41.3, 47.3, 54.0, 59.5, 64.4, 74.6, 83.2, 91.9, 99.1, 101.6])
    
    gm = GM11()
    result = gm.fit_predict(gdp_data, steps=5)
    
    print("\n【结果】")
    print(f"发展系数 a = {result['model_params']['发展系数_a']:.6f}")
    print(f"灰作用量 b = {result['model_params']['灰作用量_b']:.6f}")
    print(f"拟合值: {[round(x, 2) for x in result['fitted_values']]}")
    print(f"预测值(未来5年): {[round(x, 2) for x in result['predicted_values']]}")
    print(f"模型精度: {result['accuracy']['等级']}")
    
    return result


def example_regression_forecast():
    """示例：回归预测"""
    print("\n" + "="*60)
    print("示例2: 回归预测模型对比")
    print("="*60)
    
    # 模拟人口数据（万人）
    population = np.array([1328, 1334, 1341, 1347, 1354, 1360, 1367, 1373, 1378, 1383])
    
    template = PredictionProblemTemplate()
    
    # 对比多种预测模型
    results = template.compare_prediction_models(population, test_ratio=0.2)
    
    print("\n【模型选择建议】")
    best_model = min(results.items(), key=lambda x: x[1].get('RMSE', float('inf')))
    print(f"推荐使用: {best_model[0]} (RMSE={best_model[1].get('RMSE', 'N/A')})")
    
    return results


def example_ahp_evaluation():
    """示例：AHP层次分析评价"""
    print("\n" + "="*60)
    print("示例3: AHP层次分析法评价")
    print("="*60)
    
    # 构建判断矩阵（5个评价指标）
    comparison_matrix = np.array([
        [1.0, 3.0, 5.0, 7.0, 9.0],
        [1.0/3.0, 1.0, 3.0, 5.0, 7.0],
        [1.0/5.0, 1.0/3.0, 1.0, 3.0, 5.0],
        [1.0/7.0, 1.0/5.0, 1.0/3.0, 1.0, 3.0],
        [1.0/9.0, 1.0/7.0, 1.0/5.0, 1.0/3.0, 1.0]
    ])
    
    ahp = AHP()
    result = ahp.compare(comparison_matrix)
    
    print("\n【结果】")
    print(f"各指标权重: {result['weights']}")
    print(f"一致性比率 CR: {result['consistency_ratio']:.6f}")
    print(f"一致性检验: {result['consistency_level']}")
    
    return result


def example_topsis_evaluation():
    """示例：TOPSIS评价"""
    print("\n" + "="*60)
    print("示例4: TOPSIS综合评价")
    print("="*60)
    
    # 模拟6个城市的发展水平评价数据
    # 指标：GDP、人均收入、教育水平、医疗水平、环境指数
    data = np.array([
        [95, 88, 92, 85, 78],  # 城市A
        [82, 75, 80, 90, 85],  # 城市B
        [78, 82, 75, 70, 92],  # 城市C
        [88, 90, 88, 82, 75],  # 城市D
        [75, 70, 78, 88, 88],  # 城市E
        [92, 85, 90, 78, 80]   # 城市F
    ])
    
    # 所有指标均为效益型（越大越好）
    types = ['benefit'] * 5
    
    topsis = TOPSIS()
    result = topsis.evaluate(data, types=types)
    
    print("\n【结果】")
    print(f"{'城市':<6} {'得分':<10} {'排名':<8}")
    print("-" * 30)
    for i, (score, rank) in enumerate(zip(result['scores'], result['rankings'])):
        print(f"城市{chr(65+i):<4} {score:<10.6f} {rank:<8}")
    
    return result


def example_entropy_weight():
    """示例：熵权法赋权"""
    print("\n" + "="*60)
    print("示例5: 熵权法客观赋权")
    print("="*60)
    
    # 与TOPSIS示例相同的数据
    data = np.array([
        [95, 88, 92, 85, 78],
        [82, 75, 80, 90, 85],
        [78, 82, 75, 70, 92],
        [88, 90, 88, 82, 75],
        [75, 70, 78, 88, 88],
        [92, 85, 90, 78, 80]
    ])
    
    entropy = EntropyWeight()
    result = entropy.evaluate(data)
    
    print("\n【结果】")
    print(f"各指标权重: {result['weights']}")
    print(f"各指标熵值: {result['entropies']}")
    
    return result


def example_pca():
    """示例：PCA主成分分析"""
    print("\n" + "="*60)
    print("示例6: PCA主成分分析")
    print("="*60)
    
    # 模拟高维数据（10个样本，8个特征）
    np.random.seed(42)
    data = np.random.randn(10, 8) * 10 + 50
    
    pca = PCA(n_components=3)
    result = pca.transform_with_details(data)
    
    print("\n【结果】")
    print(f"降维后形状: {data.shape} -> {result['transformed'].shape}")
    print(f"解释方差比例: {result['explained_variance_ratio']}")
    print(f"累积解释方差: {result['cumulative_explained_variance']}")
    
    return result


if __name__ == "__main__":
    example_grey_prediction()
    example_regression_forecast()
    example_ahp_evaluation()
    example_topsis_evaluation()
    example_entropy_weight()
    example_pca()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
