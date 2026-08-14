"""
综合示例：完整的数学建模解题流程
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from algorithms.optimization import LinearProgramming
from algorithms.grey_model import GM11
from algorithms.topsis import TOPSIS
from algorithms.entropy_weight import EntropyWeight
from utils.data_preprocessor import DataPreprocessor
from utils.helpers import Utils


def example_complete_workflow():
    """
    完整解题工作流示例
    
    模拟CUMCM典型问题：资源分配与效果评价
    """
    print("=" * 70)
    print("数学建模竞赛完整解题工作流示例")
    print("=" * 70)
    
    # ========== 第一步：数据预处理 ==========
    print("\n【步骤1】数据预处理")
    print("-" * 50)
    
    # 模拟原始数据（含缺失值和异常值）
    raw_data = np.array([
        [95, 88, 92, 85, np.nan],
        [82, np.nan, 80, 90, 85],
        [78, 82, 75, np.nan, 92],
        [88, 90, 88, 82, 75],
        [75, 70, 78, 88, 88],
        [92, 85, 90, 78, 80]
    ])
    
    preprocessor = DataPreprocessor()
    result = preprocessor.process(raw_data, fill_missing=True, remove_outliers=False, normalize=True)
    
    clean_data = result['processed_data']
    print(f"原始数据形状: {raw_data.shape}")
    print(f"清洗后数据形状: {clean_data.shape}")
    print(f"归一化方法: {result.get('norm_method', 'N/A')}")
    
    # ========== 第二步：客观赋权 ==========
    print("\n【步骤2】熵权法客观赋权")
    print("-" * 50)
    
    entropy = EntropyWeight()
    weight_result = entropy.evaluate(clean_data)
    
    weights = np.array(weight_result['weights'])
    print(f"各指标权重: {[round(w, 4) for w in weights]}")
    print(f"差异系数: {[round(d, 4) for d in weight_result['differences']]}")
    
    # ========== 第三步：TOPSIS评价 ==========
    print("\n【步骤3】TOPSIS综合评价")
    print("-" * 50)
    
    topsis = TOPSIS(weights=weights)
    eval_result = topsis.evaluate(clean_data, types=['benefit'] * 5)
    
    print(f"{'方案':<8} {'得分':<12} {'排名':<8}")
    print("-" * 35)
    for scheme_idx, score, rank in eval_result['score_ranking']:
        print(f"方案{scheme_idx:<6} {score:<12.6f} {rank:<8}")
    
    # ========== 第四步：优化问题求解 ==========
    print("\n【步骤4】资源分配优化")
    print("-" * 50)
    
    # 问题：在预算约束下最大化综合效益
    # 目标：max sum(w_i * x_i)
    # 约束：sum(x_i) <= 100, 0 <= x_i <= 30
    
    c = -weights  # 最小化负效益 = 最大化正效益
    A_ub = np.array([[1, 1, 1, 1, 1]])  # sum(x) <= 100
    b_ub = np.array([100])
    bounds = [(0, 30)] * 5
    
    lp = LinearProgramming()
    opt_result = lp.solve(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    print(f"最优分配方案: {[round(x, 2) for x in opt_result['optimal_solution']]}")
    print(f"最大综合效益: {-opt_result['optimal_value']:.6f}")
    
    # ========== 第五步：预测未来趋势 ==========
    print("\n【步骤5】灰色预测未来趋势")
    print("-" * 50)
    
    # 使用各方案的综合得分进行预测
    scores = np.array(eval_result['scores'])
    
    gm = GM11()
    pred_result = gm.fit_predict(scores, steps=3)
    
    print(f"当前得分: {[round(s, 4) for s in scores]}")
    print(f"预测未来3年: {[round(p, 4) for p in pred_result['predicted_values']]}")
    print(f"模型精度: {pred_result['accuracy']['等级']}")
    
    # ========== 第六步：结果汇总 ==========
    print("\n【步骤6】结果汇总")
    print("-" * 50)
    
    final_report = {
        "timestamp": str(Utils.__class__.__dict__.get('now', 'N/A')),
        "problem_type": "资源分配与评价",
        "data_preprocessing": {
            "original_shape": list(raw_data.shape),
            "cleaned_shape": list(clean_data.shape)
        },
        "entropy_weights": {f"指标{i+1}": round(w, 4) for i, w in enumerate(weights)},
        "topsis_evaluation": {
            f"方案{i+1}": {
                "score": round(eval_result['scores'][i], 6),
                "rank": int(eval_result['rankings'][i])
            }
            for i in range(len(eval_result['scores']))
        },
        "optimization_result": {
            "allocation": [round(x, 2) for x in opt_result['optimal_solution']],
            "max_benefit": round(-opt_result['optimal_value'], 6)
        },
        "prediction": {
            "fitted": [round(v, 4) for v in pred_result['fitted_values']],
            "forecasted": [round(v, 4) for v in pred_result['predicted_values']],
            "accuracy": pred_result['accuracy']['等级']
        }
    }
    
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    
    # 保存结果
    Utils.save_results(final_report, "results_final.json")
    print("\n结果已保存到: results_final.json")
    
    return final_report


if __name__ == "__main__":
    import json
    result = example_complete_workflow()
    
    print("\n" + "=" * 70)
    print("解题流程完成！")
    print("=" * 70)
