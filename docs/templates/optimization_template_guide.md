# 优化类模板用法指南

> **仓库文件**：	emplates/optimization_template.py  
> **适用题型**：CUMCM A题（优化调度类）  
> **关联算法**：ga.py · pso.py · aco.py · nsga2.py · optimization.py · convex_opt.py

---

## 一、快速开始

`python
from templates.optimization_template import OptimizationProblemTemplate
import numpy as np

template = OptimizationProblemTemplate()

# 线性规划求解
result = template.solve_linear_programming(
    objective_coeffs=np.array([2, 3, 1]),
    constraints={
        'A_eq': np.array([[1, 1, 1]]),
        'b_eq': np.array([10]),
        'A_ub': np.array([[1, 2, 3]]),
        'b_ub': np.array([15])
    },
    bounds=[(0, None), (0, None), (0, 5)]
)
print(result)
`

## 二、完整解题流程

| 步骤 | 方法 | 说明 |
|------|------|------|
| 1. 数据预处理 | preprocess() | 缺失值填充 + 标准化 |
| 2. 模型构建 | uild_model() | 定义目标函数和约束 |
| 3. 算法选择 | select_algorithm() | 根据问题类型自动推荐 |
| 4. 求解 | solve() | 执行优化求解 |
| 5. 灵敏度分析 | sensitivity_analysis() | 参数扰动检验 |
| 6. 结果可视化 | isualize() | Pareto前沿/收敛曲线 |

## 三、参数调优建议

| 算法 | 关键参数 | 推荐值 | 说明 |
|------|---------|--------|------|
| GA | pop_size | 100-200 | 种群大小 |
| GA | max_gen | 300-500 | 最大迭代数 |
| GA | mutation_rate | 0.01-0.1 | 变异概率 |
| PSO | w | 0.4-0.9 | 惯性权重 |
| PSO | c1, c2 | 1.5-2.0 | 学习因子 |
| NSGA-II | pop_size | 100-150 | 种群大小 |
| ACO | Q | 100-1000 | 信息素总量 |

## 四、常见问题

- **收敛慢**：调整 pop_size 和 max_gen，或改用 NSGA-II
- **陷入局部最优**：增大变异概率，或结合 SA 退火
- **约束冲突**：检查约束矩阵是否可行域为空
- **多目标冲突**：使用 NSGA-II 获取 Pareto 前沿

---
> **更新**：2026-08-18
