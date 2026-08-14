# 数学建模竞赛代码库 (Math Modeling Competition Toolkit)

> 针对全国大学生数学建模竞赛(CUMCM)和研究生数学建模竞赛定制的核心算法库，提供优化、预测、评价等模块的完整实现。

## 目录结构

```
math-modeling-competition/
├── core/                    # 核心包入口
│   └── __init__.py
├── algorithms/              # 算法实现模块
│   ├── optimization.py      # 线性/整数/非线性规划
│   ├── ga.py                # 遗传算法
│   ├── pso.py               # 粒子群优化
│   ├── sa.py                # 模拟退火
│   ├── grey_model.py        # GM(1,1)灰色预测
│   ├── ahp.py               # 层次分析法
│   ├── topsis.py            # TOPSIS评价
│   ├── entropy_weight.py    # 熵权法
│   ├── pca.py               # 主成分分析
│   ├── linear_regression.py # 线性回归
│   └── polynomial_regression.py # 多项式回归
├── templates/               # 竞赛模板
│   ├── optimization_template.py     # 优化类问题模板
│   ├── prediction_template.py       # 预测类问题模板
│   ├── evaluation_template.py       # 评价类问题模板
│   └── classification_clustering_template.py  # 分类聚类模板
├── utils/                   # 工具函数
│   ├── data_preprocessor.py  # 数据预处理
│   └── helpers.py            # 辅助函数
├── visualizations/          # 可视化
│   └── model_viz.py          # 模型可视化
├── examples/                # 示例脚本
│   ├── example_optimization.py
│   └── example_prediction_evaluation.py
├── tests/                   # 测试文件
├── data/                    # 示例数据
├── requirements.txt         # 依赖配置
└── README.md                # 本文件
```

## 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 基础使用

```python
import numpy as np
import sys
sys.path.insert(0, '.')

from algorithms.optimization import LinearProgramming
from algorithms.grey_model import GM11
from algorithms.topsis import TOPSIS

# 1. 线性规划求解
lp = LinearProgramming()
result = lp.solve(
    c=np.array([-3, -2]),           # 目标系数
    A_ub=np.array([[2, 1], [1, 3]]), # 约束矩阵
    b_ub=np.array([20, 30]),         # 约束向量
    bounds=[(0, None), (0, None)]
)
print(f"最优解: {result['optimal_solution']}")
print(f"最优值: {result['optimal_value']}")

# 2. 灰色预测
gm = GM11()
result = gm.fit_predict(np.array([4.87, 5.38, 5.94, 6.54]))
print(f"预测值: {result['predicted_values']}")

# 3. TOPSIS评价
topsis = TOPSIS()
result = topsis.evaluate(np.array([[85, 90, 88], [92, 85, 90]]))
print(f"得分: {result['scores']}")
```

## 算法说明

### 优化类算法

| 算法 | 适用场景 | 文件位置 |
|------|----------|----------|
| 线性规划 | 资源分配、生产计划 | `algorithms/optimization.py` |
| 整数规划 | 指派问题、背包问题 | `algorithms/optimization.py` |
| 非线性规划 | 复杂约束优化 | `algorithms/optimization.py` |
| 遗传算法 | 全局优化、NP难题 | `algorithms/ga.py` |
| 粒子群优化 | 连续空间优化 | `algorithms/pso.py` |
| 模拟退火 | 跳出局部最优 | `algorithms/sa.py` |

### 预测类算法

| 算法 | 适用场景 | 文件位置 |
|------|----------|----------|
| GM(1,1) | 小样本时间序列 | `algorithms/grey_model.py` |
| 线性回归 | 趋势预测 | `algorithms/linear_regression.py` |
| 多项式回归 | 非线性趋势 | `algorithms/polynomial_regression.py` |

### 评价类算法

| 算法 | 适用场景 | 文件位置 |
|------|----------|----------|
| AHP | 多准则决策 | `algorithms/ahp.py` |
| TOPSIS | 方案排序 | `algorithms/topsis.py` |
| 熵权法 | 客观赋权 | `algorithms/entropy_weight.py` |
| PCA | 降维分析 | `algorithms/pca.py` |

## 竞赛题型对照

### CUMCM A题（优化类）
- 典型问题：资源分配、生产调度、路径规划
- 推荐算法：线性规划、整数规划、遗传算法、粒子群
- 使用模板：`templates/optimization_template.py`

### CUMCM B题（预测类）
- 典型问题：趋势预测、人口预测、需求预测
- 推荐算法：GM(1,1)、回归分析、时间序列
- 使用模板：`templates/prediction_template.py`

### CUMCM C题（评价类）
- 典型问题：方案评价、绩效评估、竞争力分析
- 推荐算法：AHP、TOPSIS、熵权法、PCA
- 使用模板：`templates/evaluation_template.py`

## 运行示例

```bash
# 运行优化示例
cd D:\本地的知识库构建\math-modeling-competition
python examples/example_optimization.py

# 运行预测评价示例
python examples/example_prediction_evaluation.py
```

## 详细文档

每个算法模块都包含完整的docstring和使用示例。查看 `examples/` 目录了解具体用法。

## 贡献指南

欢迎提交Issue和Pull Request！请确保：
1. 代码符合PEP 8规范
2. 添加必要的注释
3. 编写测试用例

## 许可证

MIT License
