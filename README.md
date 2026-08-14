# 数学建模竞赛代码库 (Math Modeling Competition Toolkit)

> 针对全国大学生数学建模竞赛(CUMCM)和研究生数学建模竞赛(GMC)定制的核心算法库。
> 提供优化、预测、评价、分类等模块的完整Python实现，纯NumPy/SciPy依赖，开箱即用。

## 目录结构

```
math-modeling-competition/
├── core/                         # 核心包入口
│   └── __init__.py
├── algorithms/                   # 算法实现模块 (17个文件)
│   ├── optimization.py           # 线性/整数/非线性规划
│   ├── ga.py                     # 遗传算法
│   ├── pso.py                    # 粒子群优化
│   ├── sa.py                     # 模拟退火
│   ├── de.py                     # 差分进化
│   ├── aco.py                    # 蚁群算法(TSP)
│   ├── grey_model.py             # GM(1,1)灰色预测
│   ├── arima.py                  # ARIMA时间序列
│   ├── ahp.py                    # 层次分析法AHP
│   ├── topsis.py                 # TOPSIS评价
│   ├── entropy_weight.py         # 熵权法
│   ├── pca.py                    # 主成分分析
│   ├── linear_regression.py      # 线性回归
│   ├── polynomial_regression.py  # 多项式回归
│   ├── curve_fitting.py          # 曲线拟合
│   ├── nn.py                     # 神经网络(纯NumPy)
│   ├── monte_carlo.py            # 蒙特卡洛模拟
│   └── optimizer_interface.py    # 优化统一接口
├── templates/                    # 竞赛模板 (5个文件)
│   ├── optimization_template.py
│   ├── prediction_template.py
│   ├── evaluation_template.py
│   ├── classification_clustering_template.py
│   ├── nn_template.py
│   └── competition_template.py
├── utils/                        # 工具函数
│   ├── data_preprocessor.py      # 数据预处理
│   └── helpers.py                # 辅助函数
├── visualizations/               # 可视化
│   └── model_viz.py
├── examples/                     # 示例脚本
│   ├── example_optimization.py
│   ├── example_prediction_evaluation.py
│   └── example_complete_workflow.py
├── tests/                        # 测试
│   ├── test_algorithms.py
│   └── test_extended.py
├── data/                         # 示例数据目录
├── requirements.txt
├── README.md
└── PUSH_INSTRUCTIONS.md
```

## 快速开始

### 安装依赖
```bash
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

# 线性规划
lp = LinearProgramming()
result = lp.solve(
    c=np.array([-3, -2]),
    A_ub=np.array([[2, 1], [1, 3]]),
    b_ub=np.array([20, 30]),
    bounds=[(0, None), (0, None)]
)

# 灰色预测
gm = GM11()
result = gm.fit_predict(np.array([4.87, 5.38, 5.94, 6.54]))

# TOPSIS评价
topsis = TOPSIS()
result = topsis.evaluate(np.array([[85, 90], [92, 85]]))
```

## 算法对照表

### 优化算法
| 算法 | 适用场景 | 文件 |
|------|----------|------|
| 线性规划 | 资源分配、生产计划 | `optimization.py` |
| 整数规划 | 指派问题、背包问题 | `optimization.py` |
| 非线性规划 | 复杂约束优化 | `optimization.py` |
| 遗传算法(GA) | 全局优化、NP难题 | `ga.py` |
| 粒子群(PSO) | 连续空间优化 | `pso.py` |
| 模拟退火(SA) | 跳出局部最优 | `sa.py` |
| 差分进化(DE) | 多峰函数优化 | `de.py` |
| 蚁群算法(ACO) | TSP、路径规划 | `aco.py` |

### 预测算法
| 算法 | 适用场景 | 文件 |
|------|----------|------|
| GM(1,1) | 小样本时间序列 | `grey_model.py` |
| 线性回归 | 趋势预测 | `linear_regression.py` |
| 多项式回归 | 非线性趋势 | `polynomial_regression.py` |
| 曲线拟合 | 经验公式确定 | `curve_fitting.py` |
| ARIMA | 平稳时间序列 | `arima.py` |

### 评价算法
| 算法 | 适用场景 | 文件 |
|------|----------|------|
| AHP | 多准则决策 | `ahp.py` |
| TOPSIS | 方案排序 | `topsis.py` |
| 熵权法 | 客观赋权 | `entropy_weight.py` |
| PCA | 降维分析 | `pca.py` |

### 其他算法
| 算法 | 适用场景 | 文件 |
|------|----------|------|
| 蒙特卡洛 | 概率计算、风险评估 | `monte_carlo.py` |
| 神经网络 | 模式识别、回归 | `nn.py` |
| KMeans聚类 | 数据分组 | `templates/classification_clustering_template.py` |

## 竞赛题型对照

### CUMCM A题（优化类）
- 典型问题：资源分配、生产调度、路径规划
- 推荐算法：线性规划、整数规划、遗传算法
- 使用模板：`templates/optimization_template.py`

### CUMCM B题（预测类）
- 典型问题：趋势预测、人口预测、需求预测
- 推荐算法：GM(1,1)、回归分析
- 使用模板：`templates/prediction_template.py`

### CUMCM C题（评价类）
- 典型问题：方案评价、绩效评估
- 推荐算法：AHP、TOPSIS、熵权法
- 使用模板：`templates/evaluation_template.py`

## 运行示例
```bash
python examples/example_optimization.py
python examples/example_prediction_evaluation.py
python examples/example_complete_workflow.py
python -m pytest tests/ -v
```
