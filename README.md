# Mathematical Modeling Competition Toolkit

> 全国大学生数学建模竞赛(CUMCM)核心算法库，提供优化、预测、评价、图像处理等模块的完整实现。

## 仓库结构

```
math-modeling-competition/
├── algorithms/           # 核心算法 (17个文件)
│   ├── optimization.py   # 线性/整数/非线性规划
│   ├── ga.py             # 遗传算法
│   ├── pso.py            # 粒子群优化
│   ├── sa.py             # 模拟退火
│   ├── de.py             # 差分进化
│   ├── aco.py            # 蚁群算法(TSP)
│   ├── grey_model.py     # GM(1,1)灰色预测
│   ├── arima.py          # ARIMA时间序列
│   ├── ahp.py            # 层次分析法
│   ├── topsis.py         # TOPSIS评价
│   ├── entropy_weight.py # 熵权法
│   ├── pca.py            # 主成分分析
│   ├── linear_regression.py
│   ├── polynomial_regression.py
│   ├── curve_fitting.py  # 曲线拟合
│   ├── nn.py             # 神经网络
│   ├── monte_carlo.py    # 蒙特卡洛模拟
│   ├── ensemble.py       # 回归集成
│   ├── timeseries.py     # 时间序列分解
│   ├── image.py          # 图像处理
│   └── constrained_opt.py # 约束优化
├── templates/            # 竞赛模板
├── utils/                # 工具函数
├── visualizations/       # 可视化
├── examples/             # 示例脚本
├── tests/                # 测试 (15项全部通过)
├── requirements.txt
└── README.md
```

## 快速开始

```bash
pip install numpy scipy matplotlib scikit-learn pandas
python -m pytest tests/ -v
```

## 算法对照表

### 优化算法 (CUMCM A题)
- 线性/整数/非线性规划
- 遗传算法(GA)、粒子群(PSO)、模拟退火(SA)
- 差分进化(DE)、蚁群算法(ACO-TSP)
- 约束优化、多目标优化

### 预测算法 (CUMCM B题)
- GM(1,1)灰色预测
- 线性/多项式回归、曲线拟合
- ARIMA时间序列
- 时间序列分解（趋势+季节+残差）
- 回归集成预测

### 评价算法 (CUMCM C题)
- AHP层次分析法
- TOPSIS逼近理想解
- 熵权法客观赋权
- PCA主成分分析

### 图像处理 (图像识别类题目)
- 灰度化、边缘检测(Sobel/Canny)
- 阈值分割(Otsu/自适应)
- KMeans图像分割
- Harris角点检测

### 其他工具
- 蒙特卡洛模拟（积分、风险评估）
- 神经网络（纯NumPy实现）
- 数据预处理（缺失值填充、归一化、异常值检测）

## 运行示例

```bash
# 运行测试
python -m pytest tests/ -v

# 运行完整示例
python examples/example_full.py
```

## 竞赛题型覆盖

| 题型 | 推荐算法 | 模板文件 |
|------|----------|----------|
| A题-优化 | LP/IP/GA/PSO/ACO | optimization_template.py |
| B题-预测 | GM(1,1)/回归/ARIMA | prediction_template.py |
| C题-评价 | AHP/TOPSIS/熵权 | evaluation_template.py |
| 图像识别 | 边缘检测/分割 | image.py |

## 依赖

- Python 3.8+
- numpy, scipy, matplotlib, scikit-learn, pandas

## 测试

```bash
python -m pytest tests/ -v
# 15 passed
```
