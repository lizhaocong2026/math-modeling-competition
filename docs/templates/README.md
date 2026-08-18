# 模板快速速查表

> **仓库文件**：templates/ 目录下的6个模板  
> **对应笔记**：[[论文模板与灵敏度分析]] · [[算法实现详解-代码速查]]

---

## 模板总览

| 模板文件 | 适用题型 | 核心类 | 主要方法 |
|---------|---------|--------|---------|
| competition_template.py | 通用 | CompetitionTemplate | load/preprocess/solve/validate/visualize |
| optimization_template.py | A题优化 | OptimizationProblemTemplate | solve_linear_programming/solve_ga/solve_nsga2 |
| prediction_template.py | B题预测 | PredictionProblemTemplate | solve_grey_model/solve_arima/solve_lstm |
| evaluation_template.py | C题评价 | EvaluationProblemTemplate | solve_topsis/solve_ahp/solve_entropy |
| nn_template.py | 深度学习 | NeuralNetworkTemplate | build_model/train/solve_lstm |
| classification_clustering_template.py | 分类聚类 | ClassificationClusteringTemplate | kmeans_cluster/svm_classifier/feature_selection |

---

## 最短调用示例

### 优化问题（5行代码）
`python
from templates.optimization_template import OptimizationProblemTemplate
t = OptimizationProblemTemplate()
r = t.solve_linear_programming(c=[2,3,1], constraints={...})
`

### 预测问题（5行代码）
`python
from templates.prediction_template import PredictionProblemTemplate
t = PredictionProblemTemplate()
r = t.solve_grey_model(data=[10,12,15,18,22])
`

### 评价问题（5行代码）
`python
from templates.evaluation_template import EvaluationProblemTemplate
t = EvaluationProblemTemplate()
r = t.solve_topsis(data=np.array([[90,85],[80,90]]))
`

---

## 模板依赖关系

`
competition_template (通用框架)
    ├── optimization_template (A题)
    ├── prediction_template (B题)
    ├── evaluation_template (C题)
    ├── nn_template (深度学习)
    └── classification_clustering_template (分类聚类)

所有模板共同依赖:
    ├── algorithms/*.py (算法核心)
    ├── utils/data_preprocessor.py (数据预处理)
    └── visualizations/model_viz.py (可视化)
`

---
> **更新**：2026-08-18
