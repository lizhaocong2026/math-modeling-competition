# 完整工作流示例详解

> **仓库文件**：xamples/example_full.py  
> **覆盖内容**：端到端完整解题流程（含可视化）  
> **对应笔记**：[[论文写作与可视化-竞赛手册]] · [[算法实现详解-代码速查]]

---

## 一、完整流程

`
数据加载 → 预处理 → 特征工程 → 模型选择 → 训练 → 评估
    → 灵敏度分析 → 结果可视化 → 论文输出
`

## 二、关键代码段

### 2.1 特征工程
`python
from utils.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
# 缺失值处理
preprocessor.fill_missing(method='median')
# 异常值检测
preprocessor.detect_outliers(method='iqr', threshold=1.5)
# 特征编码
preprocessor.encode_categorical(method='onehot')
`

### 2.2 模型集成
`python
from algorithms.ensemble import RegressionEnsemble

ensemble = RegressionEnsemble(method='stacking')
ensemble.add_model('rf', RandomForestRegressor())
ensemble.add_model('svm', SVR())
ensemble.add_model('lr', LinearRegression())
result = ensemble.fit_predict(X_train, y_train)
`

### 2.3 灵敏度分析
`python
from templates.competition_template import CompetitionTemplate
template = CompetitionTemplate()

# 单因素灵敏度
sensitivity = template.sensitivity_analysis(
    result, param_names=['pop_size', 'mutation_rate'],
    ranges=[(50, 200), (0.01, 0.1)]
)
`

## 三、可视化输出

| 图表类型 | 函数 | 用途 |
|---------|------|------|
| 收敛曲线 | plot_convergence() | 算法迭代过程 |
| Pareto前沿 | plot_pareto() | 多目标解集 |
| 雷达图 | plot_radar() | 多指标对比 |
| 热力图 | plot_heatmap() | 相关性矩阵 |
| 箱线图 | plot_boxplot() | 灵敏度分布 |

---
> **更新**：2026-08-18
