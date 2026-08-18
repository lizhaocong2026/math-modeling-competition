# 完整工作流示例详解

> **仓库文件**：xamples/example_complete_workflow.py  
> **覆盖内容**：数据预处理 → 评价 → 预测 → 可视化全流程  
> **对应笔记**：[[数据处理与预处理-竞赛手册]] · [[评价算法-竞赛手册]] · [[预测算法-竞赛手册]]

---

## 一、流程概览

`
原始数据 → 预处理(缺失值/标准化) → 熵权法赋权 → TOPSIS评价
    → 灰色预测GM(1,1) → 曲线拟合 → 结果可视化
`

## 二、代码解读

### 2.1 数据预处理模块
`python
preprocessor = DataPreprocessor()
result = preprocessor.process(raw_data, fill_missing=True, normalize=True)
# fill_missing: 均值填充
# normalize: Min-Max标准化到[0,1]
`

### 2.2 评价模块
`python
entropy = EntropyWeight()
ew_result = entropy.evaluate(data)
# 输出: weights(权重), scores(得分), rankings(排名)

topsis = TOPSIS()
topsis_result = topsis.evaluate(data, weights=ew_result['weights'])
`

### 2.3 预测模块
`python
gm = GM11()
forecast = gm.forecast(n=5)  # 预测未来5期

cf = CurveFitting()
fit_result = cf.fit(x, y, degree=3)
`

## 三、输出说明

| 输出项 | 格式 | 用途 |
|--------|------|------|
| processed_data | numpy array | 后续分析输入 |
| entropy_weights | list[float] | 客观权重 |
| topsis_scores | list[float] | 方案排序 |
| forecast_values | list[float] | 未来预测 |
| fit_params | dict | 拟合参数 |

---
> **更新**：2026-08-18
