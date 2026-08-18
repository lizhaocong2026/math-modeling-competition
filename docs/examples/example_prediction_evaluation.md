# 预测与评价示例详解

> **仓库文件**：xamples/example_prediction_evaluation.py  
> **覆盖内容**：时间序列预测 + 多指标评价  
> **对应笔记**：[[预测算法-竞赛手册]] · [[评价算法-竞赛手册]]

---

## 一、ARIMA预测示例

`python
from algorithms.arima import ARIMA
from statsmodels.tsa.stattools import adfuller

# 平稳性检验
result = adfuller(series)
print(f'ADF p-value: {result[1]:.4f}')  # <0.05则平稳

# ARIMA建模
arima = ARIMA()
model = arima.fit(series, order=(1,1,1))
forecast = arima.predict(model, steps=10)
`

## 二、Prophet预测示例

`python
from algorithms.prophet import ProphetDecompose

prophet = ProphetDecompose(
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05
)
result = prophet.fit_predict(df, periods=30)
# df: {ds: date, y: value}
`

## 三、综合评价示例

`python
from algorithms.topsis import TOPSIS
from algorithms.ahp import AHP

# 步骤1: AHP主观赋权
ahp = AHP()
sub_weights = ahp.calculate_weights(judgment_matrix)

# 步骤2: 熵权客观赋权
ew = EntropyWeight()
obj_weights = ew.evaluate(data)['weights']

# 步骤3: 组合赋权
combined = 0.5 * sub_weights + 0.5 * obj_weights

# 步骤4: TOPSIS排序
topsis = TOPSIS()
result = topsis.evaluate(data, weights=combined)
`

---
> **更新**：2026-08-18
