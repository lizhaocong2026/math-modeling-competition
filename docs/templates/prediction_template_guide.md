# 预测类模板用法指南

> **仓库文件**：	emplates/prediction_template.py  
> **适用题型**：CUMCM B题（预测趋势类）  
> **关联算法**：grey_model.py · arima.py · prophet.py · lstm.py · timeseries.py

---

## 一、快速开始

`python
from templates.prediction_template import PredictionProblemTemplate

template = PredictionProblemTemplate()

# 灰色预测 GM(1,1)
result = template.solve_grey_model(data=[10, 12, 15, 18, 22])

# ARIMA 时间序列
result = template.solve_arima(series, order=(1,1,1))

# LSTM 深度学习预测
result = template.solve_lstm(X_train, y_train, epochs=100)
`

## 二、模型选择决策树

`
数据量 < 10 → GM(1,1) 灰色预测
数据量 10-100 → ARIMA / Prophet
数据量 > 100 且有规律 → LSTM / GRU
有强季节性 → Prophet
多特征影响 → 线性回归 + 特征工程
`

## 三、评估指标

| 指标 | 公式 | 含义 |
|------|------|------|
| RMSE | sqrt(MSE) | 预测误差标准差 |
| MAE | mean|y-ŷ| | 平均绝对误差 |
| MAPE | mean|(y-ŷ)/y|*100% | 相对误差百分比 |
| R² | 1-SSres/SStot | 拟合优度 |

## 四、ARIMA定阶指南

1. 平稳性检验（ADF检验）
2. 自相关图(ACF)和偏自相关图(PACF)
3. AIC/BIC信息准则定阶
4. 残差白噪声检验

---
> **更新**：2026-08-18
