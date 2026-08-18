# 神经网络模板用法指南

> **仓库文件**：	emplates/nn_template.py  
> **适用题型**：B题深度学习预测、C题分类评价  
> **关联算法**：nn.py · lstm.py · random_forest.py · svm.py

---

## 一、快速开始

`python
from templates.nn_template import NeuralNetworkTemplate

template = NeuralNetworkTemplate()

# 基础NN训练
model = template.build_model(input_dim=5, hidden_layers=[64, 32])
result = template.train(model, X_train, y_train, epochs=100)

# LSTM时序预测
lstm_result = template.solve_lstm(
    data=series,
    look_back=10,
    hidden_units=50,
    epochs=50
)
`

## 二、模型架构选择

| 任务类型 | 推荐架构 | 激活函数 | 适用数据 |
|---------|---------|---------|---------|
| 回归预测 | MLP/Dense | ReLU | 结构化数值 |
| 时序预测 | LSTM/GRU | sigmoid/tanh | 时间序列 |
| 分类 | MLP + Softmax | ReLU | 类别标签 |
| 特征提取 | Autoencoder | ReLU | 高维数据 |

## 三、超参数调优指南

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| 学习率 | 0.001-0.01 | Adam优化器 |
| batch_size | 32-128 | 内存限制 |
| epochs | 50-200 | 配合早停 |
| dropout | 0.2-0.5 | 防过拟合 |
| hidden_layers | [64,32] | 逐层递减 |

---
> **更新**：2026-08-18
