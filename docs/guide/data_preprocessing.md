# 数据预处理手册

## 一、数据清洗

### 1. 缺失值处理

#### 检测方法
```python
# 查看缺失情况
df.isnull().sum()
df.isnull().mean()  # 缺失比例
```

#### 处理方法

| 方法 | 适用场景 | 说明 |
|------|----------|------|
| 删除 | 缺失<5% | 直接删除缺失行 |
| 均值填充 | 数值型，近似正态 | 用列均值填充 |
| 中位数填充 | 有异常值 | 用列中位数填充 |
| 众数填充 | 分类变量 | 用出现次数最多的值 |
| 插值 | 时间序列 | 线性/样条插值 |
| 模型预测 | 缺失较多 | 用其他变量预测 |

### 2. 异常值处理

#### 检测方法
- **3σ原则**: |x - μ| > 3σ
- **箱线图**: Q1-1.5IQR 或 Q3+1.5IQR
- **Z-score**: |z| > 3

#### 处理方法
- 删除
- 修正（如有错误）
- 截断（Winsorize）
- 替换为中位数

---

## 二、数据标准化

### 1. Min-Max标准化
```
x' = (x - min) / (max - min)
```
结果: [0, 1]

### 2. Z-score标准化
```
x' = (x - μ) / σ
```
结果: 均值0，标准差1

### 3. 归一化
```
x' = x / ||x||
```
结果: 向量模为1

### 4. 选择建议

| 场景 | 推荐方法 |
|------|----------|
| 神经网络 | Z-score |
| KNN/KMeans | Min-Max |
| 矩阵分解 | Z-score |
| 距离计算 | Min-Max |

---

## 三、数据变换

### 1. 对数变换
```
x' = log(x)
```
适用: 右偏分布

### 2. 平方根变换
```
x' = √x
```
适用: 轻度右偏

### 3. Box-Cox变换
```
x'(λ) = {(x^λ - 1)/λ, λ≠0; log(x), λ=0}
```
自动选择最优λ

---

## 四、特征工程

### 1. 特征选择

#### 过滤法
- 相关系数
- 卡方检验
- 信息增益

#### 包裹法
- 递归特征消除 (RFE)
- 前向/后向选择

#### 嵌入法
- Lasso回归
- 树模型重要性

### 2. 特征构造
- 多项式特征
- 交叉特征
- 时间特征（年/月/日/周几）
- 比率特征

---

## 五、代码示例

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# 1. 缺失值填充
imputer = SimpleImputer(strategy='median')
X_filled = imputer.fit_transform(X)

# 2. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

# 3. 结合使用
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols)
    ])
X_processed = preprocessor.fit_transform(df)
```

---
