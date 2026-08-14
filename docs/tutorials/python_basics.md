# Python入门与数据科学环境

## 一、环境配置

### 1. 安装Python
```bash
# Windows
winget install Python.Python.3.11

# 或使用Anaconda
# https://www.anaconda.com/download
```

### 2. 核心依赖
```bash
pip install numpy scipy matplotlib scikit-learn pandas
pip install statsmodels seaborn
```

---

## 二、NumPy基础

### 1. 数组创建
```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3])
b = np.zeros(5)
c = np.ones((3, 3))
d = np.arange(10)
e = np.linspace(0, 1, 5)

# 随机数组
f = np.random.rand(3, 3)
g = np.random.randn(100)
```

### 2. 数组运算
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 基本运算
c = a + b      # 逐元素相加
d = a * b      # 逐元素相乘
e = a @ b      # 点积
f = a.dot(b)   # 点积

# 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B      # 矩阵乘法
D = A.inv()    # 求逆
```

### 3. 常用函数
```python
a = np.array([1, 2, 3, 4, 5])

np.mean(a)     # 均值
np.std(a)      # 标准差
np.sum(a)      # 求和
np.max(a)      # 最大值
np.min(a)      # 最小值
np.sort(a)     # 排序
np.argsort(a)  # 排序索引
```

---

## 三、Pandas基础

### 1. 数据读取
```python
import pandas as pd

# 读取数据
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')

# 查看数据
df.head()          # 前5行
df.info()          # 信息摘要
df.describe()      # 统计描述
df.shape           # 形状
df.columns         # 列名
```

### 2. 数据筛选
```python
# 条件筛选
df[df['age'] > 20]
df[(df['age'] > 20) & (df['gender'] == 'M')]

# 按索引
df.loc[0:5, ['name', 'age']]
df.iloc[0:5, 0:3]
```

### 3. 数据处理
```python
# 缺失值
df.dropna()
df.fillna(df.mean())

# 分组统计
df.groupby('category')['value'].mean()

# 排序
df.sort_values('value', ascending=False)
```

---

## 四、Matplotlib基础

### 1. 基本绘图
```python
import matplotlib.pyplot as plt

x = np.arange(10)
y = x ** 2

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Line Chart')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. 多子图
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].bar(x, y)
axes[1, 1].hist(y)

plt.tight_layout()
plt.show()
```

---

## 五、常用技巧

### 1. 随机种子
```python
np.random.seed(42)  # 保证结果可复现
```

### 2. 性能优化
```python
# 向量化运算（避免循环）
# 错误: 慢
result = []
for i in range(1000000):
    result.append(i**2)

# 正确: 快
result = np.arange(1000000) ** 2
```

### 3. 文件输出
```python
# 保存数据
df.to_csv('output.csv', index=False)

# 保存图片
plt.savefig('chart.png', dpi=300, bbox_inches='tight')
```

---