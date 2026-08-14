# 可视化指南

## 一、图表选择原则

### 1. 对比类
| 场景 | 图表 |
|------|------|
| 类别对比 | 柱状图 |
| 趋势对比 | 折线图 |
| 占比对比 | 饼图/环形图 |

### 2. 关系类
| 场景 | 图表 |
|------|------|
| 相关性 | 散点图 |
| 分布关系 | 箱线图 |
| 层次关系 | 树状图 |

### 3. 分布类
| 场景 | 图表 |
|------|------|
| 单变量分布 | 直方图/KDE |
| 多变量分布 | 热力图 |
| 三维分布 | 3D散点图 |

---

## 二、常用图表代码

### 1. 折线图
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
y = np.random.randn(10).cumsum()

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='Data')
plt.fill_between(x, y, alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Trend Line Chart')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 2. 柱状图
```python
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

plt.bar(categories, values, color='steelblue', edgecolor='black')
for i, v in enumerate(values):
    plt.text(i, v + 1, str(v), ha='center')
plt.ylabel('Value')
plt.title('Bar Chart')
plt.show()
```

### 3. 散点图
```python
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5

plt.scatter(x, y, c='blue', alpha=0.6, s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.grid(True, alpha=0.3)
plt.show()
```

### 4. 热力图
```python
corr = np.corrcoef(data.T)
plt.imshow(corr, cmap='RdYlBu_r', aspect='auto')
plt.colorbar(label='Correlation')
plt.xticks(range(len(cols)), cols, rotation=45)
plt.yticks(range(len(cols)), cols)
plt.title('Correlation Heatmap')
plt.show()
```

---

## 三、数学建模常用图

### 1. 收敛曲线
```python
plt.plot(ga.best_history, 'b-', label='Best')
plt.plot(ga.avg_history, 'r--', label='Average')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Convergence Curve')
plt.legend()
plt.show()
```

### 2. 拟合曲线
```python
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, 'r-', label='Fit')
plt.legend()
plt.show()
```

### 3. 误差棒图
```python
plt.errorbar(x, means, yerr=errors, fmt='o', capsize=5)
plt.show()
```

---

## 四、期刊配图规范

### 1. 字体要求
- 中文期刊: 宋体正文 + Times New Roman数字
- 英文期刊: Arial/Times New Roman

### 2. 线条宽度
- 坐标轴: 1.5pt
- 数据曲线: 1.5-2pt
- 网格线: 0.5pt

### 3. 分辨率
- 印刷: 300 DPI
- 屏幕: 150 DPI
- 矢量图: PDF/EPS格式

### 4. 配色方案
- 色盲安全: 使用colorbrewer
- 灰度兼容: 用图案区分
- 避免彩虹色

---
