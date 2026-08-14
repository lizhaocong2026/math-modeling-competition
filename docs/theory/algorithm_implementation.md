# 常用算法实现详解

## 一、优化算法

### 1. 线性规划

#### 数学模型
```
min c^T x
s.t. Ax ≤ b
     x ≥ 0
```

#### 代码实现
```python
from scipy.optimize import linprog
import numpy as np

# 问题：max 3x1 + 2x2
# s.t. x1 + x2 <= 4, x1 - x2 <= 2

c = [-3, -2]  # 取负转为最小化
A = [[1, 1], [1, -1]]
b = [4, 2]
bounds = [(0, None), (0, None)]

result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
print(f"最优解: {result.x}")
print(f"最大值: {-result.fun}")
```

---

### 2. 遗传算法

#### 算法流程
```
初始化种群 → 评估适应度 → 选择 → 交叉 → 变异 → 精英保留 → 迭代
```

#### 核心代码
```python
import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size=100, max_gen=500):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
    
    def optimize(self, fitness_func, bounds, is_max=True):
        n_vars = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        # 初始化
        pop = np.random.uniform(lower, upper, (self.pop_size, n_vars))
        
        for gen in range(self.max_gen):
            # 评估
            fitness = np.array([fitness_func(x) for x in pop])
            
            # 选择（轮盘赌）
            probs = fitness / fitness.sum()
            idx = np.random.choice(self.pop_size, self.pop_size, p=probs)
            selected = pop[idx]
            
            # 交叉
            offspring = selected.copy()
            for i in range(0, self.pop_size-1, 2):
                if np.random.random() < self.crossover_rate:
                    alpha = np.random.random()
                    offspring[i] = alpha * selected[i] + (1-alpha) * selected[i+1]
                    offspring[i+1] = alpha * selected[i+1] + (1-alpha) * selected[i]
            
            # 变异
            mask = np.random.random(offspring.shape) < self.mutation_rate
            offspring[mask] += np.random.randn(*offspring.shape)[mask] * 0.1
            offspring = np.clip(offspring, lower, upper)
            
            pop = offspring
        
        best_idx = fitness.argmax() if is_max else fitness.argmin()
        return pop[best_idx], fitness[best_idx]
```

---

## 二、预测模型

### 1. GM(1,1)灰色预测

#### 推导过程
1. 原始序列: x⁽⁰⁾ = (x⁽⁰⁾(1), ..., x⁽⁰⁾(n))
2. 累加: x⁽¹⁾(k) = Σx⁽⁰⁾(i)
3. 邻均值: z⁽¹⁾(k) = 0.5(x⁽¹⁾(k) + x⁽¹⁾(k-1))
4. 白化方程: dx⁽¹⁾/dt + ax⁽¹⁾ = b
5. 离散形式: x⁽⁰⁾(k) + az⁽¹⁾(k) = b

#### 参数估计
```
â = (B^TB)⁻¹B^TY
B = [-z⁽¹⁾(2)  1]
    [-z⁽¹⁾(3)  1]
    [...      ]
    [-z⁽¹⁾(n)  1]
Y = [x⁽⁰⁾(2), x⁽⁰⁾(3), ..., x⁽⁰⁾(n)]^T
```

#### 时间响应
```
x̂⁽¹⁾(k) = (x⁽⁰⁾(1) - b/a)e^{-a(k-1)} + b/a
x̂⁽⁰⁾(k) = x̂⁽¹⁾(k) - x̂⁽¹⁾(k-1)
```

---

### 2. 线性回归

#### 最小二乘推导
```
L(β) = Σ(y_i - β₀ - β₁x_i)²

∂L/∂β₀ = -2Σ(y_i - β₀ - β₁x_i) = 0
∂L/∂β₁ = -2Σx_i(y_i - β₀ - β₁x_i) = 0

解得:
β̂₁ = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)²
β̂₀ = ȳ - β̂₁x̄
```

---

## 三、评价方法

### 1. TOPSIS

#### 矩阵运算
```python
import numpy as np

def topsis(matrix, weights, types):
    # 标准化
    norm = np.sqrt(np.sum(matrix**2, axis=0))
    Z = matrix / norm
    
    # 加权
    V = Z * weights
    
    # 理想解
    if types == 'benefit':
        v_plus = V.max(axis=0)
        v_minus = V.min(axis=0)
    else:
        v_plus = V.min(axis=0)
        v_minus = V.max(axis=0)
    
    # 距离
    D_plus = np.sqrt(np.sum((V - v_plus)**2, axis=1))
    D_minus = np.sqrt(np.sum((V - v_minus)**2, axis=1))
    
    # 贴近度
    C = D_minus / (D_plus + D_minus)
    return C
```

---

### 2. 熵权法

#### 公式推导
```
p_ij = x'_ij / Σx'_ij
e_j = -k Σ p_ij ln(p_ij),  k = 1/ln(n)
g_j = 1 - e_j
w_j = g_j / Σg_j
```

---

## 四、算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 单纯形法 | 精确解 | 维度受限 | 小规模LP |
| 遗传算法 | 全局搜索 | 参数多 | NP难问题 |
| PSO | 简单快速 | 易早熟 | 连续优化 |
| SA | 跳出局部 | 慢 | 组合优化 |
| GM(1,1) | 小样本 | 线性假设 | 贫信息 |
| ARIMA | 准确 | 需平稳 | 大样本 |

---