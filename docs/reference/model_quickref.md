# 常用模型速查表

## 一、优化模型

### 1. 线性规划 (LP)
```python
from scipy.optimize import linprog

c = [-3, -2]  # 目标系数
A = [[2, 1], [1, 3]]  # 不等式约束
b = [20, 30]
bounds = [(0, None), (0, None)]

result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
```

### 2. 整数规划
```python
from scipy.optimize import milp, Bounds, LinearConstraint

# 所有变量为整数
integrality = np.ones(n)
result = milp(c=c, constraints=[LinearConstraint(A, -inf, b)], 
              integrality=integrality, bounds=Bounds(0, inf))
```

### 3. 非线性规划
```python
from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + x[1]**2

cons = ({'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]})
result = minimize(objective, x0=[1,1], constraints=cons)
```

---

## 二、预测模型

### 1. 灰色预测 GM(1,1)
```python
from algorithms.grey_model import GM11

data = np.array([4.87, 5.38, 5.94, 6.54])
gm = GM11()
result = gm.fit_predict(data, steps=3)
```

### 2. 线性回归
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 3. ARIMA
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data, order=(1,1,1))
result = model.fit()
forecast = result.forecast(steps=5)
```

---

## 三、评价模型

### 1. AHP
```python
from algorithms.ahp import AHP

matrix = np.array([[1, 3, 5], [1/3, 1, 2], [1/5, 1/2, 1]])
ahp = AHP()
result = ahp.compare(matrix)
```

### 2. TOPSIS
```python
from algorithms.topsis import TOPSIS

data = np.array([[85, 90], [92, 85]])
topsis = TOPSIS()
result = topsis.evaluate(data)
```

### 3. 熵权法
```python
from algorithms.entropy_weight import EntropyWeight

entropy = EntropyWeight()
result = entropy.evaluate(data)
```

---

## 四、智能算法

### 1. 遗传算法
```python
from algorithms.ga import GeneticAlgorithm

def fitness(x):
    return np.sum(x**2)

ga = GeneticAlgorithm()
result = ga.optimize(fitness, [(-5,5)]*3, is_maximization=False)
```

### 2. 粒子群
```python
from algorithms.pso import ParticleSwarm

pso = ParticleSwarm()
result = pso.optimize(fitness, [(-5,5)]*3)
```

### 3. 蚁群算法 (TSP)
```python
from algorithms.aco import AntColony

dist = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
aco = AntColony()
result = aco.solve_tsp(dist)
```

---

## 五、选择指南

| 问题特征 | 推荐模型 |
|----------|----------|
| 数据少(<20) | GM(1,1) |
| 数据多 | ARIMA/回归 |
| 多指标评价 | TOPSIS/熵权 |
| 主观权重 | AHP |
| 连续优化 | PSO/DE |
| 离散优化 | GA/ACO |
| 多目标 | NSGA-II |

---