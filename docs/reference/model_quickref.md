# 常用模型速查表

> **仓库文件**：docs/reference/model_quickref.md  
> **对应笔记**：[[算法实现详解-代码速查]] · [[优化算法-竞赛手册]] · [[预测算法-竞赛手册]] · [[评价算法-竞赛手册]]

---

## 一、优化模型速查

### 线性规划 (LP)
`python
from scipy.optimize import linprog
c = [-3, -2]  # 最大化 -min(-c^T x)
A = [[2, 1], [1, 3]]
b = [20, 30]
result = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)])
`

### 整数规划 (IP)
`python
from scipy.optimize import milp, Bounds, LinearConstraint
import numpy as np
integrality = np.ones(n)  # 1表示整数变量
result = milp(c=c, constraints=[LinearConstraint(A, -np.inf, b)],
              integrality=integrality, bounds=Bounds(0, np.inf))
`

### 非线性规划 (NLP)
`python
from scipy.optimize import minimize
def objective(x):
    return x[0]**2 + x[1]**2
cons = ({type: ineq, fun: lambda x: 1 - x[0] - x[1]})
result = minimize(objective, x0=[1,1], constraints=cons)
`

### 遗传算法 (GA)
`python
from algorithms.ga import GeneticAlgorithm
ga = GeneticAlgorithm(pop_size=100, max_gen=300)
result = ga.optimize(func=lambda x: -(x[0]**2 + x[1]**2),
                     bounds=[(-5, 5), (-5, 5)])
`

### NSGA-II 多目标
`python
from algorithms.nsga2 import NSGAII
nsga = NSGAII(pop_size=100, max_gen=200)
result = nsga.optimize(
    objectives=[lambda x: x[0]**2 + x[1]**2,
                lambda x: (x[0]-3)**2 + (x[1]-3)**2],
    bounds=[(0, 5), (0, 5)]
)
# result['pareto_front'] 为Pareto解集
`

### ACO 蚁群
`python
from algorithms.aco import AntColony
aco = AntColony(n_ants=20, max_iter=100, Q=100)
dist_matrix = compute_distance_matrix(points)
best_path, best_len = aco.solve(dist_matrix)
`

---

## 二、预测模型速查

### 灰色预测 GM(1,1)
`python
from algorithms.grey_model import GM11
gm = GM11()
result = gm.fit_predict(data=[4.87, 5.38, 5.94, 6.54], steps=3)
print(result['forecast'])  # 预测值
print(result['accuracy'])  # 精度检验
`

### ARIMA
`python
from algorithms.arima import ARIMA
from statsmodels.tsa.stattools import adfuller
# 平稳性检验
p_value = adfuller(series)[1]
arima = ARIMA()
model = arima.fit(series, order=(1,1,1))
forecast = arima.predict(model, steps=10)
`

### Prophet
`python
from algorithms.prophet import ProphetDecompose
prophet = ProphetDecompose(yearly_seasonality=True,
                            weekly_seasonality=True)
result = prophet.fit_predict(df, periods=30)
# df: {ds: date, y: value}
`

### LSTM
`python
from algorithms.lstm import LSTM
lstm = LSTM(hidden_units=64, epochs=100, look_back=10)
result = lstm.fit_predict(X_train, y_train, X_test)
`

### 线性回归
`python
from algorithms.linear_regression import LinearRegression
lr = LinearRegression()
result = lr.fit(X, y)
predictions = lr.predict(X_new)
print(result['r_squared'])
print(result['coefficients'])
`

---

## 三、评价模型速查

### TOPSIS
`python
from algorithms.topsis import TOPSIS
data = np.array([[90, 85, 88], [80, 90, 82], [85, 88, 90]])
topsis = TOPSIS()
result = topsis.evaluate(data, weights=[0.3, 0.4, 0.3])
print(result['scores'])  # 各方案得分
print(result['ranking'])  # 排序
`

### AHP
`python
from algorithms.ahp import AHP
matrix = np.array([[1, 3, 5], [1/3, 1, 2], [1/5, 1/2, 1]])
ahp = AHP()
result = ahp.compare(matrix)
print(result['weights'])
print(result['consistency_ratio'])
`

### 熵权法
`python
from algorithms.entropy_weight import EntropyWeight
ew = EntropyWeight()
result = ew.evaluate(data)
print(result['weights'])  # 客观权重
print(result['scores'])   # 综合得分
`

### PCA
`python
from algorithms.pca import PCA
pca = PCA(n_components=3)
result = pca.fit_transform(data)
print(result['explained_variance_ratio'])  # 方差贡献率
print(result['components'])  # 主成分
`

### VIKOR
`python
from algorithms.mcdm_advanced import VIKOR
vikor = VIKOR(v=0.5)  # v=0.5折中策略
result = viakor.evaluate(data, weights)
print(result['Q_values'])  # 综合指数
print(result['ranking'])
`

---

## 四、深度学习模型速查

### 神经网络
`python
from algorithms.nn import NeuralNetwork
nn = NeuralNetwork(layers=[5, 16, 8, 1], activation=relu)
nn.fit(X_train, y_train, epochs=100)
pred = nn.predict(X_test)
`

### 随机森林
`python
from algorithms.random_forest import RandomForest
rf = RandomForest(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
score = rf.score(X_test, y_test)
importance = rf.feature_importances_
`

### SVM
`python
from algorithms.svm import SVM
svm = SVM(kernel=rbf, C=1.0)
svm.fit(X_train, y_train)
pred = svm.predict(X_test)
`

---

## 五、仿真与统计模型速查

### 蒙特卡洛模拟
`python
from algorithms.monte_carlo import MonteCarlo
mc = MonteCarlo(n_simulations=10000)
result = mc.simulate(func, params)
print(result['mean'], result['std'])
print(result['confidence_interval'])
`

### 卡尔曼滤波
`python
from algorithms.kalman_filter import KalmanFilter
kf = KalmanFilter(dim_state=2, dim_obs=1)
state, cov = kf.update(measurement)
`

### 假设检验
`python
from algorithms.statistics import HypothesisTest
test = HypothesisTest()
result = test.t_test(sample1, sample2, equal_var=True)
print(result['p_value'])
print(result['conclusion'])
`

### ODE求解
`python
from algorithms.ode_solver import RK4
def lotka_volterra(t, y):
    prey, predator = y
    return [1.5*prey - prey*predator, prey*predator - predator]
rk4 = RK4()
result = rk4.solve(lotka_volterra, y0=[10, 5], t_span=(0, 20), n_steps=1000)
`

---

## 六、模型选择决策速查

| 问题特征 | 数据量 | 推荐模型 | 仓库文件 |
|---------|--------|---------|---------|
| 优化+约束 | 任意 | 线性/整数规划 | optimization.py |
| 全局优化+NP难 | 任意 | GA/PSO/ACO | ga.py/pso.py/aco.py |
| 多目标冲突 | 任意 | NSGA-II | nsga2.py |
| 小样本预测 | <10 | GM(1,1) | grey_model.py |
| 平稳时序 | >50 | ARIMA | arima.py |
| 带季节性的预测 | >50 | Prophet | prophet.py |
| 大量历史数据 | >500 | LSTM | lstm.py |
| 多指标评价 | 任意 | TOPSIS+熵权 | topsis.py/entropy_weight.py |
| 有专家知识 | 任意 | AHP | ahp.py |
| 高维数据 | 任意 | PCA | pca.py |
| 分类任务 | 任意 | SVM/RF | svm.py/random_forest.py |
| 聚类任务 | 任意 | KMeans/DBSCAN | classification.py |

---
> **更新日期**：2026-08-18
