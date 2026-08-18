# 优化问题示例详解

> **仓库文件**：xamples/example_optimization.py  
> **覆盖内容**：线性规划 + 遗传算法 + 多目标优化  
> **对应笔记**：[[优化算法-竞赛手册]]

---

## 一、线性规划示例

`python
from algorithms.optimization import LinearProgramming

lp = LinearProgramming(verbose=True)
result = lp.solve(
    c=np.array([-1, -2]),  # 最大化 -min(-c^T x)
    A_ub=np.array([[1, 1], [1, 0], [0, 1]]),
    b_ub=np.array([4, 2, 3]),
    bounds=[(0, None), (0, None)]
)
`

## 二、遗传算法示例

`python
from algorithms.ga import GeneticAlgorithm

ga = GeneticAlgorithm(pop_size=100, max_gen=300)
result = ga.optimize(
    func=lambda x: -(x[0]**2 + x[1]**2),  # 最大化
    bounds=[(-5, 5), (-5, 5)]
)
`

## 三、NSGA-II多目标示例

`python
from algorithms.nsga2 import NSGAII

nsga = NSGAII(pop_size=100, max_gen=200)
result = nsga.optimize(
    objectives=[
        lambda x: x[0]**2 + x[1]**2,  # 最小化距离
        lambda x: (x[0]-3)**2 + (x[1]-3)**2  # 最小化到(3,3)
    ],
    bounds=[(0, 5), (0, 5)]
)
# result['pareto_front'] 为Pareto前沿解集
`

---
> **更新**：2026-08-18
