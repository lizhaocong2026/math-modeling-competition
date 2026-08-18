# 评价类模板用法指南

> **仓库文件**：	emplates/evaluation_template.py  
> **适用题型**：CUMCM C题（评价决策类）  
> **关联算法**：ahp.py · topsis.py · entropy_weight.py · pca.py · mcdm_advanced.py

---

## 一、快速开始

`python
from templates.evaluation_template import EvaluationProblemTemplate
import numpy as np

template = EvaluationProblemTemplate()

# TOPSIS评价
data = np.array([[90, 85, 88], [80, 90, 82], [85, 88, 90]])
result = template.solve_topsis(data, weights=[0.3, 0.4, 0.3])

# 熵权法客观赋权
from algorithms.entropy_weight import EntropyWeight
ew = EntropyWeight()
weights = ew.evaluate(data)['weights']

# AHP层次分析
from algorithms.ahp import AHP
ahp = AHP()
judgment_matrix = np.array([[1, 2, 3], [1/2, 1, 2], [1/3, 1/2, 1]])
weights = ahp.calculate_weights(judgment_matrix)
`

## 二、主客观赋权组合策略

| 策略 | 方法 | 适用场景 |
|------|------|---------|
| 纯客观 | 熵权法 | 数据充分，避免主观偏差 |
| 纯主观 | AHP | 专家知识丰富，数据少 |
| 组合赋权 | 熵权+AHP | 两者互补，结果更稳健 |

## 三、常见问题

- **权重全为0**：数据方差太小，检查数据质量
- **RI > 0.1**：AHP判断矩阵一致性不通过，需重新打分
- **TOPSIS排序与直觉不符**：检查效益型/成本型指标方向

---
> **更新**：2026-08-18
