# 算法实现指南

> **仓库文件**：docs/theory/algorithm_implementation.md  
> **对应笔记**：[[算法实现详解-代码速查]]

---

## 一、统一接口规范

所有算法遵循统一接口：

- fit(data, **params): 训练/拟合
- predict(data): 预测/求解
- evaluate(data, **params): 评价/打分
- get_params(): 获取当前参数

---

## 二、算法文件命名规范

| 类型 | 命名 | 示例 |
|------|------|------|
| 单一算法 | snake_case | ga.py, pso.py, topsis.py |
| 多算法组合 | snake_case plural | mcdm_advanced.py |
| 案例 | cumcm YEAR LETTER DESC.py | cumcm_2023a_drone_inspection.py |
| 模板 | snake_case_template.py | optimization_template.py |
| 工具函数 | snake_case.py | data_preprocessor.py |

---

## 三、返回结果规范

所有算法返回结构化字典：
- status: success/error
- result: 主结果
- metrics: 评估指标
- params: 使用的参数
- warnings: 警告信息
- time_cost: 耗时(秒)

---

## 四、测试规范

每个算法模块需有对应测试，核心算法覆盖率 > 90%

---

## 五、依赖管理

核心依赖：
- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.5

可选依赖：
- scikit-learn >= 1.0（分类/聚类/SVM/RF）
- statsmodels >= 0.13（ARIMA）
- pandas >= 1.3（数据处理）
- akshare >= 1.0（数据获取）

---

## 六、代码注释规范

使用三引号文档字符串，包含功能说明、作者、日期、参考文献

---
> **更新日期**：2026-08-18
