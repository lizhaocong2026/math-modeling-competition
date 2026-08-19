# 数学建模竞赛全算法目录 · 完整映射（v2.0）

> **核心结论**：本库是 CUMCM 竞赛的完整知识体系——代码（50算法+4案例+6模板）、数据源（109来源）、理论（7篇文档）、文档（17篇）、笔记（22篇MOC）五位一体，3跳内可达任意知识点。

---

## 仓库资产速查

| 模块 | 数量 | 说明 |
|------|------|------|
| algorithms/ | 50个.py | 优化/预测/评价/深度学习/仿真/统计 |
| cases/ | 4个 | 2020-2023 CUMCM 完整案例 |
| templates/ | 6个 | 竞赛论文全类型模板 |
| docs/data_sources/ | 5个.md | 109+ 权威数据源文档 |
| docs/guide/ | 4个.md | 竞赛指南/数据预处理/论文写作/可视化 |
| docs/theory/ | 7个.md | 数学推导/算法理论 |
| docs/templates/ | 7个.md | 模板详细用法指南（新增） |
| docs/examples/ | 4个.md | 示例代码详解（新增） |
| docs/faq/ | 3个.md | 常见问题/决策树/避坑指南（新增） |
| examples/ | 4个.py | 完整工作流示例 |
| tests/ | 29项 | 全部通过 |
| utils/ | 2个.py | 数据预处理+工具函数 |
| visualizations/ | 1个.py | 模型可视化 |

---

## 知识库笔记索引

### 算法核心（9篇）

| 笔记 | 核心内容 | 关键文件 |
|------|---------|---------|
| [[优化算法-竞赛手册]] | GA/PSO/SA/DE/ACO/NSGA-II/规划/贝叶斯优化 | ga.py, pso.py, aco.py, nsga2.py, optimization.py |
| [[预测算法-竞赛手册]] | GM(1,1)/ARIMA/LSTM/Prophet/回归/拟合 | grey_model.py, arima.py, lstm.py, prophet.py |
| [[评价算法-竞赛手册]] | AHP/TOPSIS/熵权/PCA/VIKOR/PROMETHEE/ELECTRE | ahp.py, topsis.py, entropy_weight.py, pca.py |
| [[深度学习与仿真算法-竞赛手册]] | NN/RF/SVM/蒙特卡洛/卡尔曼/排队论/MCMC | nn.py, random_forest.py, svm.py, monte_carlo.py |
| [[数据处理与预处理-竞赛手册]] | 缺失值/异常值/标准化/特征工程 | utils/data_preprocessor.py |
| [[高级算法专题-博弈论图论金融空间统计]] | 博弈论/图论/金融数学/空间统计/ODE/PDE/组合优化 | game_theory.py, graph.py, finance.py |
| [[算法实现详解-代码速查]] | 所有算法的调用接口和代码模板 | 全库算法 |
| [[数学建模选题决策树]] | 赛题到算法到数据完整决策流程（新增） | - |
| [[常见错误与避坑指南]] | 建模/算法/论文/数据四阶段常见错误（新增） | - |

### 模板速查（3篇）

| 笔记 | 核心内容 | 对应仓库文件 |
|------|---------|-------------|
| [[优化类模板用法指南]] | A题优化模板参数调优+常见问题 | templates/optimization_template.py |
| [[预测类模板用法指南]] | B题预测模型选择+评估指标 | templates/prediction_template.py |
| [[评价类模板用法指南]] | C题赋权组合策略+TOPSIS应用 | templates/evaluation_template.py |

### 论文写作（3篇）

| 笔记 | 核心内容 |
|------|---------|
| [[竞赛指南-CUMCM全流程]] | 三天安排/题型分类/FQA/团队分工 |
| [[论文写作与可视化-竞赛手册]] | 摘要模板/图表规范/LaTeX格式/配色 |
| [[论文模板与灵敏度分析]] | LaTeX模板/Python模板/灵敏度分析代码 |

### 案例与导航（3篇）

| 笔记 | 核心内容 |
|------|---------|
| [[案例解析-CUMCM历年真题]] | 2020-2023四题深度解析+可复用点 |
| [[算法-数据源匹配速查表]] | 选题到算法到数据的完整决策链+可信度分级 |
| [[数学建模国赛数据库汇总-总控]] | 算法乘数据源全景地图 |

### 数据源（4篇）

| 笔记 | 来源数 | 核心来源 |
|------|--------|---------|
| [[数学建模国赛数据库汇总-金融财经数据]] | 16 | 同花顺/东方财富/CCER/INSEE |
| [[数学建模国赛数据库汇总-互联网与报告分析]] | 20 | 百度指数/QuestMobile/易观/清博 |
| [[数学建模国赛数据库汇总-地理环境与遥感]] | 49 | PM25.in/GLCF/NOAA/NASA/FAO |
| [[数学建模国赛数据库汇总-政府与行业数据]] | 24 | 国家数据/统计年鉴/CADMAPPER |

### 审查与治理（1篇）

| 笔记 | 内容 |
|------|------|
| [[数学建模数据源融合-最终审查报告]] | 仓库脑暴结论+融合评估+后续建议 |

---

## 快速入口

- 新手入门：[[竞赛指南-CUMCM全流程]] -> [[算法实现详解-代码速查]] -> [[常见错误与避坑指南]]
- 选题决策：[[数学建模选题决策树]] -> [[算法-数据源匹配速查表]] -> 对应数据源笔记
- 模板速查：[[优化类模板用法指南]] / [[预测类模板用法指南]] / [[评价类模板用法指南]]
- 算法学习：[[优化算法-竞赛手册]] / [[预测算法-竞赛手册]] / [[评价算法-竞赛手册]]
- 论文写作：[[论文写作与可视化-竞赛手册]] -> [[论文模板与灵敏度分析]]
- 真题参考：[[案例解析-CUMCM历年真题]]
- 高级专题：[[高级算法专题-博弈论图论金融空间统计]]
- FAQ：docs/faq/faq.md（仓库内常见问题）

---

> 创建日期：2026-08-18  最后更新：2026-08-18  
> 数据源来源：4 张原始截图（109来源）  仓库地址：https://github.com/lizhaocong2026/math-modeling-competition  
> 本轮新增：模板用法指南3篇 + 选题决策树1篇 + 避坑指南1篇 + FAQ文档3篇 + Example文档4篇


---

## 新增资产（2026-08-19 深度脑暴后）

### 新增算法（6个）
| 文件 | 说明 | 适用题型 |
|------|------|---------|
| lgorithms/svr.py | SVM回归预测 | B题/C题 |
| lgorithms/xgboost.py | XGBoost梯度提升 | B题/C题 |
| lgorithms/lightgbm.py | LightGBM快速梯度提升 | B题/C题 |
| lgorithms/lda.py | 线性判别分析 | C题分类降维 |
| lgorithms/stl_decompose.py | STL时间序列分解 | B题预测 |
| lgorithms/sarima.py | SARIMA季节模型 | B题周期预测 |
| lgorithms/rf_regression.py | 随机森林回归 | B题/C题 |
| lgorithms/cma_es.py | CMA-ES进化策略 | A题优化 |
| lgorithms/gan.py | 生成对抗网络 | 数据增强 |
| lgorithms/automl.py | 自动机器学习流水线 | 全流程 |

### 新增论文模板
- paper/document.tex - 主编译文件
- paper/texfile/1abstract.tex - 摘要模板
- paper/texfile/2ProblemRestatement.tex - 问题重述
- paper/texfile/3ProblemAnalysis.tex - 问题分析
- paper/texfile/4AssumptionAndSign.tex - 假设与符号
- paper/texfile/5MakeModel.tex - 模型建立与求解
- paper/texfile/6ErrorAnalysis.tex - 误差分析
- paper/texfile/7ModelEvaluation.tex - 模型评价
- paper/texfile/8Reference.tex - 参考文献
- paper/texfile/9Appendix.tex - 附录

### 新增文档
- docs/brainstorming_report.md - 深度脑暴报告
- docs/paper_writing/paper_writing_guide.md - 论文写作指南
- docs/references/external_resources.md - 外部资源汇总
- 
eference_papers/README.md - 范文存放说明

### 外部资源收录
- EmpyreanHYR/CUMCM-Latex-template (37星)
- Sustainable-Enjoyment/CUMCM-LaTeX-Template (37星)
- personqianduixue/CUMCM_LaTeX_Template (49星)
- zhanwen/MathModel (11265星)
- machinelearninglab/mathematical-modeling (84星)

---

## 快速开始

1. 查看脑暴报告：docs/brainstorming_report.md
2. 使用论文模板：paper/document.tex
3. 参考写作指南：docs/paper_writing/paper_writing_guide.md
4. 浏览外部资源：docs/references/external_resources.md

