# 数学建模国赛数据库汇总 · 总控

> **数据源总数**：109个（4大类）  
> **对应仓库文档**：docs/data_sources/00-algorithm-data-mapping.md  
> **关联笔记**：[[算法-数据源匹配速查表]] · [[数学建模选题决策树]]

---

## 数据源全景地图

| 分类 | 来源数 | 文档 | 典型赛题 | 核心算法 |
|------|--------|------|---------|---------|
| 金融财经 | 16 | [01-finance-data.md](../data_sources/01-finance-data.md) | A题投资组合、B题汇率预测、C题企业评价 | finance.py, arima.py, lstm.py, monte_carlo.py, bayesian.py |
| 互联网与报告 | 20 | [02-internet-data.md](../data_sources/02-internet-data.md) | B题热度预测、C题行业评价 | prophet.py, arima.py, ensemble.py, random_forest.py, svm.py |
| 地理环境与遥感 | 49 | [03-geography-env-data.md](../data_sources/03-geography-env-data.md) | A题土地利用、B题气候预测 | spatial.py, image.py, pca.py, cellular_automaton.py, arima.py |
| 政府与行业 | 24 | [04-government-industry-data.md](../data_sources/04-government-industry-data.md) | A题区域经济、B题人口预测、C题城市评价 | ahp.py, topsis.py, entropy_weight.py, grey_model.py, linear_regression.py |

---

## 可信度分级

| 等级 | 来源类型 | 代表来源 | 适用场景 |
|------|---------|---------|---------|
| ★★★★★ | 国家级权威 | 国家数据、国家统计局、中国统计年鉴 | 论文核心数据 |
| ★★★★★ | 国际组织 | NOAA、FAO、NASA、World Bank | 国际对比研究 |
| ★★★★ | 学术数据库 | CCER、GLCF、中科院数据 | 科研级分析 |
| ★★★★ | 商业平台 | 东方财富、同花顺、QuestMobile | 时效性强的数据 |
| ★★★ | 政府开放平台 | data.gov系列 | 国际城市研究 |
| ★★★ | 行业报告 | 易观、艾瑞、CBNData、DataEye | 趋势分析参考 |
| ★★ | 社区/开源 | Awesome Public Datasets、figshare | 探索性分析 |

---

## 快速跳转

- [金融财经数据 (16)](./01-finance-data.md) — [[数学建模国赛数据库汇总-金融财经数据]]
- [互联网与报告 (20)](./02-internet-data.md) — [[数学建模国赛数据库汇总-互联网与报告分析]]
- [地理环境与遥感 (49)](./03-geography-env-data.md) — [[数学建模国赛数据库汇总-地理环境与遥感]]
- [政府与行业数据 (24)](./04-government-industry-data.md) — [[数学建模国赛数据库汇总-政府与行业数据]]
- [算法-数据双向映射](./00-algorithm-data-mapping.md)

---
> **更新**：2026-08-18
