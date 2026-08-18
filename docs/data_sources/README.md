# 数学建模竞赛数据源总览

> **核心结论**：本仓库包含 90+ 个权威数据源，覆盖金融财经、互联网报告、地理环境遥感、政府行业四大类，为 CUMCM 赛题提供完整的数据支撑。

---

## 数据源分类索引

| 分类 | 文档 | 来源数量 | 适用赛题 |
|------|------|---------|---------|
| 金融财经数据 | [01-finance-data.md](./01-finance-data.md) | 16 | A题投资组合、B题金融预测 |
| 互联网与报告分析 | [02-internet-data.md](./02-internet-data.md) | 20 | B题趋势预测、C题行业评价 |
| 地理环境与遥感 | [03-geography-env-data.md](./03-geography-env-data.md) | 49 | A题环境规划、B题气候预测 |
| 政府与行业数据 | [04-government-industry-data.md](./04-government-industry-data.md) | 24 | 全类型赛题 |

---

## 数据来源可信度分级

| 等级 | 来源类型 | 典型来源 | 适用场景 |
|------|---------|---------|---------|
| ★★★★★ | 国家级权威 | 国家数据、国家统计局、PM25.in | 论文核心数据 |
| ★★★★ | 国际组织 | NOAA、FAO、NASA、World Bank | 国际对比研究 |
| ★★★★ | 学术数据库 | CCER、GLCF、中科院数据 | 科研级分析 |
| ★★★ | 商业平台 | 东方财富、同花顺、QuestMobile | 时效性强的数据 |
| ★★★ | 政府开放平台 | data.gov、data.gov.sg | 国际城市研究 |
| ★★☆ | 行业报告 | 易观、艾瑞、CBNData | 趋势分析参考 |
| ★★☆ | 社区/开源 | Awesome Public Datasets、figshare | 探索性分析 |

---

## 论文数据来源标注规范

`
[1] 国家统计局. 中国统计年鉴2023[M]. 北京: 中国统计出版社, 2023.
[2] 同花顺数据中心. 股票历史行情数据[EB/OL]. https://data.10jqka.com.cn
[3] PM25.in. 中国空气质量实时监测[EB/OL]. http://pm25.in
[4] FAO. Global Forest Resources Assessment 2020[R]. Rome, 2020.
[5] GLCF. Global Land Cover Facility Data[DB/OL]. http://glcf.geodata.cn
`

---

## 算法-数据源匹配速查

| 算法类别 | 推荐数据源 | 对应案例 |
|----------|-----------|---------|
| 优化 (GA/PSO/NSGA-II) | 政府数据 + 行业数据 | cases/cumcm_2022b_ev_charging.py |
| 预测 (ARIMA/LSTM) | 金融财经 + 气象数据 | cases/cumcm_2023a_drone_inspection.py |
| 评价 (AHP/TOPSIS/熵权) | 国家数据 + 行业报告 | cases/cumcm_2020b_ev_station.py |
| 仿真 (蒙特卡洛/卡尔曼) | 遥感 + 环境数据 | cases/cumcm_2021a_power_network.py |

---

> **更新日期**：2026-08-18  
> **数据来源**：数学建模国赛数据库汇总（4张原始截图）