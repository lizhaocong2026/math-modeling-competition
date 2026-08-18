# 金融财经数据

> **适用赛题**：A题（投资组合优化、生产调度）、B题（股票/汇率预测）、C题（企业绩效评估）  
> **关联算法**：algorithms/finance.py · algorithms/bayesian.py · algorithms/monte_carlo.py

---

## 数据来源列表

| 序号 | 数据源 | 网址 | 数据类型 | 费用 | 备注 |
|------|--------|------|---------|------|------|
| 1 | 同花顺数据中心 | https://data.10jqka.com.cn | 股票、债券等金融数据 | 免费/付费 | 覆盖面最广 |
| 2 | 和讯数据 | http://data.hexun.com | 股票、基金、外汇、债券实时数据 | 有免费有付费 | 实时性强 |
| 3 | 零壹财经 | https://www.zeroapi.com | 网贷数据、排行榜 | 免费 | 互联网金融细分 |
| 4 | 金融数据网 | https://www.jrj.com.cn/data | 黄金、汇率、农产品、汽油价格 | 免费 | 每日更新 |
| 5 | 萝卜投研 | https://www.lbbok.com | 股市、证券（数据研究报告形式） | 免费/付费 | 研报格式丰富 |
| 6 | 金融界 | https://www.jjrj.com | 股市、融资、资金流向、财报研报 | 免费 | 综合金融门户 |
| 7 | 东方财富网 | https://data.eastmoney.com | 多国股票、财税、行业、消费数据 | 免费 | 海量宏观数据 |
| 8 | 吉林金融网 | https://www.jlfh.com.cn | 吉林省融资、市场数据 | 免费 | 区域金融 |
| 9 | 搜狐证券 | https://q.stock.sohu.com | 货币、外汇、行业、宏观数据 | 免费 | 宏观数据丰富 |
| 10 | CCER经济金融数据库 | https://www.ccer.cn | 企业财务年度数据、股票收益 | 付费 | 学术常用 |
| 11 | 香港金融管理局 | https://www.hkex.com.hk | 香港宏观经济及金融数据 | 免费 | 宏观为主 |
| 12 | 世纪未来 | https://www.caijing.com.cn | 银行研究、金融大数据 | 付费 | 付费数据服务 |
| 13 | 新浪财经 | https://finance.sina.com.cn | 国民经济、行业、对外经贸、居民收入 | 免费 | 宏观数据权威 |
| 14 | 司尔亚司数据信息有限公司 | — | 195+国家经济数据库 | 付费 | 国际数据覆盖广 |
| 15 | INSEE 数据 | https://www.insee.fr | 法国统计与经济研究院公开数据 | 免费 | 欧洲宏观数据 |
| 16 | 投中研究院 | https://www.chinaventure.com.cn | 投资领域分析报告 | 免费/付费 | 更新频率高 |

---

## 典型应用案例（对应仓库案例）

| 案例 | 涉及算法 | 数据来源 |
|------|---------|---------|
| cumcm_2022b_ev_charging.py | 线性规划 + 整数规划 | 能源价格数据（金融数据网/东方财富） |
| cumcm_2021a_power_network.py | 图算法 + 优化 | 电价数据（同花顺/和讯） |
| 投资组合优化 | 均值-方差 + NSGA-II | 股票收益率（同花顺数据中心） |
| 汇率预测 | ARIMA + LSTM | 外汇实时数据（和讯/金融数据网） |

---

## 数据获取代码模板

`python
# 方式一：akshare（免费，无需登录）
import akshare as ak
# 安装: pip install akshare
stock_data = ak.stock_zh_a_hist(symbol="000001", period="daily", adjust="qfq")

# 方式二：东方财富 API（免费）
import requests
url = "https://push2.eastmoney.com/api/qt/stock/get"
params = {"secid": "1.600519", "fields": "f43,f44,f45,f46,f47,f48"}
resp = requests.get(url, params=params)
`

---

> **更新日期**：2026-08-18