# 政府与行业数据

> **适用赛题**：A题（政策优化）、B题（人口/经济预测）、C题（绩效评价）  
> **关联算法**：algorithms/optimization.py · algorithms/statistics.py · algorithms/ahp.py

---

## 一、政府公开数据

| 序号 | 数据源 | 网址 | 数据类型 | 说明 |
|------|--------|------|---------|------|
| 1 | 新加坡政府公开数据 | https://data.gov.sg | 新加坡各类政府数据 | 开放数据标杆 |
| 2 | 美国政府公开数据 | https://www.data.gov | 美国全国各类政府数据 | 全球最大开放数据平台 |
| 3 | 法国政府公开数据 | https://www.data.gouv.fr | 法国开放数据平台 | 欧盟开放数据 |
| 4 | 英国政府公开数据 | https://data.gov.uk | 英国各类政府数据 | 开放数据 |
| 5 | 国家数据 | https://www.stats.gov.cn | 中国国家统计局权威数据 | **竞赛最常用** |
| 6 | 中国统计年鉴 | https://www.stats.gov.cn | 1999年至今统计年鉴 | 单页 Excel 下载 |
| 7 | 中国统计信息网 | https://www.stars.gov.cn | 全国各级政府统计公报、年鉴 | 收费数据 |
| 8 | 年鉴汪 | https://www.nianjianwang.com | 全国城市统计数据搜索引擎 | 浏览免费，下载收费 |
| 9 | 伦敦市公开数据 | https://data.london.gov.uk | 伦敦人口/就业/环境等数据 | 国际城市数据 |
| 10 | 国土资源部 | https://www.mnr.gov.cn | 国土资源部信息公开报告 | 土地/矿产数据 |

---

## 二、其他细分行业数据

| 序号 | 数据源 | 网址 | 数据类型 | 说明 |
|------|--------|------|---------|------|
| 1 | 中研网数据 | https://www.chinairn.com | 医疗、房产、制造业、服务业、零售、车辆 | 全行业数据 |
| 2 | 中国报告大厅 | https://www.chinabgao.com | 各行业基础数据、调查报告、预测报告 | 种类丰富 |
| 3 | CADMAPPER | https://cadmapper.com | 世界各大城市 DXF 文件（OSM/NASA/USGS） | GIS 专用 |
| 4 | 亚马逊网络服务公共数据集 | https://registry.opendata.aws | 化学/生物/经济等多领域数据集 | AWS 托管 |
| 5 | Awesome Public Datasets | https://github.com/awesomedata/awesome-public-datasets | 自然科学、社会科学公共数据集合 | GitHub 开源项目 |
| 6 | figshare | https://figshare.com | 数据分析与研究成果共享平台 | 学术数据 |
| 7 | 英国公开数据浏览工具 | https://www.nsa.gov.uk/data | James Trimble 制作的可视化浏览工具集 | 英国数据 |
| 8 | 数据法国 | https://www.data.gouv.fr/fr | 法国各类数据可视化呈现 | 法国数据 |
| 9 | DataEye | https://www.dataeye.com | 国内游戏、汽车行业多角度行业调查报告 | 行业数据 |
| 10 | CBO 中国票房 | https://www.cbo.cn | 国内票房数据、排行、上座率 | 电影行业 |
| 11 | 易车指数 | https://index.bitauto.com | 国内汽车销售市场数据指数 | 汽车行业 |
| 12 | 高德地图 | https://ditu.amap.com | 国内交通情况、周期数据报告 | 交通数据 |
| 13 | 房天下 | https://www.fang.com | 中国指数研究院和 CREIS 中指数据 | 房地产数据 |
| 14 | 艺恩 | https://www.endata.com.cn | 中国票房数据提供方、动漫 IP 价值研究 | 文娱数据 |

---

## 三、典型赛题应用

| 赛题类型 | 推荐数据源 | 推荐算法 |
|----------|-----------|---------|
| 人口增长预测 | 国家数据 + 中国统计年鉴 | ARIMA / GM(1,1) / Logistic |
| 城市交通拥堵评价 | 高德地图 + 国家数据 | TOPSIS / 熵权 / VIKOR |
| 区域经济发展评价 | 国家数据 + 年鉴汪 | AHP + TOPSIS / PCA |
| 房地产市场分析 | 房天下 + DataEye | 回归 + 聚类 |
| 汽车市场预测 | 易车指数 + CBO 票房 | Prophet / LSTM |
| 环境污染综合评价 | 国家数据 + PM25.in | 熵权 + PROMETHEE |
| 公共服务效率评价 | 各国政府公开数据 | DEA / AHP |

---

## 四、数据获取代码模板

`python
# 国家统计数据获取（pandas + requests）
import pandas as pd
import requests

# 国家统计局 API
url = "http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2023/"
# 或使用 pandas 直接读取
df = pd.read_excel("中国统计年鉴2023.xlsx")

# 年鉴汪搜索
# https://www.nianjianwang.com 提供全国城市统计数据
`

`python
# CADMAPPER 城市 GIS 数据
import requests
url = "https://cadmapper.com/api/3.1/rest/download_ready?city_name=Beijing&country_code=CN"
`

---

> **更新日期**：2026-08-18