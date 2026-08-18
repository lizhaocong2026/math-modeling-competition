# 互联网数据与报告分析

> **适用赛题**：B题（互联网趋势预测）、C题（行业评价体系）  
> **关联算法**：algorithms/time_series.py · algorithms/prophet.py · algorithms/ensemble.py

---

## 一、营销查询类数据源

| 序号 | 数据源 | 网址 | 数据类型 | 说明 |
|------|--------|------|---------|------|
| 1 | 5118/chinaz | https://www.chinaz.com | 网站排名、百度收录、发展趋势 | 网站 SEO 数据分析 |
| 2 | 百度指数 | https://index.baidu.com | 网民行为数据、需求图谱 | 搜索热度趋势 |
| 3 | 微信指数 | 微信→搜索"微信指数" | 移动端搜索行为数据 | 热点话题追踪 |
| 4 | 移动观察台 | https://www.mobiletag.cn | 应用/公众号排行、用户画像 | 移动互联网数据 |
| 5 | 新榜/微小宝/易赞 | https://www.newrank.cn | 公众号排行、人群画像 | 新媒体数据分析 |
| 6 | 阿里指数 | https://zhishu.alibaba.com | 淘宝平台市场动向 | 电商消费趋势 |

---

## 二、报告分析类数据源

| 序号 | 数据源 | 网址 | 数据类型 | 说明 |
|------|--------|------|---------|------|
| 1 | 易观智库 | https://www.analysys.cn | 战略新兴产业、电商、共享经济、社交营销 | 权威的互联网数据平台 |
| 2 | 艾瑞网 | https://www.iresearch.cn | 互联网前沿资讯、艾瑞指数 | 高频更新 |
| 3 | 艾媒网 | https://www.iimedia.cn | 移动互联网研究报告 | 偏移动方向 |
| 4 | CBNData | https://www.cbnData.com | 阿里巴巴商业数据库、产业经济分析 | 阿里生态数据 |
| 5 | QuestMobile | https://www.questmobile.com.cn | APP 研究报告（周期性发布） | 移动端深度分析 |
| 6 | 阿里研究院 | https://www.aliyun.com/research | 电商趋势数据报告 | 与阿里数据相关 |
| 7 | 360 研究报告 | https://report.vcbeat.net | 移动/PC/网站/企业/诈骗安全研究 | 安全领域数据 |
| 8 | 中国互联网信息研究中心 | https://www.cnnic.net.cn | 互联网信息报告（国家批准机构） | 权威性最高 |
| 9 | 中国信通院 | https://www.caict.ac.cn | 行业发展趋势白皮书 | 角度宏观 |
| 10 | 中国互联网数据平台 | https://www.cnnic.net.cn/Data | 全国各地区互联网发展报告 | 学术倾向研究 |
| 11 | 清博大数据 | https://www.gsdata.cn | 微信/微博/头条榜单、舆情报告 | 社媒数据分析 |
| 12 | 数据观 | https://www.dgview.com | 前沿行业资讯、研究报告下载 | 综合数据平台 |
| 13 | 腾讯大数据 | https://bigdata.tencent.com | 调查研究、移动互联网报告 | 质量较高 |
| 14 | 大数据世界 | https://www.daworld.cn | 大数据资讯、应用案例、技术方案 | 技术导向 |

---

## 三、典型赛题应用

| 赛题类型 | 推荐数据源 | 推荐算法 |
|----------|-----------|---------|
| 社交媒体热度预测 | 百度指数 + 微信指数 + 清博大数据 | ARIMA / Prophet / LSTM |
| 电商平台销量预测 | 阿里指数 + CBNData | 时序分解 + 集成学习 |
| APP 活跃度分析 | QuestMobile + 移动观察台 | 聚类 + 分类 |
| 行业竞争格局评价 | 易观智库 + 艾瑞网 | AHP + TOPSIS / 熵权法 |
| 政策影响评估 | 中国信通院白皮书 | 因果推断 + 差分法 |

---

> **更新日期**：2026-08-18