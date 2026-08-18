# 分类聚类模板用法指南

> **仓库文件**：	emplates/classification_clustering_template.py  
> **适用题型**：C题分类评价、B题聚类分析  
> **关联算法**：random_forest.py · svm.py · pca.py · ensemble.py

---

## 一、快速开始

`python
from templates.classification_clustering_template import ClassificationClusteringTemplate
import numpy as np

template = ClassificationClusteringTemplate()

# KMeans聚类
labels, centers = template.kmeans_cluster(data, n_clusters=3)

# SVM分类
result = template.svm_classifier(X_train, y_train, kernel='rbf')

# 随机森林特征选择
important_features = template.feature_selection(
    X_train, y_train, n_features=5
)
`

## 二、聚类 vs 分类选择

| 条件 | 选择 | 说明 |
|------|------|------|
| 有标签数据 | 分类（SVM/RF） | 监督学习 |
| 无标签数据 | 聚类（KMeans/DBSCAN） | 无监督学习 |
| 高维数据 | PCA降维后再聚类 | 缓解维度灾难 |
| 类别不平衡 | 随机森林+采样 | 处理imbalance |

## 三、评估指标

| 任务 | 指标 | 说明 |
|------|------|------|
| 分类 | Accuracy/Precision/Recall/F1 | 类别评估 |
| 聚类 | Silhouette Score | 簇内紧密度 |
| 聚类 | Davies-Bouldin Index | 簇间分离度 |
| 特征选择 | Mutual Information | 信息增益 |

---
> **更新**：2026-08-18
