"""
数据包罗所有竞赛常用数据集
"""
import numpy as np
import os

DATA_DIR = os.path.dirname(__file__)

def get_population_data():
    """中国人口数据 1990-2023"""
    path = os.path.join(DATA_DIR, 'china_population_1990_2023.csv')
    return np.loadtxt(path, delimiter=',', skiprows=1)

def get_stock_data():
    """模拟股票数据"""
    path = os.path.join(DATA_DIR, 'stock_simulated.csv')
    return np.loadtxt(path, delimiter=',', skiprows=1, converters={0: lambda x: x.decode()})

def get_city_evaluation_data():
    """城市评价指标数据"""
    path = os.path.join(DATA_DIR, 'city_evaluation.csv')
    headers = []
    with open(path, 'r', encoding='utf-8') as f:
        headers = f.readline().strip().split(',')
        data = np.loadtxt(f, delimiter=',')
    return headers, data

def get_time_series():
    """时间序列数据"""
    path = os.path.join(DATA_DIR, 'time_series_200points.csv')
    return np.loadtxt(path, delimiter=',')

def get_classification_data():
    """分类数据集"""
    path = os.path.join(DATA_DIR, 'classification_data.csv')
    data = np.loadtxt(path, delimiter=',')
    return data[:, :-1], data[:, -1].astype(int)

def get_clustering_data():
    """聚类数据集"""
    path = os.path.join(DATA_DIR, 'clustering_data.csv')
    return np.loadtxt(path, delimiter=',')

def get_regression_data():
    """回归数据集"""
    path = os.path.join(DATA_DIR, 'regression_data.csv')
    data = np.loadtxt(path, delimiter=',')
    return data[:, :-1], data[:, -1]