"""工具函数模块"""
import numpy as np
from typing import Optional, Tuple, List, Any
import json
import os


class Utils:
    """工具函数集合"""
    
    @staticmethod
    def read_csv_data(filepath: str) -> np.ndarray:
        """读取CSV数据文件"""
        import csv
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    data.append([float(x) for x in row])
                except ValueError:
                    continue
        return np.array(data)
    
    @staticmethod
    def save_results(results: Dict[str, Any], filepath: str):
        """保存结果到JSON文件"""
        # 处理不可序列化的类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            return obj
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=convert)
    
    @staticmethod
    def calculate_correlation_matrix(data: np.ndarray) -> np.ndarray:
        """计算相关系数矩阵"""
        return np.corrcoef(data.T)
    
    @staticmethod
    def calculate_covariance_matrix(data: np.ndarray) -> np.ndarray:
        """计算协方差矩阵"""
        return np.cov(data.T)
    
    @staticmethod
    def time_series_split(
        data: np.ndarray, 
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        时间序列数据分割
        
        参数:
            data: 时间序列数据
            train_ratio: 训练集比例
            
        返回:
            (train_data, test_data)
        """
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]
    
    @staticmethod
    def check_model_quality(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        模型质量评估
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            
        返回:
            包含各项指标字典
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        residuals = y_true - y_pred
        
        # MAE
        mae = np.mean(np.abs(residuals))
        
        # RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # MAPE (避免除零)
        mask = y_true != 0
        mape = np.mean(np.abs(residuals[mask] / y_true[mask])) * 100 if np.any(mask) else float('inf')
        
        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "R_squared": float(r2)
        }
