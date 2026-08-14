"""数据预处理模块"""
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd


class DataPreprocessor:
    """数据预处理工具类"""
    
    def __init__(self):
        self.scaler_type = None
        self.fit_params = None
        self.transformed_data = None
        
    def normalize(
        self, 
        data: np.ndarray, 
        method: str = "min-max",
        axis: int = 0
    ) -> np.ndarray:
        """
        数据归一化
        
        参数:
            data: 输入数据
            method: 归一化方法 ('min-max', 'z-score', 'max')
            axis: 归一化轴 0=按列, 1=按行
            
        返回:
            归一化后的数据
        """
        data = np.asarray(data, dtype=float)
        
        if method == "min-max":
            if axis == 0:
                min_vals = np.min(data, axis=0)
                max_vals = np.max(data, axis=0)
                ranges = max_vals - min_vals
                ranges[ranges == 0] = 1  # 防止除零
                result = (data - min_vals) / ranges
            else:
                min_vals = np.min(data, axis=1, keepdims=True)
                max_vals = np.max(data, axis=1, keepdims=True)
                ranges = max_vals - min_vals
                ranges[ranges == 0] = 1
                result = (data - min_vals) / ranges
            
        elif method == "z-score":
            if axis == 0:
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                std[std == 0] = 1
                result = (data - mean) / std
            else:
                mean = np.mean(data, axis=1, keepdims=True)
                std = np.std(data, axis=1, keepdims=True)
                std[std == 0] = 1
                result = (data - mean) / std
                
        elif method == "max":
            if axis == 0:
                max_vals = np.max(np.abs(data), axis=0)
                max_vals[max_vals == 0] = 1
                result = data / max_vals
            else:
                max_vals = np.max(np.abs(data), axis=1, keepdims=True)
                max_vals[max_vals == 0] = 1
                result = data / max_vals
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        self.scaler_type = method
        return result
    
    def standardize(self, data: np.ndarray) -> np.ndarray:
        """Z-score标准化"""
        return self.normalize(data, method="z-score")
    
    def fill_missing(
        self, 
        data: np.ndarray, 
        method: str = "mean"
    ) -> np.ndarray:
        """
        缺失值填充
        
        参数:
            data: 输入数据（NaN表示缺失）
            method: 填充方法 ('mean', 'median', 'forward', 'interpolate')
            
        返回:
            填充后的数据
        """
        data = np.asarray(data, dtype=float)
        missing_mask = np.isnan(data)
        
        if not np.any(missing_mask):
            return data
        
        if method == "mean":
            fill_values = np.nanmean(data, axis=0)
            result = data.copy()
            for j in range(data.shape[1]):
                result[missing_mask[:, j], j] = fill_values[j]
                
        elif method == "median":
            fill_values = np.nanmedian(data, axis=0)
            result = data.copy()
            for j in range(data.shape[1]):
                result[missing_mask[:, j], j] = fill_values[j]
                
        elif method == "forward":
            result = data.copy()
            for i in range(1, data.shape[0]):
                for j in range(data.shape[1]):
                    if np.isnan(result[i, j]) and not np.isnan(result[i-1, j]):
                        result[i, j] = result[i-1, j]
            # 处理开头的缺失值
            for j in range(data.shape[1]):
                first_valid = np.where(~np.isnan(data[:, j]))[0]
                if len(first_valid) > 0:
                    result[:first_valid[0], j] = result[first_valid[0], j]
                    
        elif method == "interpolate":
            result = pd.DataFrame(data).interpolate(method='linear').values
            # 处理首尾缺失
            for j in range(data.shape[1]):
                col = result[:, j]
                valid = ~np.isnan(col)
                if not np.all(valid):
                    if not valid[0]:
                        first_valid_idx = np.where(valid)[0][0]
                        col[:first_valid_idx] = col[first_valid_idx]
                    if not valid[-1]:
                        last_valid_idx = np.where(valid)[0][-1]
                        col[last_valid_idx+1:] = col[last_valid_idx]
                    result[:, j] = col
                    
        else:
            raise ValueError(f"不支持的填充方法: {method}")
        
        return result
    
    def outlier_detection(
        self, 
        data: np.ndarray, 
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        异常值检测
        
        参数:
            data: 输入数据
            method: 检测方法 ('iqr', 'zscore')
            threshold: 阈值
            
        返回:
            (cleaned_data, outlier_mask)
        """
        data = np.asarray(data, dtype=float)
        outlier_mask = np.zeros(data.shape, dtype=bool)
        
        if method == "iqr":
            for j in range(data.shape[1]):
                q1 = np.percentile(data[:, j], 25)
                q3 = np.percentile(data[:, j], 75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                outlier_mask[:, j] = (data[:, j] < lower) | (data[:, j] > upper)
                
        elif method == "zscore":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1
            z_scores = np.abs((data - mean) / std)
            outlier_mask = z_scores > threshold
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        # 用中位数填充异常值
        cleaned_data = data.copy()
        for j in range(data.shape[1]):
            median_val = np.median(data[~outlier_mask[:, j], j]) if np.any(~outlier_mask[:, j]) else 0
            cleaned_data[outlier_mask[:, j], j] = median_val
        
        return cleaned_data, outlier_mask
    
    def process(
        self, 
        data: np.ndarray,
        fill_missing: bool = True,
        remove_outliers: bool = True,
        normalize: bool = True,
        norm_method: str = "min-max"
    ) -> Dict[str, Any]:
        """
        完整数据处理流程
        
        参数:
            data: 原始数据
            fill_missing: 是否填充缺失值
            remove_outliers: 是否去除异常值
            normalize: 是否归一化
            norm_method: 归一化方法
            
        返回:
            处理结果字典
        """
        result = {"original_shape": data.shape}
        
        processed = data.copy()
        
        if fill_missing:
            processed = self.fill_missing(processed)
            result["missing_filled"] = True
            
        if remove_outliers:
            processed, outliers = self.outlier_detection(processed)
            result["outliers_removed"] = np.sum(outliers)
            
        if normalize:
            processed = self.normalize(processed, method=norm_method)
            result["normalized"] = True
            result["norm_method"] = norm_method
        
        self.transformed_data = processed
        result["processed_data"] = processed
        result["final_shape"] = processed.shape
        
        return result
