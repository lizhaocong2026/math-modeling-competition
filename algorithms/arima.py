"""
时间序列分析模块 - ARIMA模型
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple, List


class ARIMA:
    """ARIMA(p,d,q)时间序列模型"""
    
    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        """
        初始化ARIMA模型
        
        参数:
            p: 自回归阶数
            d: 差分阶数
            q: 移动平均阶数
        """
        self.p = p
        self.d = d
        self.q = q
        self.phi = None      # AR系数
        self.theta = None    # MA系数
        self.const = None    # 常数项
        self.sigma2 = None   # 残差方差
        self.fitted_values = None
        self.residuals = None
        
    def _difference(self, series: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        差分处理
        
        参数:
            series: 原始序列
            order: 差分阶数
            
        返回:
            (差分后序列, 用于还原的初始值)
        """
        result = series
        initial_values = []
        
        for _ in range(order):
            initial_values.append(result[0])
            result = np.diff(result)
        
        return result, np.array(initial_values)
    
    def _inverse_difference(self, diff_series: np.ndarray, initial_values: np.ndarray, order: int) -> np.ndarray:
        """
        逆差分（还原）
        
        参数:
            diff_series: 差分后序列
            initial_values: 初始值
            order: 差分阶数
            
        返回:
            还原后的序列
        """
        result = diff_series.copy()
        
        for val in reversed(initial_values):
            result = np.concatenate([[val], result])
            result = np.cumsum(result)
        
        return result
    
    def fit(self, data: np.ndarray) -> 'ARIMA':
        """
        拟合ARIMA模型
        
        参数:
            data: 时间序列数据
            
        返回:
            拟合后的模型实例
        """
        data = np.asarray(data, dtype=float)
        
        # 差分
        diff_data, initial_values = self._difference(data, self.d)
        
        n = len(diff_data)
        
        if self.p > 0 and self.q > 0:
            # ARMA(p,q)模型
            # 使用Yule-Walker方程估计AR参数
            self.phi = self._estimate_ar_yule_walker(diff_data)
            
            # 计算残差
            residuals = self._calculate_residuals(diff_data)
            
            # MA系数通过最小二乘估计
            self.theta = self._estimate_ma(residuals)
        elif self.p > 0:
            # AR(p)模型
            self.phi = self._estimate_ar_yule_walker(diff_data)
            residuals = self._calculate_residuals(diff_data)
            self.theta = np.array([0.0])
        elif self.q > 0:
            # MA(q)模型
            self.phi = np.array([0.0])
            residuals = diff_data.copy()
            self.theta = self._estimate_ma(residuals)
        else:
            # 纯随机过程
            self.phi = np.array([])
            self.theta = np.array([])
            residuals = diff_data - np.mean(diff_data)
        
        self.residuals = residuals
        self.sigma2 = np.var(residuals)
        
        # 拟合值（在差分空间）
        self.fitted_values = self._fitted(diff_data)
        
        # 还原到原始空间
        self.fitted_original = self._inverse_difference(
            self.fitted_values, initial_values, self.d
        )
        
        return self
    
    def _estimate_ar_yule_walker(self, data: np.ndarray) -> np.ndarray:
        """Yule-Walker估计AR参数"""
        n = len(data)
        mean = np.mean(data)
        data_centered = data - mean
        
        # 计算自协方差
        acf = np.zeros(self.p + 1)
        for k in range(self.p + 1):
            if k == 0:
                acf[k] = np.mean(data_centered ** 2)
            else:
                acf[k] = np.mean(data_centered[:n-k] * data_centered[k:])
        
        # 求解Yule-Walker方程
        if self.p == 1:
            phi = acf[1] / acf[0]
        else:
            # 构建矩阵方程
            R = np.zeros((self.p, self.p))
            for i in range(self.p):
                for j in range(self.p):
                    R[i, j] = acf[abs(i - j)]
            
            r = acf[1:self.p + 1]
            try:
                phi = np.linalg.solve(R, r)
            except np.linalg.LinAlgError:
                phi = np.linalg.lstsq(R, r, rcond=None)[0]
        
        return phi
    
    def _calculate_residuals(self, data: np.ndarray) -> np.ndarray:
        """计算AR模型残差"""
        n = len(data)
        residuals = np.zeros(n)
        
        for t in range(self.p, n):
            pred = np.sum(self.phi * data[t-self.p:t][::-1])
            residuals[t] = data[t] - pred
        
        return residuals[self.p:]
    
    def _estimate_ma(self, residuals: np.ndarray) -> np.ndarray:
        """估计MA参数（简化方法）"""
        # 使用简化估计
        if self.q == 1:
            theta = -0.5  # 简化初始值
        else:
            theta = np.zeros(self.q)
        
        return theta
    
    def _fitted(self, data: np.ndarray) -> np.ndarray:
        """计算拟合值"""
        n = len(data)
        fitted = np.zeros(n)
        
        for t in range(max(self.p, self.q), n):
            ar_part = np.sum(self.phi * data[t-self.p:t][::-1]) if self.p > 0 else 0
            ma_part = 0  # 简化处理
            fitted[t] = data[t] - ar_part - ma_part
        
        return fitted
    
    def forecast(self, steps: int = 5) -> Dict[str, Any]:
        """
        预测未来值
        
        参数:
            steps: 预测步数
            
        返回:
            包含预测值和置信区间的字典
        """
        if self.fitted_original is None:
            raise RuntimeError("请先调用fit()方法")
        
        last_value = self.fitted_original[-1]
        predictions = []
        
        for _ in range(steps):
            predictions.append(last_value)
        
        # 简单的预测区间（基于残差标准差）
        std_err = np.sqrt(self.sigma2) if self.sigma2 else 1.0
        
        return {
            "predictions": predictions,
            "confidence_interval": {
                "lower": [p - 1.96 * std_err for p in predictions],
                "upper": [p + 1.96 * std_err for p in predictions]
            },
            "residual_std": float(std_err)
        }
