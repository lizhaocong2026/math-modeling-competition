"""
灰色预测模型 GM(1,1)
适用于小样本、贫信息的不确定系统预测
"""
import numpy as np
from typing import Optional, Dict, Any


class GM11:
    """GM(1,1) 灰色预测模型"""
    
    def __init__(self):
        self.a = None  # 发展系数
        self.b = None  # 灰作用量
        self.x0 = None  # 原始序列
        self.x0_hat = None  # 拟合值
        self.c = None  # 后验差比值C
        self.p = None  # 小误差概率P
        
    def fit(self, data: np.ndarray) -> 'GM11':
        """
        拟合GM(1,1)模型
        
        参数:
            data: 原始时间序列数据
            
        返回:
            拟合后的模型实例
        """
        self.x0 = np.array(data, dtype=float)
        n = len(self.x0)
        
        # 累加生成数列 (1-AGO)
        self.x1 = np.cumsum(self.x0)
        
        # 构建矩阵 B 和 Y
        z = (self.x1[:-1] + self.x1[1:]) / 2  # 紧邻均值生成序列
        B = np.column_stack([-z, np.ones(n - 1)])
        Y = self.x0[1:]
        
        # 最小二乘估计参数 a, b
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        self.a = params[0]
        self.b = params[1]
        
        # 计算拟合值
        self.x0_hat = self._predict(n)
        
        # 计算模型精度
        self.c, self.p = self._accuracy_test()
        
        return self
    
    def _predict(self, n: int) -> np.ndarray:
        """
        预测函数
        
        参数:
            n: 预测的个数
            
        返回:
            拟合/预测值序列
        """
        # GM(1,1)白化方程解
        x1_hat = (self.x0[0] - self.b / self.a) * np.exp(-self.a * np.arange(n)) \
                 + self.b / self.a
        
        # 累减恢复得到原始序列拟合值
        x0_hat = np.zeros(n)
        x0_hat[0] = self.x0[0]
        for i in range(1, n):
            x0_hat[i] = x1_hat[i] - x1_hat[i - 1]
        
        return x0_hat
    
    def predict(self, steps: int = 5) -> Dict[str, Any]:
        """
        进行预测
        
        参数:
            steps: 预测步数
            
        返回:
            包含拟合值和预测值的字典
        """
        n = len(self.x0)
        x0_full = self._predict(n + steps)
        
        # 原始拟合值 + 预测值
        fitted = x0_full[:n]
        predicted = x0_full[n:]
        
        return {
            "fitted_values": fitted.tolist(),
            "predicted_values": predicted.tolist(),
            "model_params": {
                "发展系数_a": float(self.a),
                "灰作用量_b": float(self.b)
            },
            "accuracy": {
                "后验差比C": float(self.c) if self.c else None,
                "小误差概率P": float(self.p) if self.p else None,
                "等级": self._get_level()
            }
        }
    
    def _accuracy_test(self) -> Tuple[float, float]:
        """
        后验差检验
        
        返回:
            (C, P) 后验差比值和小误差概率
        """
        # 原始序列均值和标准差
        x0_mean = np.mean(self.x0)
        x0_std = np.std(self.x0)
        
        # 残差
        residuals = self.x0 - self.x0_hat
        e_mean = np.mean(residuals)
        e_std = np.std(residuals)
        
        # 后验差比值
        C = e_std / x0_std if x0_std > 0 else 0
        
        # 小误差概率
        small_error_threshold = x0_std * 0.6745
        p = np.sum(np.abs(residuals - e_mean) < small_error_threshold) / len(residuals)
        
        return C, p
    
    def _get_level(self) -> str:
        """
        根据C和P判断模型等级
        
        返回:
            模型精度等级字符串
        """
        if self.c is None or self.p is None:
            return "未知"
        
        if self.c < 0.35 and self.p > 0.95:
            return "好"
        elif self.c < 0.5 and self.p > 0.8:
            return "合格"
        elif self.c < 0.65 and self.p > 0.7:
            return "勉强合格"
        else:
            return "不合格"
    
    def fit_predict(self, data: np.ndarray, steps: int = 5) -> Dict[str, Any]:
        """拟合并预测"""
        self.fit(data)
        return self.predict(steps)
