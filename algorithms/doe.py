"""
蒙特卡洛实验设计 (Design of Experiments)
用于参数敏感性分析和优化
"""
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from scipy import stats


class ExperimentalDesign:
    """实验设计类"""
    
    def __init__(self, n_params: int):
        self.n_params = n_params
        self.design_matrix = None
        self.results = None
        
    def full_factorial(self, levels: int = 3) -> np.ndarray:
        """
        全因子设计
        
        参数:
            levels: 每因子的水平数
            
        返回:
            设计矩阵
        """
        n_runs = levels ** self.n_params
        design = np.zeros((n_runs, self.n_params))
        
        for i in range(n_runs):
            temp = i
            for j in range(self.n_params - 1, -1, -1):
                design[i, j] = temp % levels
                temp //= levels
        
        # 转换为 -1, 1 编码
        design = 2 * design / (levels - 1) - 1
        
        self.design_matrix = design
        return design
    
    def latin_hypercube(self, n_samples: int = 50) -> np.ndarray:
        """
        拉丁超立方抽样
        
        参数:
            n_samples: 样本数
            
        返回:
            设计矩阵
        """
        n = n_samples
        p = self.n_params
        
        design = np.zeros((n, p))
        for j in range(p):
            # 分层抽样
            intervals = np.linspace(0, 1, n + 1)
            points = np.random.uniform(intervals[:-1], intervals[1:], n)
            # 随机排列
            np.random.shuffle(points)
            design[:, j] = points
        
        self.design_matrix = design
        return design
    
    def sobol_sequence(self, n_samples: int = 100) -> np.ndarray:
        """
        Sobol序列（准蒙特卡洛）
        
        参数:
            n_samples: 样本数
            
        返回:
            设计矩阵
        """
        # 简化版：使用Halton序列代替
        design = np.zeros((n_samples, self.n_params))
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        
        for j in range(self.n_params):
            base = primes[j % len(primes)]
            for i in range(n_samples):
                n = i + 1
                result = 0
                f = 1.0 / base
                while n > 0:
                    result += f * (n % base)
                    n //= base
                    f /= base
                design[i, j] = result
        
        self.design_matrix = design
        return design
    
    def sensitivity_analysis(self, model_func: Callable, 
                            method: str = "lhs") -> Dict[str, Any]:
        """
        灵敏度分析
        
        参数:
            model_func: 模型函数
            method: 抽样方法 ('lhs', 'sobol', 'full')
            
        返回:
            灵敏度分析结果
        """
        # 生成设计矩阵
        if method == "lhs":
            design = self.latin_hypercube(100)
        elif method == "sobol":
            design = self.sobol_sequence(100)
        else:
            design = self.full_factorial(3)
        
        # 运行模型
        outputs = np.array([model_func(row) for row in design])
        
        # 计算Spearman相关系数
        n_params = self.n_params
        sensitivities = {}
        
        for i in range(n_params):
            corr, _ = stats.spearmanr(design[:, i], outputs)
            sensitivities[f"param_{i+1}"] = {
                "spearman_correlation": float(corr),
                "abs_correlation": float(abs(corr))
            }
        
        # 计算主效应
        main_effects = {}
        for i in range(n_params):
            unique_vals = np.unique(design[:, i])
            if len(unique_vals) > 1:
                groups = [outputs[design[:, i] == v] for v in unique_vals]
                main_effects[f"param_{i+1}"] = {
                    "mean": float(np.mean(outputs)),
                    "std": float(np.std(outputs)),
                    "range": float(max(outputs) - min(outputs))
                }
        
        return {
            "sensitivities": sensitivities,
            "main_effects": main_effects,
            "n_samples": len(design),
            "design_method": method
        }