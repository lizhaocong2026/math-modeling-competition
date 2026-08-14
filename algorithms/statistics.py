"""
统计检验与假设检验
用于数据分析验证
"""
import numpy as np
from scipy import stats
from typing import Tuple, List, Dict, Any, Optional


class HypothesisTest:
    """假设检验工具类"""
    
    @staticmethod
    def t_test_one_sample(data: np.ndarray, mu0: float = 0, 
                          alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        单样本t检验
        
        参数:
            data: 样本数据
            mu0: 假设的均值
            alternative: 备择假设 ('two-sided', 'greater', 'less')
        """
        t_stat, p_value = stats.ttest_1samp(data, mu0)
        
        if alternative == 'greater':
            p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        elif alternative == 'less':
            p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(n)
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean": mean,
            "std": std,
            "se": se,
            "n": n,
            "mu0": mu0,
            "significant": p_value < 0.05
        }
    
    @staticmethod
    def t_test_two_sample(data1: np.ndarray, data2: np.ndarray,
                          equal_var: bool = True,
                          alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        双样本t检验
        
        参数:
            data1, data2: 两组样本
            equal_var: 是否假设方差相等
            alternative: 备择假设类型
        """
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean1": mean1, "mean2": mean2,
            "std1": std1, "std2": std2,
            "n1": n1, "n2": n2,
            "significant": p_value < 0.05
        }
    
    @staticmethod
    def chi_square_test(observed: np.ndarray, expected: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """卡方检验"""
        observed = np.asarray(observed)
        
        if expected is None:
            # 拟合优度检验：假设均匀分布
            expected = np.ones_like(observed) * np.mean(observed)
        else:
            expected = np.asarray(expected)
        
        chi2, p_value = stats.chisquare(observed, f_exp=expected)
        df = len(observed) - 1
        
        return {
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": df,
            "expected": expected,
            "significant": p_value < 0.05
        }
    
    @staticmethod
    def anova_one_way(*groups: np.ndarray) -> Dict[str, Any]:
        """单因素方差分析"""
        f_stat, p_value = stats.f_oneway(*groups)
        
        n = len(groups)
        grand_mean = np.mean([np.mean(g) for g in groups])
        
        # 计算SS
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
        
        ms_between = ss_between / (n - 1)
        ms_within = ss_within / (sum(len(g) for g in groups) - n)
        
        return {
            "f_statistic": f_stat,
            "p_value": p_value,
            "n_groups": n,
            "ss_between": ss_between,
            "ss_within": ss_within,
            "ms_between": ms_between,
            "ms_within": ms_within,
            "significant": p_value < 0.05
        }


class NormalityTest:
    """正态性检验"""
    
    @staticmethod
    def shapiro_test(data: np.ndarray) -> Dict[str, Any]:
        """Shapiro-Wilk检验"""
        stat, p_value = stats.shapiro(data)
        return {"test": "Shapiro-Wilk", "statistic": stat, "p_value": p_value,
                "is_normal": p_value > 0.05}
    
    @staticmethod
    def kolmogorov_smirnov(data: np.ndarray, mu: float = None, sigma: float = None) -> Dict[str, Any]:
        """K-S检验"""
        if mu is None:
            mu = np.mean(data)
        if sigma is None:
            sigma = np.std(data)
        
        stat, p_value = stats.kstest(data, 'norm', args=(mu, sigma))
        return {"test": "Kolmogorov-Smirnov", "statistic": stat, "p_value": p_value,
                "is_normal": p_value > 0.05}
    
    @staticmethod
    def anderson_darling(data: np.ndarray) -> Dict[str, Any]:
        """Anderson-Darling检验"""
        result = stats.anderson(data, dist='norm')
        critical_values = {5: 1.931, 2.5: 2.315, 1: 2.874}
        
        return {
            "test": "Anderson-Darling",
            "statistic": result.statistic,
            "critical_values": result.critical_values,
            "significance_levels": result.significance_level,
            "is_normal": result.statistic < result.critical_values[0]
        }


class CorrelationTest:
    """相关性检验"""
    
    @staticmethod
    def pearson(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Pearson相关系数"""
        r, p_value = stats.pearsonr(x, y)
        n = len(x)
        
        # t检验
        t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else float('inf')
        
        return {
            "correlation": r,
            "p_value": p_value,
            "t_statistic": t_stat,
            "n": n,
            "significant": p_value < 0.05
        }
    
    @staticmethod
    def spearman(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Spearman秩相关"""
        rho, p_value = stats.spearmanr(x, y)
        return {"correlation": rho, "p_value": p_value, "significant": p_value < 0.05}
    
    @staticmethod
    def kendall(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Kendall Tau相关"""
        tau, p_value = stats.kendalltau(x, y)
        return {"correlation": tau, "p_value": p_value, "significant": p_value < 0.05}