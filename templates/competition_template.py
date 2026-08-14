"""
数学建模竞赛综合模板 - 完整解决方案框架
结合多种算法，提供端到端的解题流程
"""
import numpy as np
import sys
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

sys.path.insert(0, '..')

from algorithms.optimization import LinearProgramming, NonlinearProgramming
from algorithms.ga import GeneticAlgorithm
from algorithms.pso import ParticleSwarm
from algorithms.grey_model import GM11
from algorithms.ahp import AHP
from algorithms.topsis import TOPSIS
from algorithms.entropy_weight import EntropyWeight
from algorithms.pca import PCA
from utils.data_preprocessor import DataPreprocessor
from utils.helpers import Utils
from visualizations.model_viz import ModelVisualization


class CompetitionSolution:
    """
    数学建模竞赛完整解决方案框架
    
    使用示例:
        solver = CompetitionSolution(problem_type="optimization")
        result = solver.solve(data, params)
        solver.save_report("result.json")
    """
    
    def __init__(self, problem_type: str = "general"):
        """
        初始化
        
        参数:
            problem_type: 问题类型 ('optimization', 'prediction', 'evaluation', 'general')
        """
        self.problem_type = problem_type
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.preprocessor = DataPreprocessor()
        self.viz = ModelVisualization()
        
    def load_data(self, filepath: str) -> np.ndarray:
        """加载CSV数据"""
        return self.preprocessor.read_csv(filepath)
    
    def preprocess_data(
        self, 
        data: np.ndarray,
        fill_missing: bool = True,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """数据预处理"""
        return self.preprocessor.process(data, fill_missing, False, normalize)
    
    def solve_optimization(
        self,
        objective: np.ndarray,
        constraints: Dict[str, Any],
        bounds: List[Tuple[float, float]] = None,
        method: str = "lp"
    ) -> Dict[str, Any]:
        """
        求解优化问题
        
        参数:
            objective: 目标函数系数
            constraints: 约束条件
            bounds: 变量边界
            method: 求解方法 ('lp', 'nlp', 'ga')
        """
        if method == "lp":
            lp = LinearProgramming()
            return lp.solve(objective, **constraints, bounds=bounds)
        elif method == "ga":
            ga = GeneticAlgorithm()
            # 需要定义适应度函数
            raise NotImplementedError("GA优化需要自定义适应度函数")
        else:
            raise ValueError(f"不支持的优化方法: {method}")
    
    def solve_prediction(
        self,
        data: np.ndarray,
        model_type: str = "grey",
        steps: int = 5
    ) -> Dict[str, Any]:
        """
        求解预测问题
        
        参数:
            data: 历史数据
            model_type: 模型类型 ('grey', 'regression', 'arima')
            steps: 预测步数
        """
        if model_type == "grey":
            gm = GM11()
            return gm.fit_predict(data, steps)
        elif model_type == "regression":
            from algorithms.linear_regression import LinearRegression
            lr = LinearRegression()
            n = len(data)
            X = np.arange(n).reshape(-1, 1)
            return lr.fit_predict(X, data)
        else:
            raise ValueError(f"不支持的预测模型: {model_type}")
    
    def solve_evaluation(
        self,
        data: np.ndarray,
        method: str = "topsis",
        weights: np.ndarray = None,
        types: List[str] = None
    ) -> Dict[str, Any]:
        """
        求解评价问题
        
        参数:
            data: 评价矩阵
            method: 评价方法 ('topsis', 'ahp', 'entropy')
            weights: 权重（用于TOPSIS）
            types: 指标类型
        """
        if method == "topsis":
            topsis = TOPSIS(weights=weights)
            return topsis.evaluate(data, types)
        elif method == "entropy":
            entropy = EntropyWeight()
            return entropy.evaluate(data)
        elif method == "ahp":
            # AHP需要判断矩阵，这里简化处理
            raise NotImplementedError("AHP需要提供判断矩阵")
        else:
            raise ValueError(f"不支持的评价方法: {method}")
    
    def save_results(self, results: Dict[str, Any], filepath: str = None):
        """保存结果到JSON文件"""
        if filepath is None:
            filepath = f"results_{self.timestamp}.json"
        Utils.save_results(results, filepath)
    
    def generate_report(self) -> str:
        """生成解题报告"""
        report = f"""
# 数学建模竞赛解题报告

## 基本信息
- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 问题类型: {self.problem_type}

## 解题过程
1. 数据预处理
2. 模型选择与构建
3. 算法求解
4. 结果分析

## 结论
待填写...
"""
        return report
