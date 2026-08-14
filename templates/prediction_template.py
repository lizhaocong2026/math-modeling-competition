"""
数学建模竞赛模板 - 预测类问题
适用于CUMCM B题（预测类题目）
"""
import numpy as np
from typing import Dict, Any, List
import sys
sys.path.insert(0, '..')

from algorithms.grey_model import GM11
from algorithms.linear_regression import LinearRegression
from algorithms.polynomial_regression import PolynomialRegression
from utils.data_preprocessor import DataPreprocessor
from visualizations.model_viz import ModelVisualization


class PredictionProblemTemplate:
    """预测类问题模板"""
    
    def __init__(self):
        self.gm11 = GM11()
        self.lr = LinearRegression()
        self.pr = PolynomialRegression()
        self.preprocessor = DataPreprocessor()
        self.viz = ModelVisualization()
        
    def grey_prediction(
        self, 
        data: np.ndarray, 
        steps: int = 5
    ) -> Dict[str, Any]:
        """
        灰色预测 GM(1,1)
        
        参数:
            data: 原始时间序列数据
            steps: 预测步数
            
        返回:
            包含拟合值和预测值的字典
        """
        result = self.gm11.fit_predict(data, steps)
        
        print("=" * 50)
        print("GM(1,1)灰色预测结果")
        print("=" * 50)
        print(f"发展系数 a: {result['model_params']['发展系数_a']:.6f}")
        print(f"灰作用量 b: {result['model_params']['灰作用量_b']:.6f}")
        print(f"拟合值: {result['fitted_values']}")
        print(f"预测值: {result['predicted_values']}")
        print(f"精度等级: {result['accuracy']['等级']}")
        
        # 绘制拟合与预测对比图
        self._plot_prediction_comparison(
            data, 
            result['fitted_values'], 
            result['predicted_values'],
            title="GM(1,1)灰色预测"
        )
        
        return result
    
    def linear_forecast(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        forecast_steps: int = 5
    ) -> Dict[str, Any]:
        """
        线性回归预测
        
        参数:
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征（可选）
            forecast_steps: 预测步数
        """
        # 拟合模型
        result = self.lr.fit_predict(X_train, y_train)
        
        print("=" * 50)
        print("线性回归预测结果")
        print("=" * 50)
        print(f"系数: {result['coefficients']}")
        print(f"截距: {result['intercept']:.6f}")
        print(f"R²: {result['r_squared']:.6f}")
        
        # 预测
        if X_test is not None:
            forecast_result = self.lr.forecast(X_test)
            result['forecast'] = forecast_result
        
        return result
    
    def polynomial_forecast(
        self,
        data: np.ndarray,
        degree: int = 2,
        forecast_steps: int = 5
    ) -> Dict[str, Any]:
        """
        多项式回归预测
        
        参数:
            data: 时间序列数据
            degree: 多项式阶数
            forecast_steps: 预测步数
        """
        n = len(data)
        X = np.arange(n).reshape(-1, 1)
        
        result = self.pr.fit_predict(X, data)
        
        print("=" * 50)
        print("多项式回归预测结果")
        print("=" * 50)
        print(f"多项式阶数: {result['degree']}")
        print(f"系数: {result['coefficients']}")
        print(f"R²: {result['r_squared']:.6f}")
        
        # 外推预测
        X_future = np.arange(n, n + forecast_steps).reshape(-1, 1)
        predictions = self.pr.predict(X_future)
        result['future_predictions'] = predictions.tolist()
        
        # 绘制拟合曲线
        self._plot_polynomial_fit(X, data, result['coefficients'], degree)
        
        return result
    
    def compare_prediction_models(
        self,
        data: np.ndarray,
        test_ratio: float = 0.2
    ) -> Dict[str, Any]:
        """
        对比多种预测模型
        
        参数:
            data: 时间序列数据
            test_ratio: 测试集比例
        """
        n = len(data)
        split_idx = int(n * (1 - test_ratio))
        
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        results = {}
        
        # GM(1,1)
        try:
            gm_result = self.gm11.fit_predict(train_data, steps=len(test_data))
            gm_pred = np.array(gm_result['predicted_values'][:len(test_data)])
            gm_metrics = self._calculate_metrics(test_data, gm_pred)
            results['GM(1,1)'] = {**gm_metrics, 'params': gm_result['model_params']}
        except Exception as e:
            results['GM(1,1)'] = {'error': str(e)}
        
        # 线性回归
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        try:
            lr_result = self.lr.fit_predict(X_train, train_data)
            X_test = np.arange(split_idx, n).reshape(-1, 1)
            lr_pred = self.lr.predict(X_test)
            lr_metrics = self._calculate_metrics(test_data, lr_pred)
            results['Linear Regression'] = lr_metrics
        except Exception as e:
            results['Linear Regression'] = {'error': str(e)}
        
        # 多项式回归 (3阶)
        try:
            pr_result = self.pr.fit_predict(X_train, train_data, degree=3)
            X_test = np.arange(split_idx, n).reshape(-1, 1)
            pr_pred = self.pr.predict(X_test)
            pr_metrics = self._calculate_metrics(test_data, pr_pred)
            results['Polynomial (3rd)'] = pr_metrics
        except Exception as e:
            results['Polynomial (3rd)'] = {'error': str(e)}
        
        # 打印对比结果
        print("\n" + "=" * 60)
        print("预测模型对比结果")
        print("=" * 60)
        print(f"{'模型':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10}")
        print("-" * 60)
        
        for model_name, metrics in results.items():
            if 'error' in metrics:
                print(f"{model_name:<20} {'Error':<68}")
            else:
                print(f"{model_name:<20} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} "
                      f"{metrics['MAPE']:<10.4f} {metrics['R²']:<10.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算预测评估指标"""
        residuals = y_true - y_pred
        
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        mask = y_true != 0
        mape = np.mean(np.abs(residuals[mask] / y_true[mask])) * 100 if np.any(mask) else float('inf')
        
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'R²': float(r2)
        }
    
    def _plot_prediction_comparison(
        self, 
        original: np.ndarray, 
        fitted: np.ndarray, 
        predicted: np.ndarray,
        title: str = "预测对比"
    ):
        """绘制预测对比图"""
        n_orig = len(original)
        n_total = n_orig + len(predicted)
        
        x = np.arange(n_total)
        y_full = np.concatenate([original, np.zeros(len(predicted))])
        
        # 拟合值
        y_fit = np.concatenate([fitted, np.zeros(len(predicted))])
        
        # 预测值
        y_pred_full = np.concatenate([np.full(len(fitted), np.nan), predicted])
        
        self.viz.fig, self.viz.ax = plt.subplots(figsize=(12, 6))
        
        self.viz.ax.plot(x[:n_orig], y_full[:n_orig], 'bo-', label='Original', markersize=6)
        self.viz.ax.plot(x[:n_orig], y_fit[:n_orig], 'r--', label='Fitted', linewidth=2)
        self.viz.ax.plot(x[n_orig:], y_pred_full[n_orig:], 'g^-', label='Predicted', markersize=8, linewidth=2)
        
        self.viz.ax.axvline(x=n_orig-0.5, color='gray', linestyle=':', alpha=0.5)
        self.viz.ax.set_xlabel('Time Step')
        self.viz.ax.set_ylabel('Value')
        self.viz.ax.set_title(title)
        self.viz.ax.legend(loc='best')
        self.viz.ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_polynomial_fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        coefficients: List[float], 
        degree: int
    ):
        """绘制多项式拟合图"""
        self.viz.fig, self.viz.ax = plt.subplots(figsize=(10, 6))
        
        # 绘制原始数据点
        self.viz.ax.scatter(X, y, color='blue', label='Data', s=50, zorder=5)
        
        # 绘制拟合曲线
        x_smooth = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_smooth = self.pr._build_polynomial_matrix(x_smooth, degree).dot(coefficients)
        self.viz.ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'Polynomial (degree={degree})')
        
        self.viz.ax.set_xlabel('X')
        self.viz.ax.set_ylabel('Y')
        self.viz.ax.set_title('Polynomial Regression Fit')
        self.viz.ax.legend()
        self.viz.ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
