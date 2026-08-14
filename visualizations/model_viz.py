"""
可视化模块
提供数学建模竞赛常用图表绘制功能
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
import warnings


class ModelVisualization:
    """模型可视化类"""
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """
        初始化可视化类
        
        参数:
            style: matplotlib样式
        """
        try:
            plt.style.use(style)
        except:
            pass
        
        self.fig = None
        self.ax = None
        
    def plot_optimization_convergence(
        self,
        best_history: List[float],
        avg_history: Optional[List[float]] = None,
        title: str = "优化收敛曲线",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制优化算法收敛曲线
        
        参数:
            best_history: 每代最佳适应度历史
            avg_history: 每代平均适应度历史
            title: 图表标题
            save_path: 保存路径
            
        返回:
            matplotlib Figure对象
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        generations = np.arange(1, len(best_history) + 1)
        
        self.ax.plot(generations, best_history, 'b-', linewidth=2, label='Best Fitness')
        
        if avg_history:
            self.ax.plot(generations, avg_history, 'r--', linewidth=1.5, label='Avg Fitness')
        
        self.ax.set_xlabel('Generation', fontsize=12)
        self.ax.set_ylabel('Fitness', fontsize=12)
        self.ax.set_title(title, fontsize=14)
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return self.fig
    
    def plot_comparison_bar(
        self,
        labels: List[str],
        values: List[List[float]],
        legend_labels: Optional[List[str]] = None,
        title: str = "方案对比",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制方案对比柱状图
        
        参数:
            labels: 方案标签
            values: 各方案得分列表（每个方案一个列表）
            legend_labels: 图例标签
            title: 图表标题
            save_path: 保存路径
        """
        n_schemes = len(labels)
        n_methods = len(values)
        x = np.arange(n_schemes)
        width = 0.8 / n_methods if n_methods > 1 else 0.6
        
        self.fig, self.ax = plt.subplots(figsize=(max(8, n_schemes * 1.2), 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_methods))
        
        for i, (method_values, color) in enumerate(zip(values, colors)):
            offset = (i - n_methods/2 + 0.5) * width
            bars = self.ax.bar(
                x + offset, 
                method_values, 
                width, 
                label=legend_labels[i] if legend_labels else f'Method {i+1}',
                color=color,
                edgecolor='black',
                alpha=0.8
            )
            # 添加数值标签
            for bar, val in zip(bars, method_values):
                self.ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.01,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=9
                )
        
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels)
        self.ax.set_ylabel('Score', fontsize=12)
        self.ax.set_title(title, fontsize=14)
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return self.fig
    
    def plot_heatmap(
        self,
        matrix: np.ndarray,
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制热力图
        
        参数:
            matrix: 数据矩阵
            titles: 行列标题
            save_path: 保存路径
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        im = self.ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto')
        
        if titles:
            self.ax.set_xticks(range(len(titles)))
            self.ax.set_xticklabels(titles, rotation=45, ha='right')
            self.ax.set_yticks(range(len(titles)))
            self.ax.set_yticklabels(titles)
        
        # 添加数值标注
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = self.ax.text(j, i, f'{matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        self.ax.set_title('Correlation Matrix', fontsize=14)
        plt.colorbar(im, ax=self.ax, shrink=0.8)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return self.fig
    
    def plot_error_bars(
        self,
        means: List[float],
        errors: List[float],
        labels: List[str],
        title: str = "误差棒图",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制误差棒图
        
        参数:
            means: 均值列表
            errors: 误差列表
            labels: 标签列表
            title: 图表标题
            save_path: 保存路径
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(means))
        bars = self.ax.bar(x, means, yerr=errors, capsize=5, 
                          color='steelblue', edgecolor='black', alpha=0.7)
        
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, rotation=30)
        self.ax.set_ylabel('Value', fontsize=12)
        self.ax.set_title(title, fontsize=14)
        self.ax.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return self.fig
