"""
数据可视化模板
针对数学建模竞赛的专用图表
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
import warnings


class ModelVisualization:
    """模型可视化类"""
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
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
        """绘制优化算法收敛曲线"""
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
    
    def plot_tsp_result(
        self,
        points: np.ndarray,
        best_tour: List[int],
        title: str = "TSP求解结果",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制TSP路径图"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # 绘制点
        x, y = points[:, 0], points[:, 1]
        self.ax.scatter(x, y, c='red', s=100, zorder=5, label='Cities')
        
        # 绘制路径
        tour_points = points[best_tour]
        self.ax.plot(tour_points[:, 0], tour_points[:, 1], 'b-', linewidth=2, alpha=0.7)
        
        # 标注城市编号
        for i, (xi, yi) in enumerate(zip(x, y)):
            self.ax.annotate(str(i), (xi, yi), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
        
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_title(title, fontsize=14)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return self.fig
    
    def plot_pareto_front(
        self,
        objectives: np.ndarray,
        title: str = "Pareto前沿",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制Pareto前沿"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        if objectives.shape[1] == 2:
            # 二维Pareto前沿
            self.ax.scatter(objectives[:, 0], objectives[:, 1], 
                          c='blue', s=50, alpha=0.7, label='Solutions')
            
            # 标记Pareto最优解
            pareto_mask = self._is_pareto_efficient(objectives[:, 0], objectives[:, 1])
            self.ax.scatter(objectives[pareto_mask, 0], objectives[pareto_mask, 1],
                          c='red', s=100, edgecolors='black', label='Pareto Optimal', zorder=5)
            
            self.ax.set_xlabel('Objective 1', fontsize=12)
            self.ax.set_ylabel('Objective 2', fontsize=12)
        
        elif objectives.shape[1] == 3:
            # 三维Pareto前沿
            from mpl_toolkits.mplot3d import Axes3D
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                          c='blue', s=50, alpha=0.7)
            
            self.ax.set_xlabel('Objective 1')
            self.ax.set_ylabel('Objective 2')
            self.ax.set_zlabel('Objective 3')
        
        self.ax.set_title(title, fontsize=14)
        self.ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return self.fig
    
    def _is_pareto_efficient(self, costs, n_objectives=2):
        """判断是否为Pareto有效解"""
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # 检查是否有其他解支配当前解
                for j, c2 in enumerate(costs):
                    if i != j and is_efficient[j]:
                        if np.all(c2 <= c) and np.any(c2 < c):
                            is_efficient[i] = False
                            break
        
        return is_efficient
    
    def plot_3d_surface(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        title: str = "3D函数图像",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制3D曲面图"""
        self.fig = plt.figure(figsize=(12, 8))
        ax = self.fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        self.fig.colorbar(surf, shrink=0.5, aspect=10)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(title, fontsize=14)
        
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
        title: str = "热力图",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制热力图"""
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
        
        self.ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=self.ax, shrink=0.8)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return self.fig