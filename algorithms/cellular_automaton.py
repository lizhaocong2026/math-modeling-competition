"""
元胞自动机
用于模拟复杂系统、交通流、传染病等
"""
import numpy as np
from typing import Callable, Optional, Dict, Any, List
import copy


class CellularAutomaton:
    """元胞自动机基类"""
    
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)
        self.history = []
        
    def initialize(self, initial_state: np.ndarray):
        """初始化状态"""
        self.grid = initial_state.copy()
        self.history.append(initial_state.copy())
    
    def step(self):
        """执行一步迭代"""
        raise NotImplementedError
    
    def run(self, n_steps: int, rule: Callable = None) -> List[np.ndarray]:
        """运行指定步数"""
        for _ in range(n_steps):
            self.step()
            self.history.append(self.grid.copy())
        return self.history
    
    def get_state(self) -> np.ndarray:
        """获取当前状态"""
        return self.grid.copy()
    
    def count_alive(self) -> int:
        """计算存活细胞数"""
        return np.sum(self.grid)


class GameOfLife(CellularAutomaton):
    """康威生命游戏"""
    
    def __init__(self, height: int, width: int, survival: List[int] = None):
        """
        参数:
            height: 网格高度
            width: 网格宽度
            survival: 存活条件，默认[2,3]表示有2或3个邻居存活
        """
        super().__init__(height, width)
        self.survival = set(survival or [2, 3])
        self.birth = set([3])
        
    def step(self):
        """生命游戏规则"""
        new_grid = self.grid.copy()
        
        for i in range(self.height):
            for j in range(self.width):
                # 计算邻居（环绕边界）
                neighbors = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni = (i + di) % self.height
                        nj = (j + dj) % self.width
                        neighbors += self.grid[ni, nj]
                
                # 应用规则
                if self.grid[i, j] == 1:
                    if neighbors in self.survival:
                        new_grid[i, j] = 1
                    else:
                        new_grid[i, j] = 0
                else:
                    if neighbors in self.birth:
                        new_grid[i, j] = 1
        
        self.grid = new_grid


class LangtonAnt(CellularAutomaton):
    """Langton蚂蚁"""
    
    def __init__(self, height: int, width: int):
        super().__init__(height, width)
        self.ant_x = height // 2
        self.ant_y = width // 2
        self.ant_dir = 0  # 0:上, 1:右, 2:下, 3:左
        
    def step(self):
        """Langton蚂蚁规则"""
        x, y = self.ant_x, self.ant_y
        direction = self.ant_dir
        
        # 当前格子颜色
        color = self.grid[x, y]
        
        if color == 0:
            # 白格：右转，翻转
            direction = (direction + 1) % 4
            self.grid[x, y] = 1
        else:
            # 黑格：左转，翻转
            direction = (direction - 1) % 4
            self.grid[x, y] = 0
        
        # 移动蚂蚁
        if direction == 0:  # 上
            self.ant_x = (x - 1) % self.height
        elif direction == 1:  # 右
            self.ant_y = (y + 1) % self.width
        elif direction == 2:  # 下
            self.ant_x = (x + 1) % self.height
        else:  # 左
            self.ant_y = (y - 1) % self.width
        self.ant_dir = direction


class SchellingSegregation(CellularAutomaton):
    """Schelling分离模型"""
    
    def __init__(self, height: int, width: int, 
                 same_threshold: float = 0.3,
                 n_same: int = None, n_diff: int = None):
        """
        参数:
            height, width: 网格尺寸
            same_threshold: 同类型邻居比例阈值
            n_same: A类型数量
            n_diff: B类型数量
        """
        super().__init__(height, width)
        self.same_threshold = same_threshold
        
        # 初始化
        n_cells = height * width
        if n_same is None:
            n_same = n_cells // 2
        if n_diff is None:
            n_diff = n_cells // 2
        
        self.grid = np.zeros((height, width), dtype=int)
        indices = np.random.choice(n_cells, n_same + n_diff, replace=False)
        self.grid.flat[indices[:n_same]] = 1  # A类型
        self.grid.flat[indices[n_same:]] = 2  # B类型
        
    def step(self):
        """执行一步"""
        happy = np.zeros((self.height, self.width), dtype=bool)
        unhappy_indices = []
        
        # 检查每个居民的满意度
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 0:  # 空位
                    continue
                
                # 计算邻居
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni = (i + di) % self.height
                        nj = (j + dj) % self.width
                        if self.grid[ni, nj] != 0:
                            neighbors.append(self.grid[ni, nj])
                
                if len(neighbors) == 0:
                    happy[i, j] = True
                    continue
                
                # 同类型比例
                same_ratio = sum(1 for n in neighbors if n == self.grid[i, j]) / len(neighbors)
                happy[i, j] = same_ratio >= (1 - self.same_threshold)
                
                if not happy[i, j]:
                    unhappy_indices.append((i, j))
        
        # 不满意者移动到空位
        for i, j in unhappy_indices:
            empty_cells = np.argwhere(self.grid == 0)
            if len(empty_cells) > 0:
                new_pos = empty_cells[np.random.randint(len(empty_cells))]
                self.grid[new_pos[0], new_pos[1]] = self.grid[i, j]
                self.grid[i, j] = 0
        
        return np.mean(happy)


class IslandModel(CellularAutomaton):
    """岛屿扩散模型"""
    
    def __init__(self, height: int, width: int, growth_rate: float = 0.5):
        super().__init__(height, width)
        self.growth_rate = growth_rate
        
    def step(self):
        """岛屿扩散"""
        new_grid = self.grid.copy()
        
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:  # 已有岛屿
                    # 向周围扩散
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni = (i + di) % self.height
                            nj = (j + dj) % self.width
                            if self.grid[ni, nj] == 0 and np.random.random() < self.growth_rate:
                                new_grid[ni, nj] = 1
        
        self.grid = new_grid