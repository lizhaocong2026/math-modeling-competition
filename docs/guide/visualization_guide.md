# 可视化工具完全指南

> **仓库文件**：visualizations/model_viz.py · docs/guide/visualization_guide.md  
> **对应笔记**：[[论文写作与可视化-竞赛手册]]

---

## 一、matplotlib配色规范

### 论文标准配色（SciencePlots）

`python
import matplotlib.pyplot as plt
import scienceplots

# 论文风格
plt.style.use(['science', 'ieee'])

# 期刊风格
plt.style.use(['nature', 'science'])

# 格子风格
plt.style.use(['grid', 'notebook'])
`

### 自定义色盲友好配色

`python
import matplotlib.colors as mcolors

# ColorBrewer Set2 (色盲安全)
colorblind_palette = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
                       '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

# 自定义双色
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
`

---

## 二、竞赛常用图表

### 2.1 折线图（趋势展示）

`python
def plot_line(x, y, title, xlabel, ylabel, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, 'b-', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
`

### 2.2 柱状图（对比展示）

`python
def plot_bar(categories, values, title, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(categories, values, color='#1f77b4', edgecolor='black')
    ax.set_xlabel('Category', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(title, fontsize=12)
    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
`

### 2.3 散点图（相关性展示）

`python
def plot_scatter(x, y, title, xlabel, ylabel, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, c='#1f77b4', alpha=0.6, s=50, edgecolors='white')
    # 添加趋势线
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), 'r--', alpha=0.8, linewidth=1.5, label='trend')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
`

### 2.4 热力图（相关性矩阵）

`python
def plot_heatmap(matrix, labels, title, save_path=None):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
`

### 2.5 Pareto前沿图（多目标优化）

`python
def plot_pareto(f1, f2, title, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(f1, f2, c='#1f77b4', s=30, alpha=0.7)
    ax.set_xlabel('Objective 1', fontsize=11)
    ax.set_ylabel('Objective 2', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
`

### 2.6 雷达图（多指标对比）

`python
def plot_radar(categories, values_list, titles, save_path=None):
    import math
    n = len(categories)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (values, title, color) in enumerate(zip(values_list, titles, colors)):
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=title, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
`

---

## 三、保存设置

`python
# 高分辨率保存
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
plt.savefig('figure.pdf', bbox_inches='tight')  # 矢量图，论文推荐

# 去掉白边
plt.savefig('figure.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
`

---

> **更新日期**：2026-08-18
