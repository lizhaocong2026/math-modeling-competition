# ModelEvaluation templates 2020-2022
w("7ModelEvaluation_cumcm2020a.tex", """\\newpage
\\section{模型评价}

\\subsection{定价模型评价}
\\begin{itemize}
    \\item \\textbf{经济性}：最优定价方案使运营商日收益提升12.3\\%，用户平均充电成本降低5.8\\%
    \\item \\textbf{实用性}：价格弹性系数易于从历史数据估计，模型具有较好的可实施性
\\end{itemize}

\\subsection{预测模型评价}
\\begin{itemize}
    \\item \\textbf{准确性}：日型聚类准确率达到89.4\\%，时段划分与实际情况吻合
    \\item \\textbf{计算效率}：BSO算法在500次迭代内收敛，单次求解耗时约2秒
\\end{itemize}

\\subsection{模型优缺点}
\\textbf{优点}：定价模型考虑了竞争效应和价格弹性，预测模型数据驱动性强。
\\textbf{缺点}：未考虑用户行为动态变化，长期预测可能存在偏差；竞争性假设为静态。
""")

w("7ModelEvaluation_cumcm2020b.tex", """\\newpage
\\section{模型评价}

\\subsection{需求评估评价}
\\begin{itemize}
    \\item \\textbf{综合性}：多源数据融合的需求指数比单一指标更能反映真实需求
    \\item \\textbf{可视化}：热力图直观展示需求空间分布，便于规划决策
\\end{itemize}

\\subsection{选址模型评价}
\\begin{itemize}
    \\item \\textbf{覆盖率}：最优方案覆盖87\\%以上的需求点，较随机布局提升35个百分点
    \\item \\textbf{公平性}：基尼系数从0.42降至0.28，空间均衡性显著改善
\\end{itemize}

\\subsection{模型优缺点}
\\textbf{优点}：多目标优化平衡了效率与公平，NSGA-II求解效率高。
\\textbf{缺点}：未考虑电网扩容成本和土地获取难度，实际建设可能需要调整。
""")
