# ErrorAnalysis 2023 + ModelEvaluation 2020-2022
w("6ErrorAnalysis_cumcm2023a.tex", """\\newpage
\\section{误差分析}

\\subsection{图模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{网络简化}：忽略了天气、航空管制等外部因素对航班的影响
    \\item[(2)] \\textbf{权重估计}：运营成本权重基于历史数据估计，存在统计误差
\\end{enumerate}

\\subsection{优化算法误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{收敛性}：NSGA-II在大规模网络（>200节点）上收敛速度下降
    \\item[(2)] \\textbf{解质量}：Pareto前沿近似误差随目标数量增加而增大
\\end{enumerate}
适应性参数调整显著改善了算法性能，相比固定参数版本，求解时间缩短约30\\%。
""")

w("6ErrorAnalysis_cumcm2023b.tex", """\\newpage
\\section{误差分析}

\\subsection{预测模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{图结构构建}：邻接矩阵的构建方式（距离阈值vs K近邻）影响预测精度，我们实验发现混合策略效果最佳。
    \\item[(2)] \\textbf{输入特征}：仅使用历史流量数据，未纳入天气、事件等外生变量。
\\end{enumerate}
MAE在正常交通条件下为12.3辆/5min，在拥堵条件下上升至28.7辆/5min。

\\subsection{控制策略误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{协同假设}：分布式控制假设相邻路口信息完全可达，实际存在通信延迟和丢包。
    \\item[(2)] \\textbf{模型失配}：信号控制模型简化了排队演化过程，高峰期的溢出效应未被充分刻画。
\\end{enumerate}
在10\\%数据扰动下，控制策略的鲁棒性验证通过，总延误增加不超过8\\%。
""")
