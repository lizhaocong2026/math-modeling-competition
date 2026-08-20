w("6ErrorAnalysis_cumcm2021a.tex", """\\newpage
\\section{误差分析}

\\subsection{频域参数辨识误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{模型阶次}：一阶惯性环节无法完全刻画储能的动态响应，高阶模型精度更高但辨识难度增加。
    \\item[(2)] \\textbf{测量噪声}：频率测量存在传感器噪声，影响参数估计精度。
\\end{enumerate}

\\subsection{容量配置误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{循环寿命估计}：实际电池衰减受温度、深度放电等因素影响，实验室数据可能高估寿命。
    \\item[(2)] \\textbf{收益预测}：调频市场价格波动大，历史收益不能完全代表未来收益。
\\end{enumerate}
通过蒙特卡洛模拟评估不确定性，容量配置结果的标准差约为最优值的8\\%。
""")

w("6ErrorAnalysis_cumcm2021b.tex", """\\newpage
\\section{误差分析}

\\subsection{预测模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{图结构简化}：固定邻接矩阵无法反映交通流的动态传播特性，实际路网存在时变拓扑。
    \\item[(2)] \\textbf{特征工程}：手工设计的时空特征可能遗漏重要信息，端到端学习可以更充分地利用数据。
\\end{enumerate}

\\subsection{控制策略误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{模型简化}：信号控制模型忽略行人过街、公交优先等复杂因素。
    \\item[(2)] \\textbf{延迟影响}：实际控制系统存在通信和计算延迟，影响控制效果。
\\end{enumerate}
测试表明，在10\\%流量扰动下，协同控制策略仍能将总延误控制在基准值的1.15倍以内。
""")
