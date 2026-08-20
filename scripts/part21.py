w("7ModelEvaluation_cumcm2021a.tex", """\\newpage
\\section{模型评价}

\\subsection{频域模型评价}
\\begin{itemize}
    \\item \\textbf{拟合精度}：一阶惯性模型在0-5Hz频段内的拟合误差<5\\%
    \\item \\textbf{响应速度}：储能系统频响时间常数约0.5s，满足调频快速响应要求
\\end{itemize}

\\subsection{容量配置评价}
\\begin{itemize}
    \\item \\textbf{经济性}：最优容量配置方案内部收益率IRR=18.7\\%，投资回收期约5.2年
    \\item \\textbf{可靠性}：在95\\%置信水平下，储能系统可满足调频需求
\\end{itemize}

\\subsection{模型优缺点}
\\textbf{优点}：MPC框架能处理多约束优化，滚动求解适应实时调度需求。
\\textbf{缺点}：一阶模型精度有限，高阶模型辨识复杂；市场价格预测不确定性大。
""")

w("7ModelEvaluation_cumcm2021b.tex", """\\newpage
\\section{模型评价}

\\subsection{预测模型评价}
\\begin{itemize}
    \\item \\textbf{精度优势}：GraphSAGE-LSTM混合模型MAE=10.2辆/5min，较LSTM降低15\\%
    \\item \\textbf{泛化能力}：在测试集上MAPE稳定在8\\%-12\\%区间
    \\item \\textbf{计算效率}：单步预测耗时约0.05秒（CPU），满足实时预测需求
\\end{itemize}

\\subsection{控制策略评价}
\\begin{itemize}
    \\item \\textbf{收敛性}：Q-Learning在200轮后reward曲线趋于平稳
    \\item \\textbf{对比优势}：较固定配时方案，总延误降低22\\%，停车次数减少18\\%
\\end{itemize}

\\subsection{模型优缺点}
\\textbf{优点}：时空特征提取充分、控制策略适应性强、计算效率满足实时要求。
\\textbf{缺点}：纯数据驱动缺乏物理可解释性、未考虑行人过街和公交优先。
""")
