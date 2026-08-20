w("7ModelEvaluation_cumcm2022a.tex", """\\newpage
\\section{模型评价}

\\subsection{模型优势}
\\begin{itemize}
    \\item \\textbf{物理可解释}：基于质量作用定律，参数有明确化学意义
    \\item \\textbf{预测精度高}：交叉验证RMSE=0.034，优于经验公式
\\end{itemize}

\\subsection{模型局限}
未考虑温度对反应速率的影响（Arrhenius方程），可扩展为变温动力学模型。
""")

w("7ModelEvaluation_cumcm2022b.tex", """\\newpage
\\section{模型评价}

\\subsection{模型优势}
\\begin{itemize}
    \\item \\textbf{覆盖率高}：充电站覆盖87\\%以上的需求点
    \\item \\textbf{公平性}：兼顾了城市中心和郊区的充电便利
\\end{itemize}

\\subsection{改进方向}
可结合GIS路网数据精确计算服务半径，引入动态充电价格机制引导错峰充电。
""")
