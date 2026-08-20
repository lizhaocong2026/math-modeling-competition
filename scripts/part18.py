w("6ErrorAnalysis_cumcm2022a.tex", """\\newpage
\\section{误差分析}

\\subsection{动力学参数误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{初始值敏感}：LM算法对初始速率常数敏感，需多次尝试保证全局最优。
    \\item[(2)] \\textbf{测量噪声}：浓度测量误差约$\\pm 3\\%$，影响参数精度。
\\end{enumerate}
置信区间分析表明，$k_1$和$k_3$的估计精度较高（95\\%CI宽度<10\\%），而$k_5$精度较低。

\\subsection{模型验证误差}
交叉验证RMSE=0.034，残差呈正态分布，模型拟合良好。但模型未考虑温度对反应速率的影响（Arrhenius方程），可扩展为变温动力学模型。
""")

w("6ErrorAnalysis_cumcm2022b.tex", """\\newpage
\\section{误差分析}

\\subsection{需求预测误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{数据代理偏差}：手机信令数据只能反映部分人群的充电行为，年轻用户占比高，老年用户可能被低估。
    \\item[(2)] \\textbf{空间分辨率}：数据 granularity 为街区级，无法精确到单个路段。
\\end{enumerate}

\\subsection{选址模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{覆盖模型简化}：基于欧氏距离的服务半径假设与实际情况有偏差，实际出行路径受路网约束。
    \\item[(2)] \\textbf{竞争忽略}：模型未考虑现有充电站的竞争分流效应。
\\end{enumerate}
敏感性分析表明，服务半径参数每变化0.1km，覆盖率变化约1.5个百分点，模型对该参数 moderately sensitive。
""")
