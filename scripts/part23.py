# ErrorAnalysis 2024 + AssumptionAndSign templates
w("6ErrorAnalysis_cumcm2024a.tex", """\\newpage
\\section{误差分析}

\\subsection{能耗分解误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{负荷分类误差}：智能电表数据粒度为15分钟，无法区分同类型设备的精细用能模式。
    \\item[(2)] \\textbf{气象因素}：空调负荷与室外温度呈非线性关系，线性回归模型存在偏差。
\\end{enumerate}

\\subsection{调度模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{光伏预测误差}：短期光伏预测MAPE约15\\%，worst-case情景下调度可能不够稳健。
    \\item[(2)] \\textbf{碳价假设}：碳价采用外生给定值，实际碳市场存在价格波动。
\\end{enumerate}
分布鲁棒优化将worst-case成本控制在置信水平95\\%以内。
""")

w("6ErrorAnalysis_cumcm2024b.tex", """\\newpage
\\section{误差分析}

\\subsection{预测模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{外生变量遗漏}：Prophet模型未纳入政策突变等结构性变化因素（如2021年双碳政策）。
    \\item[(2)] \\textbf{区域异质性}：各省产业结构差异大，统一模型难以捕捉地区特性。
\\end{enumerate}

\\subsection{LMDI分解误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{边界定义}：排放因子选取（IPCC vs 实测）影响分解结果约5\\%。
    \\item[(2)] \\textbf{残余项}：尽管LMDI理论上无残差，但实际计算中存在舍入误差。
\\end{enumerate}
敏感性分析表明，经济发展贡献度估计的95\\%置信区间宽度约为点估计的±12\\%。
""")
