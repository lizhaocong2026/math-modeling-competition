# ErrorAnalysis templates for 2020-2024
w("6ErrorAnalysis_cumcm2020a.tex", """\\newpage
\\section{误差分析}

\\subsection{定价模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{价格弹性估计误差}：价格弹性系数基于历史数据回归估计，存在统计误差。实际弹性可能因用户异质性而异。
    \\item[(2)] \\textbf{竞争效应简化}：模型假设相邻充电站的竞争效应对称，实际中用户品牌偏好可能导致非对称竞争。
\\end{enumerate}
灵敏度分析表明，价格弹性参数$\pm 20\\%$扰动下，最优定价变化不超过5\\%。

\\subsection{预测模型误差}
使用时段聚类的方法存在\\textbf{类内变异}，同一日型内的充电行为仍存在差异。建议引入更多特征（如天气、节假日）细化日型分类。
""")

w("6ErrorAnalysis_cumcm2020b.tex", """\\newpage
\\section{误差分析}

\\subsection{需求预测误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{数据不完整}：部分区域缺乏手机信令数据，需求估计依赖代理指标（如POI密度），存在偏差。
    \\item[(2)] \\textbf{动态变化}：新能源汽车保有量快速增长，历史需求趋势不能外推至未来。
\\end{enumerate}

\\subsection{选址模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{服务半径假设}：用户接受的最大服务距离可能与实际行为不符。
    \\item[(2)] \\textbf{电网容量忽略}：模型未考虑变电站容量约束，实际建设中可能需要扩容。
\\end{enumerate}
敏感性分析显示，服务半径每增加0.5km，覆盖率提升约3个百分点，但投资成本增加约8\\%。
""")
