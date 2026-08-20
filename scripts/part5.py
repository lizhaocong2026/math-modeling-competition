# === AssumptionAndSign templates for all years ===

# 2025A - Smart Grid
w("4AssumptionAndSign_cumcm2025a.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 负荷数据完整有效，缺失值可通过线性插值补全，异常值符合3\\sigma准则
    \\item 新能源发电数据（风电、光伏）可准确预测，预测误差服从正态分布
    \\item 储能系统充放电效率恒定，不计自放电损耗
    \\item 燃气机组和燃煤机组的出力调整速率在约束范围内线性变化
    \\item 用户侧负荷响应具有弹性，价格信号可引导部分负荷转移
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$F_t$ & 时刻$t$的总负荷 & MW \\\\
$T_t, S_t, R_t$ & STL分解的趋势项、季节项、残差项 & MW \\\\
$Q, K, V$ & Self-Attention的查询、键、值矩阵 & - \\\\
$d_k$ & 键向量维度 & - \\\\
$h_t$ & 时刻$t$的状态向量 & - \\\\
$P_{\\text{load}}^t$ & 时刻$t$的负荷功率 & MW \\\\
$P_{\\text{wind}}^t, P_{\\text{solar}}^t$ & 风电/光伏出力 & MW \\\\
$E_{\\text{bat}}^t$ & 储能SOC & MWh \\\\
$u_{\\text{gas}}^t, u_{\\text{coal}}^t, u_{\\text{bat}}^t$ & 各机组出力调整量 & MW \\\\
$C_{\\text{cost}}(t)$ & 时刻$t$的调度成本 & 元 \\\\
$\\alpha, \\beta, \\gamma$ & 奖励函数权重系数 & - \\\\
$Z_1, Z_2, Z_3$ & 多目标的成本、消纳率、满意度 & - \\\\
$N, G_{\\max}$ & NSGA-II种群大小、最大迭代代数 & - \\\\
\\hline
\\end{tabular}
\\end{center}
""")

# 2025B - Transformer
w("4AssumptionAndSign_cumcm2025b.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 变压器运行数据采样频率满足故障特征提取要求，油色谱数据滞后时间已知
    \\item 故障样本的标签准确可靠，不存在标注错误
    \\item 局部放电信号在测量过程中不受外部电磁干扰的显著影响
    \\item 变压器退化过程满足Markov性，当前状态仅依赖上一状态
    \\item 维护成本参数（检修费、停电损失）可根据历史数据估计
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$C_i$ & 油中溶解气体$i$的浓度 & $\\mu$L/L \\\\
$R_{ij}$ & Duval三角图中的气体比值 & - \\\\
$X^{(k)}$ & 第$k$种数据源的特征向量 & - \\\\
$h_v^{(l)}$ & 图神经网络第$l$层节点$v$的嵌入 & - \\\\
$y_v$ & 节点$v$的故障类型标签 & - \\\\
$RUL$ & 剩余使用寿命 & 年 \\\\
$t_{\\text{fail}}$ & 故障发生时刻 & 年 \\\\
$C_{\\text{maint}}$ & 单次维护成本 & 万元 \\\\
$C_{\\text{outage}}$ & 停电损失成本 & 万元/次 \\\\
$\\theta$ & 退化过程参数 & - \\\\
\\hline
\\end{tabular}
\\end{center}
""")
