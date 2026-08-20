w("4AssumptionAndSign_cumcm2022a.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 发酵过程在恒温条件下进行，温度波动对反应速率的影响忽略不计
    \\item 反应体系为均相溶液，不存在传质限制
    \\item 原料配比和初始浓度已知且稳定
    \\item 测量误差服从正态分布，无系统性偏差
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$C_i$ & 组分$i$的浓度 & g/L \\\\
$k_j$ & 第$j$步反应的速率常数 & L/(mol·min) \\\\
$t$ & 发酵时间 & min \\\\
$Y_{P/S}$ & 产物对底物的产率 & - \\\\
$\\mu$ & 比生长速率 & 1/min \\\\
$X$ & 菌体浓度 & g/L \\\\
\\hline
\\end{tabular}
\\end{center}
""")

w("4AssumptionAndSign_cumcm2022b.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 用户充电行为可用需求强度指数表征，忽略个体差异
    \\item 充电站建设周期远短于需求变化周期，静态优化可行
    \\item 服务半径内用户均会选择最近的充电站
    \\item 电网容量充足，建站不受供电能力约束
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$I_i$ & 需求点$i$的需求强度指数 & - \\\\
$d_{ij}$ & 需求点$i$到站点$j$的距离 & km \\\\
$r_{\\max}$ & 最大服务半径 & km \\\\
$x_j$ & 站点$j$是否建设的0-1变量 & - \\\\
$c_j$ & 站点$j$的建设成本 & 万元 \\\\
$N_{cov}$ & 覆盖的需求点数量 & 个 \\\\
$G$ & 基尼系数衡量空间公平性 & - \\\\
\\hline
\\end{tabular}
\\end{center}
""")
