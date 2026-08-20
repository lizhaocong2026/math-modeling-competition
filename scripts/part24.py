# AssumptionAndSign for 2020-2024
w("4AssumptionAndSign_cumcm2020a.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 充电车位使用数据完整有效，不存在系统性缺失
    \\item 价格弹性系数在考察期内保持稳定
    \\item 相邻充电站之间的竞争效应满足对称性假设
    \\item 用户充电商决策基于效用最大化原则
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$p_t$ & 时刻$t$的充电价格 & 元/kWh \\\\
$D(p_t)$ & 价格$p_t$下的充电需求 & 车次/小时 \\\\
$\\eta$ & 价格弹性系数 & - \\\\
$R$ & 运营商日收益 & 元 \\\\
$c_i$ & 第$i$类用户的成本敏感系数 & - \\\\
$\\alpha$ & 竞争效应参数 & - \\\\
\\hline
\\end{tabular}
\\end{center}
""")

w("4AssumptionAndSign_cumcm2020b.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 用户选择充电站遵循最短路径原则，不考虑导航偏差
    \\item 充电站服务半径为固定值，超出半径的用户不再选择该站点
    \\item 电网容量充足，建站不受供电能力限制
    \\item 建设成本与充电桩数量成线性关系
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$S_j$ & 第$j$个候选站点位置 & - \\\\
$d_{ij}$ & 需求点$i$到站点$j$的距离 & km \\\\
$r$ & 服务半径 & km \\\\
$D_i$ & 需求点$i$的需求强度 & - \\\\
$x_j$ & 站点$j$的选址决策变量 & 0-1 \\\\
$C_j$ & 站点$j$的建设成本 & 万元 \\\\
$N$ & 覆盖的需求点数量 & 个 \\\\
\\hline
\\end{tabular}
\\end{center}
""")
