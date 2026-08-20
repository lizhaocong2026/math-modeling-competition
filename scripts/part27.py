# AssumptionAndSign 2023 + MakeModel templates
w("4AssumptionAndSign_cumcm2023a.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 输电线路拓扑结构已知且稳定，杆塔位置数据准确
    \\item 无人机电池容量固定，飞行速度和能耗率恒定
    \\item 拍摄角度和高度在安全范围内，不影响图像质量
    \\item 航线规划不考虑天气、空域管制等动态约束
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$G=(V,E)$ & 输电线路图 & - \\\\
$v_i$ & 第$i$个杆塔节点 & - \\\\
$e_{ij}$ & 连接节点$i,j$的线路段 & m \\\\
$B$ & 无人机电池容量 & Wh \\\\
$v$ & 飞行速度 & m/s \\\\
$P_{fly}$ & 飞行功耗 & W \\\\
$h$ & 飞行高度 & m \\\\
$\\theta$ & 拍摄角度 & $^\\circ$ \\\\
$t_k$ & 在第$k$个节点的停留时间 & s \\\\
\\hline
\\end{tabular}
\\end{center}
""")

w("4AssumptionAndSign_cumcm2023b.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 交通流数据采样间隔为5分钟，能够捕捉短时波动
    \\item 相邻路口的交通流存在显著时空相关性
    \\item 信号控制周期固定，相位切换无延迟
    \\item 车辆排队服从确定性演化，忽略随机到达效应
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$A_i^k(t)$ & 时刻$t$第$k$相位第$i$方向到达车辆数 & 辆 \\\\
$D_i^k(t)$ & 时刻$t$第$k$相位第$i$方向离开车辆数 & 辆 \\\\
$q_i^k(t)$ & 饱和度 & - \\\\
$g_i^k$ & 绿灯时长 & s \\\\
$Q_i(t)$ & 排队长度 & 辆 \\\\
$\\mathbf{X}^{(k)}$ & 第$k$层图特征矩阵 & - \\\\
$\\mathbf{H}^{(k)}$ & 第$k$层节点嵌入 & - \\\\
$L$ & 总延误时间 & 辆·s \\\\
\\hline
\\end{tabular}
\\end{center}
""")
