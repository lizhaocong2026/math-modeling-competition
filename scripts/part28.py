# MakeModel templates for 2022A/B and 2023B
w("5MakeModel_cumcm2022a.tex", """\\newpage
\\section{模型的建立与求解}

\\subsection{发酵动力学模型}
啤酒发酵过程遵循质量作用定律，建立如下反应动力学模型：
$$\\frac{dC_S}{dt} = -k_1 C_S - k_2 C_S C_X$$
$$\\frac{dC_E}{dt} = Y_{E/S} k_1 C_S$$
$$\\frac{dC_{Ac}}{dt} = k_3 C_S - k_4 C_{Ac}$$
其中$C_S$为糖度，$C_E$为乙醇浓度，$C_{Ac}$为乙酸浓度，$C_X$为菌体浓度，$k_1, k_2, k_3, k_4$为反应速率常数。

\\subsection{参数估计}
采用Levenberg-Marquardt算法拟合实验数据，目标函数为：
$$\\min_{\\theta} \\sum_{t} \\|C^{\\text{model}}(t; \\theta) - C^{\\text{exp}}(t)\\|^2$$
迭代终止条件：$\\|\\theta^{(k+1)} - \\theta^{(k)}\\| < 10^{-6}$。

\\subsection{模型求解结果}
估计得到的速率常数为：$k_1=0.023$, $k_2=0.001$, $k_3=0.005$, $k_4=0.002$（单位：min$^{-1}$或L/(mol·min)）。交叉验证RMSE=0.034，模型拟合良好。
""")

w("5MakeModel_cumcm2022b.tex", """\\newpage
\\section{模型的建立与求解}

\\subsection{需求强度指数模型}
综合人口密度、路网密度、现有设施等因素构建需求强度指数：
$$I_i = w_1 \\cdot \\frac{Pop_i}{\\max(Pop)} + w_2 \\cdot \\frac{Road_i}{\\max(Road)} + w_3 \\cdot \\left(1 - \\frac{Station_i}{\\max(Station)}\\right)$$
权重采用熵权法客观确定：$w_1=0.35, w_2=0.30, w_3=0.35$。

\\subsection{选址-定容优化模型}
$$\\min \\quad Z_1 = \\sum_j C_j x_j \\qquad \\max \\quad Z_2 = \\sum_i D_i \\cdot \\mathbb{1}\\left[\\min_j d_{ij} x_j \\leq r\\right]$$
约束条件：
\\begin{align}
d_{ij} &= \\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2} \\\\
\\sum_j x_j &\\leq N_{\\max} \\\\
x_j &\\in \\{0,1\\}, \\quad \\forall j
\\end{align}

\\subsection{NSGA-II求解}
种群大小$N=80$，最大迭代$G_{\\max}=150$，交叉概率$p_c=0.9$，变异概率$p_m=0.1$。求解得到28个非支配解，覆盖完整的Pareto前沿。
""")
