import os, sys
sys.path.insert(0, r'D:\本地的知识库构建\math-modeling-competition\scripts')
from helper import w, BS, NL

content = BS+'newpage'+NL+
BS+'section{问题分析}'+NL+NL+
BS+'subsection{问题一分析}'+NL+
'本题要求建立变压器状态评估指标体系，属于'+BS+'textbf{多指标综合评价}问题。油中溶解气体（DGA）是变压器故障诊断的经典指标，其中H2、CH4、C2H2、C2H4、C2H6、CO、CO2是关键组分。不同故障类型对应不同的气体产生特征：'+NL+
BS+'begin{itemize}'+NL+
BS+'item 局部过热：C2H4/C2H6比值升高（>3.0）'+NL+
BS+'item 放电故障：C2H2含量显著增加（>1.0 '+BS+'mu'L'/L）'+NL+
BS+'item 绝缘老化：CO和CO2持续上升，CO2/CO比值<3'+NL+
BS+'end{itemize}'+NL+
'我们需要通过相关性分析确定各参数与故障状态的关系，再采用熵权法或AHP确定权重。'+NL+NL+
BS+'subsection{问题二分析}'+NL+
'本题是'+BS+'textbf{多源异构数据融合分类}问题。三种数据源具有不同的采样频率和时间尺度。我们提出以下思路：'+NL+
BS+'enumerate'+NL+
BS+'item 分别提取各数据源的特征向量——局部放电的幅值谱/相位谱、油色谱的气体比值、振动信号的时频特征'+NL+
BS+'item 构建注意力机制加权的多通道CNN-LSTM融合网络，自动学习各数据源的贡献权重'+NL+
BS+'item 引入物理约束层（如DRTA三比值法的知识蒸馏）提升模型可解释性'+NL+
BS+'end{enumerate}'+NL+NL+
BS+'subsection{问题三分析}'+NL+
'本题是'+BS+'textbf{剩余寿命预测+维护优化}问题。RUL预测采用PHM框架，以退化轨迹建模为核心：'+NL+
BS+'itemize'+NL+
BS+'item 使用改进的Gamma过程或Wiener过程建模绝缘老化退化轨迹'+NL+
BS+'item 结合深度学习提取退化特征，建立数据驱动-机理模型混合框架'+NL+
BS+'item 维护优化建模为马尔可夫决策过程（MDP），以全寿命周期成本最小化为目标'+NL+
BS+'end{itemize}'
w('3ProblemAnalysis_cumcm2025b.tex', content)
print('done')
