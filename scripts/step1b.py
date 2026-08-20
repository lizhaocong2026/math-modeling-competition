import os
import sys
sys.path.insert(0, r'D:\本地的知识库构建\math-modeling-competition\scripts')
from helper import w, BS, NL

# 2025B Problem Restatement
w('2ProblemRestatement_cumcm2025b.tex', BS+'newpage'+NL+
BS+'section{问题重述}'+NL+NL+
BS+'subsection{问题背景}'+NL+
'电力变压器是电网的核心设备，其运行状态直接影响供电可靠性。变压器故障具有隐蔽性和突发性，传统定期检修模式存在过度检修或检修不足的问题。状态检修（CBM）通过实时监测变压器运行数据，实现故障预警和维护决策优化，已成为智能电网的重要研究方向。油中溶解气体分析（DGA）是变压器故障诊断的经典方法，IEEE C57.104标准规定了三种特征气体比值法。'+NL+NL+
BS+'subsection{问题提出}'+NL+
BS+'textbf{问题一}'+':基于变压器历史运行数据（油中溶解气体、负载率、环境温度等），分析各特征参数与变压器健康状态的相关性，建立变压器状态评估指标体系，给出各参数的权重分配方案。'+NL+NL+
BS+'textbf{问题二}'+':考虑多源异构数据（局部放电、油色谱、振动信号），建立融合深度学习与物理约束的变压器故障诊断模型，实现对多种故障类型的准确分类。'+NL+NL+
BS+'textbf{问题三}'+':在问题二基础上，建立变压器剩余使用寿命（RUL）预测模型，结合维护成本和停电损失，制定最优检修策略。'+NL+NL+
BS+'subsection{研究意义}'+NL+
'本研究对于提升变压器运维智能化水平、降低电网运行风险具有重要理论价值和工程应用前景。')
print('2025B restatement done')
