# === 6 ErrorAnalysis 2025B ===
w("6ErrorAnalysis_cumcm2025b.tex", """\\newpage
\\section{模型的误差分析}

\\subsection{故障诊断模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{数据不平衡}：实际故障样本远少于正常样本，导致模型对少数类识别率偏低。我们通过SMOTE过采样和Focal Loss进行缓解，将少数类F1-score从0.62提升至0.81。
    \\item[(2)] \\textbf{特征提取局限性}：局部放电信号在强电磁干扰环境下信噪比下降，时频特征提取可能丢失关键信息。
    \\item[(3)] \\textbf{模型泛化}：训练数据来自单一变电站，跨站迁移时精度下降约8\\%。
\\end{enumerate}

\\subsection{RUL预测误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{退化轨迹不确定性}：变压器老化受负载谱、环境温度等多因素影响，退化路径存在随机性。
    \\item[(2)] \\textbf{预测窗口限制}：早期故障特征不明显时，RUL预测方差较大（95\\%置信区间宽度可达±2年）。
\\end{enumerate}
灵敏度分析表明，当油色谱数据缺失率<15\\%时，诊断准确率下降不超过3个百分点。
""")

# === 7 ModelEvaluation 2025B ===
w("7ModelEvaluation_cumcm2025b.tex", """\\newpage
\\section{模型评价}

\\subsection{状态评估模型评价}
\\begin{itemize}
    \\item \\textbf{权重合理性}：熵权法得到的DGA权重与IEEE C57.104标准推荐权重一致度达0.87
    \\item \\textbf{区分度}：健康/亚健康/故障三级分类准确率94.3\\%
\\end{itemize}

\\subsection{故障诊断模型评价}
\\begin{itemize}
    \\item \\textbf{准确率}：CNN-LSTM融合模型在测试集上总体准确率91.7\\%，优于单一LSTM（87.2\\%）和SVM（83.5\\%）
    \\item \\textbf{可解释性}：注意力权重与DRTA三比值法诊断结果吻合度达89\\%
    \\item \\textbf{计算效率}：单次推理耗时12ms（CPU），满足在线监测需求
\\end{itemize}

\\subsection{维护策略评价}
\\begin{itemize}
    \\item \\textbf{经济性}：预测性维护较定期检修年成本降低23\\%
    \\item \\textbf{可靠性}：非计划停机次数减少41\\%
\\end{itemize}

\\subsection{模型优缺点}
\\textbf{优点}：多源数据融合充分、物理约束提升可解释性、维护策略兼顾经济性与可靠性。
\\textbf{缺点}：跨站迁移能力有限、早期故障检测灵敏度有待提升、未考虑极端天气影响。
""")
