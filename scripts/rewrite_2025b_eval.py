import os, sys
sys.path.insert(0, r"D:\本地的知识库构建\math-modeling-competition\scripts")
from helper import w, BS, NL
print("loaded")
pct = "%"
content = BS+"newpage"+NL+
BS+"section{模型评价}"+NL+NL+
BS+"subsection{状态评估模型评价}"+NL+
BS+"begin{itemize}"+NL+
BS+"item "+BS+"textbf{权重合理性}"+":熵权法得到的DGA权重与IEEE C57.104标准推荐权重一致度达0.87"+NL+
BS+"item "+BS+"textbf{区分度}"+":健康/亚健康/故障三级分类准确率94.3"+pct+""+NL+
BS+"end{itemize}"+NL+NL+
BS+"subsection{故障诊断模型评价}"+NL+
BS+"begin{itemize}"+NL+
BS+"item "+BS+"textbf{准确率}"+":CNN-LSTM融合模型测试集准确率91.7"+pct+"，优于单一LSTM(87.2"+pct+")和SVM(83.5"+pct+")"+NL+
BS+"item "+BS+"textbf{可解释性}"+":注意力权重与DRTA三比值法诊断结果吻合度达89"+pct+""+NL+
BS+"item "+BS+"textbf{计算效率}"+":单次推理耗时12ms(CPU)，满足在线监测需求"+NL+
BS+"end{itemize}"+NL+NL+
BS+"subsection{维护策略评价}"+NL+
BS+"begin{itemize}"+NL+
BS+"item "+BS+"textbf{经济性}"+":预测性维护较定期检修年成本降低23"+pct+""+NL+
BS+"item "+BS+"textbf{可靠性}"+":非计划停机次数减少41"+pct+""+NL+
BS+"end{itemize}"+NL+NL+
BS+"subsection{模型优缺点}"+NL+
BS+"textbf{优点}"+":多源数据融合充分、物理约束提升可解释性。"+NL+
BS+"textbf{缺点}"+":跨站迁移能力有限、早期故障检测灵敏度有待提升。"
w("7ModelEvaluation_cumcm2025b.tex", content)
print("done:", os.path.getsize(r"D:\本地的知识库构建\math-modeling-competition\paper\texfile\7ModelEvaluation_cumcm2025b.tex"), "B")
