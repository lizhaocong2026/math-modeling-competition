import os
path = r'D:\\本地的知识库构建\\math-modeling-competition\\paper\\texfile'

BS = chr(92)
DOL = chr(36)
NL = chr(10)

content = BS + 'newpage' + NL
content += BS + 'section{问题重述}' + NL
content += NL
content += BS + 'subsection{问题背景}' + NL
content += '近年来，随着' + BS + 'textbf{[领域背景]}' + '快速发展，' + BS + 'textbf{[具体问题]}' + '日益突出。' + NL
content += '根据' + BS + 'textbf{[数据来源]}' + '统计，' + BS + 'textbf{[关键数据与趋势]}' + '。' + NL
content += '如何科学有效地解决这一问题，已成为' + BS + 'textbf{[相关领域]}' + '亟待攻克的难题。' + NL
content += NL
content += BS + 'subsection{问题提出}' + NL
content += '本文针对' + BS + 'textbf{[问题主题]}' + '问题，提出以下三个子问题：' + NL
content += NL
content += BS + 'textbf{问题一}' + ': 基于' + BS + 'textbf{[数据类型]}' + '，分析' + BS + 'textbf{[规律特征]}' + '，建立' + BS + 'textbf{[模型类型1]}' + '模型。设观测序列为' + DOL + BS + '{' + BS + 'y_t' + BS + '}' + BS + '_{t=1}^T' + DOL + '，需提取其' + BS + 'textbf{[特征模式]}' + '。' + NL
content += NL
content += BS + 'textbf{问题二}' + ': 在问题一的基础上，构建' + BS + 'textbf{[模型类型2]}' + '，优化' + BS + 'textbf{[决策变量]}' + DOL + 'x ' + BS + 'in ' + BS + 'mathcal{X}' + DOL + '，实现' + BS + 'textbf{[目标2]}' + '：' + NL
content += DOL + DOL + BS + 'min_{x ' + BS + 'in ' + BS + 'mathcal{X}} f(x)' + BS + 'quad' + BS + 'text{s.t.}' + BS + 'quad g_j(x) ' + BS + 'leq 0, ' + BS + 'h_k(x) = 0' + DOL + DOL + NL
content += NL
content += BS + 'textbf{问题三}' + ': 建立' + BS + 'textbf{[模型类型3]}' + '，以' + BS + 'textbf{[目标函数]}' + '为核心，求解' + BS + 'textbf{[最优解]}' + '。' + NL
content += NL
content += BS + 'subsection{研究意义}' + NL
content += '本研究对于' + BS + 'textbf{[理论意义]}' + '和' + BS + 'textbf{[实际应用价值]}' + '具有重要的理论与现实意义。' + NL

for fname in ['2ProblemRestatement.tex', 
              '2ProblemRestatement_cumcm2020a.tex', '2ProblemRestatement_cumcm2020b.tex',
              '2ProblemRestatement_cumcm2021a.tex', '2ProblemRestatement_cumcm2021b.tex',
              '2ProblemRestatement_cumcm2022a.tex', '2ProblemRestatement_cumcm2022b.tex',
              '2ProblemRestatement_cumcm2023a.tex', '2ProblemRestatement_cumcm2023b.tex',
              '2ProblemRestatement_cumcm2024a.tex', '2ProblemRestatement_cumcm2024b.tex',
              '2ProblemRestatement_cumcm2025a.tex', '2ProblemRestatement_cumcm2025b.tex']:
    with open(os.path.join(path, fname), 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'{fname}: {len(content)} bytes')
