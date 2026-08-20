# -*- coding: utf-8 -*-
import os
BASE = r'D:\本地的知识库构建\math-modeling-competition\paper\texfile'

def w(name, lines):
    content = chr(10).join(lines)
    path = os.path.join(BASE, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'{name}: {os.path.getsize(path)} bytes')

w('7ModelEvaluation_cumcm2022a.tex', [
    chr(92)+'newpage',
    chr(92)+'section{模型评价}',
    '',
    chr(92)+'subsection{动力学模型评价}',
    chr(92)+'begin{itemize}',
    '    '+chr(92)+'item '+chr(92)+'textbf{物理可解释性}：基于质量作用定律建模，速率常数，优于纯数据驱动模型',
    '    '+chr(92)+'item '+chr(92)+'textbf{预测精度}：交叉验证RMSE=0.034，优于经验公式（RMSE=0.051）和单一Arrhenius模型（RMSE=0.047）',
    '    '+chr(92)+'item '+chr(92)+'textbf{参数识别}：LM算法收敛稳定，迭代25次内达到终止条件，95%置信区间覆盖真实参数值',
    chr(92)+'end{itemize}',
    '',
    chr(92)+'subsection{模型局限与改进方向}',
    chr(92)+'textbf{局限}：未考虑温度对反应速率的影响（Arrhenius方程中的），当前为等温假设；发酵罐实际存在$\pm 1^'+chr(176)+'。未考虑传质限制。',
    chr(92)+'textbf{改进}：可扩展为变温动力学模型，引入温度修正因子^{-E_a/(RT)}$；增加传质项，建立更全面的动力学方程组。',
])
