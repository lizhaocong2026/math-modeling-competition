# LaTeX论文模板

## 目录结构

`
paper/
├── document.tex          # 主文件（编译入口）
├── texfile/              # 论文各节内容
│   ├── 1abstract.tex     # 摘要
│   ├── 2ProblemRestatement.tex  # 问题重述
│   ├── 3ProblemAnalysis.tex     # 问题分析
│   ├── 4AssumptionAndSign.tex   # 假设与符号说明
│   ├── 5MakeModel.tex         # 模型的建立与求解
│   ├── 6ErrorAnalysis.tex     # 误差分析
│   ├── 7ModelEvaluation.tex   # 模型评价
│   ├── 8Reference.tex       # 参考文献
│   └── 9Appendix.tex        # 附录
├── figures/              # 图表文件
└── README.md             # 本文件
`

## 编译方法

### 方法1: 使用XeLaTeX（推荐）

`ash
xelatex document.tex
bibtex 8Reference.bbl
xelatex document.tex
xelatex document.tex
`

### 方法2: 使用build脚本

`ash
./build.sh
`

## 外部优秀模板资源

1. **EmpyreanHYR/CUMCM-Latex-template** (37星)
   - 完整9节结构
   - MIT License
   - https://github.com/EmpyreanHYR/CUMCM-Latex-template

2. **Sustainable-Enjoyment/CUMCM-LaTeX-Template** (37星)
   - 含cumcmthesis.cls
   - 含示例代码
   - https://github.com/Sustainable-Enjoyment/CUMCM-LaTeX-Template

3. **personqianduixue/CUMCM_LaTeX_Template** (49星)
   - 最新版含AI使用声明
   - https://github.com/personqianduixue/CUMCM_LaTeX_Template

4. **zhanwen/MathModel** (11265星)
   - 超大型资源库
   - 含Matlab教程、PPT等
   - https://github.com/zhanwen/MathModel

## 2025新规：AI使用声明

根据2025年CUMCM新规，参赛队需在论文中声明AI工具使用情况：

`latex
% AI使用说明（2025年新增）
\begin{center}
\textbf{AI工具使用声明}

本文在以下环节使用了AI辅助工具：
\begin{itemize}
    \item 代码生成与调试：ChatGPT-4 / Claude
    \item 文献检索与整理：Perplexity
    \item 图表绘制：WPS AI / ChatGPT DALL-E
\end{itemize}
AI工具仅用于辅助，所有模型推导、数据分析、结论均为团队独立完成。
\end{center}
`

## 论文写作要点

1. **摘要最重要**：评委主要看摘要，必须包含方法+结果+验证
2. **公式编号**：重要公式必须编号，文中引用
3. **图表规范**：所有图表必须有编号和标题
4. **参考文献**：使用BibTeX管理，至少10篇
5. **附录代码**：核心代码放入附录

---

> 创建日期：2026-08-19
> 最后更新：2026-08-19
