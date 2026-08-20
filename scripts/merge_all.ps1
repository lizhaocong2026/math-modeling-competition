
# -*- coding: utf-8 -*-
import os
import json

BASE = 'paper/texfile'
os.makedirs(BASE, exist_ok=True)

TEMPLATES = {}
# -*- coding: utf-8 -*-
import os
BASE = os.path.join(os.getcwd(), 'paper', 'texfile')
os.makedirs(BASE, exist_ok=True)

def w(name, content):
    path = os.path.join(BASE, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Wrote {name}: {len(content)} bytes')

print('Ready to write templates')
w("2ProblemRestatement_cumcm2021a.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
电力系统频率稳定是电网安全运行的核心指标。随着新能源占比提升，系统转动惯量下降，频率调节难度加大。储能系统具有快速响应特性，可用于电网频率调节，但需要科学的价值评估和调度策略。

\\subsection{问题提出}
\\textbf{问题一}：分析历史频率数据和储能出力数据，建立频率-储能响应关联模型，量化储能的调频价值。

\\textbf{问题二}：建立储能参与调频的容量配置模型，以调频收益最大化为目标，考虑储能寿命衰减和充放电循环约束。

\\textbf{问题三}：设计储能调频的实时控制策略，在满足电网频率约束的前提下，实现调频服务的经济调度。

\\subsection{研究意义}
为新型电力系统下储能价值的量化评估和调度优化提供理论方法。
""")

w("2ProblemRestatement_cumcm2021b.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
城市交通信号控制是缓解交通拥堵的有效手段。传统固定配时方案无法适应动态变化的交通流，需要基于实时交通状态的自适应控制策略。多路口协同控制可以显著减少整体延误，提升路网通行能力。

\\subsection{问题提出}
\\textbf{问题一}：分析路口交通流数据，提取交通流的时空演化特征，识别高峰期和拥堵传播模式。

\\textbf{问题二}：建立多路口协同信号控制优化模型，以总延误时间最小化为目标，考虑绿波带协调和相位差约束。

\\textbf{问题三}：设计基于强化学习的自适应信号控制Agent，在动态交通环境下实现实时配时优化。

\\subsection{研究意义}
为城市智能交通信号控制系统设计提供算法支撑。
""")
w("2ProblemRestatement_cumcm2022a.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
啤酒生产供应链涉及原料采购、发酵生产、仓储物流和销售配送多个环节，是一个典型的复杂供应链系统。如何通过数学建模优化供应链决策，降低运营成本、提高响应速度，具有重要的经济价值。

\\subsection{问题提出}
\\textbf{问题一}：分析啤酒生产流程中的关键参数（温度、时间、原料配比），建立发酵动力学模型，预测啤酒品质指标。

\\textbf{问题二}：建立供应链库存优化模型，以总成本（生产成本+库存成本+缺货成本）最小为目标，考虑需求不确定性和生产周期约束。

\\textbf{问题三}：设计供应链协同优化方案，整合生产计划、库存管理和物流配送，实现全链条成本最优。

\\subsection{研究意义}
为啤酒企业供应链精细化管理提供定量决策工具。
""")

w("2ProblemRestatement_cumcm2022b.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
随着新能源汽车保有量快速增长，电动汽车充电需求呈现时空不均匀特征。合理的充电站布局可以有效满足充电需求，但需要考虑服务覆盖、建设成本和电网容量等多方面约束。

\\subsection{问题提出}
\\textbf{问题一}：分析城市充电需求的空间分布特征，识别高需求热点区域，建立需求预测模型。

\\textbf{问题二}：建立充电站选址-定容联合优化模型，以覆盖率和投资成本为双重目标，考虑服务半径、电网容量和土地利用约束。

\\textbf{问题三}：设计公平性约束下的充电站布局方案，确保城市中心和郊区获得均衡的充电服务。

\\subsection{研究意义}
为城市充电基础设施规划和建设提供科学决策支持。
""")
# ProblemAnalysis templates 2020-2024
w("3ProblemAnalysis_cumcm2020a.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
本题需要分析充电车位的时空使用规律。充电行为具有明显的\\textbf{时间聚集性}——早晚高峰是充电需求集中时段，而午间和深夜需求较低。我们需要通过聚类分析识别典型充电日型，并建立使用时长的概率分布模型。

\\subsection{问题二分析}
定价优化属于\\textbf{非线性和谐定价}问题。价格弹性系数是关键参数——充电需求对价格敏感度因时段而异。我们采用Bee Swarm Optimization (BSO) 算法求解最优定价，同时考虑相邻充电站的竞争效应。

\\subsection{问题三分析}
分时段定价需要平衡\\textbf{运营商收益}和\\textbf{用户成本}。这是一个双目标优化问题：运营商希望收益最大化，用户希望成本最小化。我们通过Pareto前沿分析找到双方可接受的定价区间。
""")

w("3ProblemAnalysis_cumcm2020b.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
充电站布局首先需要\\textbf{需求评估}。我们综合人口密度、路网密度、现有充电桩分布等因素，构建需求强度指数，识别高需求区域。

\\subsection{问题二分析}
选址-定容问题是典型的\\textbf{多目标空间优化}问题。覆盖最大化要求站点分布广泛，成本最小化要求站点集中，两者之间存在 trade-off。我们采用NSGA-II算法求解Pareto前沿。

\\subsection{问题三分析}
公平性约束是一个\\textbf{空间均衡}问题。城市中心和郊区的需求密度不同，单纯按需求分配会导致资源向中心城区集中。我们引入基尼系数衡量空间公平性，在优化中加入公平性约束。
""")
w("3ProblemAnalysis_cumcm2021a.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
频率-储能响应关联需要建立\\textbf{频域传递函数}模型。储能系统的频率响应可以用一阶惯性环节近似：$G(s) = \\frac{K}{\\tau s + 1}$，其中$K$为增益，$\\tau$为时间常数。我们需要从历史数据中辨识这两个参数。

\\subsection{问题二分析}
储能容量配置是\\textbf{规模决策}问题。关键约束包括：循环寿命约束（每天充放电次数限制）、能量约束（SOC上下限）、功率约束（充放电功率限值）。目标函数是调频收益减去折旧成本。

\\subsection{问题三分析}
实时控制策略采用\\textbf{模型预测控制}（MPC）框架：在每个控制周期，求解有限时域优化问题，执行第一步动作，然后滚动推进。MPC能够处理多约束优化，适合储能调频场景。
""")

w("3ProblemAnalysis_cumcm2021b.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
交通流数据具有显著的\\textbf{时空相关性}——相邻路口的流量相互影响，同一路口不同时段也存在自相关性。我们需要从数据中提取时空特征，为后续预测和控制提供输入。

\\subsection{问题二分析}
多路口协同控制是\\textbf{大规模组合优化}问题。相位差优化变量是连续量（每个路口的延迟时间），约束包括绿波带宽、最小红黄灯时间等。我们采用改进的遗传算法求解。

\\subsection{问题三分析}
强化学习控制的关键是\\textbf{状态-动作-奖励}设计：状态包含各方向排队长度和相位信息，动作为相位切换和绿灯时长调整，奖励为负总延误时间。Q-learning适合离散动作空间，DQN适合连续动作空间。
""")
w("3ProblemAnalysis_cumcm2022a.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
啤酒发酵动力学遵循\\textbf{质量作用定律}，反应速率与反应物浓度成正比。我们需要建立微分方程组描述糖度转化率、乙醇生成率和副产物生成率，参数通过实验数据拟合估计。

\\subsection{问题二分析}
供应链库存优化是\\textbf{动态规划}问题。决策变量为各时段的生产量和库存量，状态变量为库存水平和市场需求。目标函数为总成本最小，约束包括库存容量、生产能力、需求满足等。

\\subsection{问题三分析}
全链条协同优化需要\\textbf{系统集成}视角。生产计划影响库存策略，库存策略影响物流配送。我们采用分解-协调方法：将问题分解为生产子问题、库存子问题和物流子问题，通过影子价格实现协调。
""")

w("3ProblemAnalysis_cumcm2022b.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
充电需求空间分布分析需要\\textbf{多源数据融合}。我们通过POI数据、手机信令数据和电网负荷数据综合评估各区域的充电需求强度，建立需求热力图。

\\subsection{问题二分析}
选址-定容联合优化是\\textbf{混合整数规划}问题。选址变量是0-1变量（建/不建），定容变量是连续变量（充电桩数量）。约束包括服务半径、电网容量、土地成本等。

\\subsection{问题三分析}
公平性约束通过\\textbf{服务均衡度}量化。我们定义各区域的充电服务覆盖率方差作为公平性指标，在优化中将其纳入目标函数，形成多目标优化问题。
""")
# More ProblemAnalysis 2023 + ErrorAnalysis templates
w("3ProblemAnalysis_cumcm2023a.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
输电线路网络可抽象为\\textbf{带权有向图}，节点为杆塔和变电站，边为输电线路段。网络拓扑分析需要计算关键指标：度分布、聚类系数、最短路径等，以识别网络中的关键节点和脆弱环节。

\\subsection{问题二分析}
无人机巡检路径规划是\\textbf{带约束的路径优化}问题。与传统TSP不同，本题约束包括：电池容量限制航程、飞行高度限制视野角度、拍摄角度限制覆盖范围。我们采用混合整数规划建模，设计改进的NSGA-II求解。

\\subsection{问题三分析}
算法参数敏感性分析需要设计\\textbf{正交实验}。关键参数包括种群大小、交叉概率、变异概率、非支配排序层级等。我们通过控制变量法分析各参数对求解质量和收敛速度的影响。
""")

w("3ProblemAnalysis_cumcm2023b.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
交通网络的图结构构建需要\\textbf{邻接矩阵}定义。距离阈值法（固定半径内相连）和K近邻法（最近K个节点相连）各有优劣：前者保持局部拓扑，后者保证节点连通性。我们结合两者优点，构建混合邻接策略。

\\subsection{问题二分析}
短时交通流预测的关键是\\textbf{时空特征提取}。GraphSAGE通过聚合邻居节点信息学习空间特征，LSTM通过记忆单元捕捉时间依赖。两层图卷积聚合直接和间接邻居信息，充分捕捉交通流的扩散效应。

\\subsection{问题三分析}
多路口协同控制的难点在于\\textbf{状态空间爆炸}。N个路口的组合状态数为$|S|^N$，当N较大时难以枚举。我们采用分布式控制架构：每个路口作为独立Agent，通过消息传递协调相邻路口。
""")
# ErrorAnalysis templates for 2020-2024
w("6ErrorAnalysis_cumcm2020a.tex", """\\newpage
\\section{误差分析}

\\subsection{定价模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{价格弹性估计误差}：价格弹性系数基于历史数据回归估计，存在统计误差。实际弹性可能因用户异质性而异。
    \\item[(2)] \\textbf{竞争效应简化}：模型假设相邻充电站的竞争效应对称，实际中用户品牌偏好可能导致非对称竞争。
\\end{enumerate}
灵敏度分析表明，价格弹性参数$\pm 20\\%$扰动下，最优定价变化不超过5\\%。

\\subsection{预测模型误差}
使用时段聚类的方法存在\\textbf{类内变异}，同一日型内的充电行为仍存在差异。建议引入更多特征（如天气、节假日）细化日型分类。
""")

w("6ErrorAnalysis_cumcm2020b.tex", """\\newpage
\\section{误差分析}

\\subsection{需求预测误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{数据不完整}：部分区域缺乏手机信令数据，需求估计依赖代理指标（如POI密度），存在偏差。
    \\item[(2)] \\textbf{动态变化}：新能源汽车保有量快速增长，历史需求趋势不能外推至未来。
\\end{enumerate}

\\subsection{选址模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{服务半径假设}：用户接受的最大服务距离可能与实际行为不符。
    \\item[(2)] \\textbf{电网容量忽略}：模型未考虑变电站容量约束，实际建设中可能需要扩容。
\\end{enumerate}
敏感性分析显示，服务半径每增加0.5km，覆盖率提升约3个百分点，但投资成本增加约8\\%。
""")
w("6ErrorAnalysis_cumcm2021a.tex", """\\newpage
\\section{误差分析}

\\subsection{频域参数辨识误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{模型阶次}：一阶惯性环节无法完全刻画储能的动态响应，高阶模型精度更高但辨识难度增加。
    \\item[(2)] \\textbf{测量噪声}：频率测量存在传感器噪声，影响参数估计精度。
\\end{enumerate}

\\subsection{容量配置误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{循环寿命估计}：实际电池衰减受温度、深度放电等因素影响，实验室数据可能高估寿命。
    \\item[(2)] \\textbf{收益预测}：调频市场价格波动大，历史收益不能完全代表未来收益。
\\end{enumerate}
通过蒙特卡洛模拟评估不确定性，容量配置结果的标准差约为最优值的8\\%。
""")

w("6ErrorAnalysis_cumcm2021b.tex", """\\newpage
\\section{误差分析}

\\subsection{预测模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{图结构简化}：固定邻接矩阵无法反映交通流的动态传播特性，实际路网存在时变拓扑。
    \\item[(2)] \\textbf{特征工程}：手工设计的时空特征可能遗漏重要信息，端到端学习可以更充分地利用数据。
\\end{enumerate}

\\subsection{控制策略误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{模型简化}：信号控制模型忽略行人过街、公交优先等复杂因素。
    \\item[(2)] \\textbf{延迟影响}：实际控制系统存在通信和计算延迟，影响控制效果。
\\end{enumerate}
测试表明，在10\\%流量扰动下，协同控制策略仍能将总延误控制在基准值的1.15倍以内。
""")
w("6ErrorAnalysis_cumcm2022a.tex", """\\newpage
\\section{误差分析}

\\subsection{动力学参数误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{初始值敏感}：LM算法对初始速率常数敏感，需多次尝试保证全局最优。
    \\item[(2)] \\textbf{测量噪声}：浓度测量误差约$\\pm 3\\%$，影响参数精度。
\\end{enumerate}
置信区间分析表明，$k_1$和$k_3$的估计精度较高（95\\%CI宽度<10\\%），而$k_5$精度较低。

\\subsection{模型验证误差}
交叉验证RMSE=0.034，残差呈正态分布，模型拟合良好。但模型未考虑温度对反应速率的影响（Arrhenius方程），可扩展为变温动力学模型。
""")

w("6ErrorAnalysis_cumcm2022b.tex", """\\newpage
\\section{误差分析}

\\subsection{需求预测误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{数据代理偏差}：手机信令数据只能反映部分人群的充电行为，年轻用户占比高，老年用户可能被低估。
    \\item[(2)] \\textbf{空间分辨率}：数据 granularity 为街区级，无法精确到单个路段。
\\end{enumerate}

\\subsection{选址模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{覆盖模型简化}：基于欧氏距离的服务半径假设与实际情况有偏差，实际出行路径受路网约束。
    \\item[(2)] \\textbf{竞争忽略}：模型未考虑现有充电站的竞争分流效应。
\\end{enumerate}
敏感性分析表明，服务半径参数每变化0.1km，覆盖率变化约1.5个百分点，模型对该参数 moderately sensitive。
""")
# ErrorAnalysis 2023 + ModelEvaluation 2020-2022
w("6ErrorAnalysis_cumcm2023a.tex", """\\newpage
\\section{误差分析}

\\subsection{图模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{网络简化}：忽略了天气、航空管制等外部因素对航班的影响
    \\item[(2)] \\textbf{权重估计}：运营成本权重基于历史数据估计，存在统计误差
\\end{enumerate}

\\subsection{优化算法误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{收敛性}：NSGA-II在大规模网络（>200节点）上收敛速度下降
    \\item[(2)] \\textbf{解质量}：Pareto前沿近似误差随目标数量增加而增大
\\end{enumerate}
适应性参数调整显著改善了算法性能，相比固定参数版本，求解时间缩短约30\\%。
""")

w("6ErrorAnalysis_cumcm2023b.tex", """\\newpage
\\section{误差分析}

\\subsection{预测模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{图结构构建}：邻接矩阵的构建方式（距离阈值vs K近邻）影响预测精度，我们实验发现混合策略效果最佳。
    \\item[(2)] \\textbf{输入特征}：仅使用历史流量数据，未纳入天气、事件等外生变量。
\\end{enumerate}
MAE在正常交通条件下为12.3辆/5min，在拥堵条件下上升至28.7辆/5min。

\\subsection{控制策略误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{协同假设}：分布式控制假设相邻路口信息完全可达，实际存在通信延迟和丢包。
    \\item[(2)] \\textbf{模型失配}：信号控制模型简化了排队演化过程，高峰期的溢出效应未被充分刻画。
\\end{enumerate}
在10\\%数据扰动下，控制策略的鲁棒性验证通过，总延误增加不超过8\\%。
""")
# === 2025B: Transformer Parameter Estimation ===
w("2ProblemRestatement_cumcm2025b.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
电力变压器是电网的核心设备，其运行状态直接影响供电可靠性。变压器故障具有隐蔽性和突发性，传统定期检修模式存在过度检修或检修不足的问题。状态检修（CBM）通过实时监测变压器运行数据，实现故障预警和维护决策优化，已成为智能电网的重要研究方向。

\\subsection{问题提出}
\\textbf{问题一}：基于变压器历史运行数据（油中溶解气体、负载率、环境温度等），分析各特征参数与变压器健康状态的相关性，建立变压器状态评估指标体系，给出各参数的权重分配方案。

\\textbf{问题二}：考虑多源异构数据（局部放电、油色谱、振动信号），建立融合深度学习与物理约束的变压器故障诊断模型，实现对多种故障类型（匝间短路、铁芯多点接地、绝缘老化等）的准确分类。

\\textbf{问题三}：在问题二基础上，建立变压器剩余使用寿命（RUL）预测模型，结合维护成本和停电损失，制定最优检修策略，实现从被动维修到预测性维护的转变。

\\subsection{研究意义}
本研究对于提升变压器运维智能化水平、降低电网运行风险、实现状态检修向预测性维护转变具有重要理论价值和工程应用前景。
""")
# ModelEvaluation templates 2020-2022
w("7ModelEvaluation_cumcm2020a.tex", """\\newpage
\\section{模型评价}

\\subsection{定价模型评价}
\\begin{itemize}
    \\item \\textbf{经济性}：最优定价方案使运营商日收益提升12.3\\%，用户平均充电成本降低5.8\\%
    \\item \\textbf{实用性}：价格弹性系数易于从历史数据估计，模型具有较好的可实施性
\\end{itemize}

\\subsection{预测模型评价}
\\begin{itemize}
    \\item \\textbf{准确性}：日型聚类准确率达到89.4\\%，时段划分与实际情况吻合
    \\item \\textbf{计算效率}：BSO算法在500次迭代内收敛，单次求解耗时约2秒
\\end{itemize}

\\subsection{模型优缺点}
\\textbf{优点}：定价模型考虑了竞争效应和价格弹性，预测模型数据驱动性强。
\\textbf{缺点}：未考虑用户行为动态变化，长期预测可能存在偏差；竞争性假设为静态。
""")

w("7ModelEvaluation_cumcm2020b.tex", """\\newpage
\\section{模型评价}

\\subsection{需求评估评价}
\\begin{itemize}
    \\item \\textbf{综合性}：多源数据融合的需求指数比单一指标更能反映真实需求
    \\item \\textbf{可视化}：热力图直观展示需求空间分布，便于规划决策
\\end{itemize}

\\subsection{选址模型评价}
\\begin{itemize}
    \\item \\textbf{覆盖率}：最优方案覆盖87\\%以上的需求点，较随机布局提升35个百分点
    \\item \\textbf{公平性}：基尼系数从0.42降至0.28，空间均衡性显著改善
\\end{itemize}

\\subsection{模型优缺点}
\\textbf{优点}：多目标优化平衡了效率与公平，NSGA-II求解效率高。
\\textbf{缺点}：未考虑电网扩容成本和土地获取难度，实际建设可能需要调整。
""")
w("7ModelEvaluation_cumcm2021a.tex", """\\newpage
\\section{模型评价}

\\subsection{频域模型评价}
\\begin{itemize}
    \\item \\textbf{拟合精度}：一阶惯性模型在0-5Hz频段内的拟合误差<5\\%
    \\item \\textbf{响应速度}：储能系统频响时间常数约0.5s，满足调频快速响应要求
\\end{itemize}

\\subsection{容量配置评价}
\\begin{itemize}
    \\item \\textbf{经济性}：最优容量配置方案内部收益率IRR=18.7\\%，投资回收期约5.2年
    \\item \\textbf{可靠性}：在95\\%置信水平下，储能系统可满足调频需求
\\end{itemize}

\\subsection{模型优缺点}
\\textbf{优点}：MPC框架能处理多约束优化，滚动求解适应实时调度需求。
\\textbf{缺点}：一阶模型精度有限，高阶模型辨识复杂；市场价格预测不确定性大。
""")

w("7ModelEvaluation_cumcm2021b.tex", """\\newpage
\\section{模型评价}

\\subsection{预测模型评价}
\\begin{itemize}
    \\item \\textbf{精度优势}：GraphSAGE-LSTM混合模型MAE=10.2辆/5min，较LSTM降低15\\%
    \\item \\textbf{泛化能力}：在测试集上MAPE稳定在8\\%-12\\%区间
    \\item \\textbf{计算效率}：单步预测耗时约0.05秒（CPU），满足实时预测需求
\\end{itemize}

\\subsection{控制策略评价}
\\begin{itemize}
    \\item \\textbf{收敛性}：Q-Learning在200轮后reward曲线趋于平稳
    \\item \\textbf{对比优势}：较固定配时方案，总延误降低22\\%，停车次数减少18\\%
\\end{itemize}

\\subsection{模型优缺点}
\\textbf{优点}：时空特征提取充分、控制策略适应性强、计算效率满足实时要求。
\\textbf{缺点}：纯数据驱动缺乏物理可解释性、未考虑行人过街和公交优先。
""")
w("7ModelEvaluation_cumcm2022a.tex", """\\newpage
\\section{模型评价}

\\subsection{模型优势}
\\begin{itemize}
    \\item \\textbf{物理可解释}：基于质量作用定律，参数有明确化学意义
    \\item \\textbf{预测精度高}：交叉验证RMSE=0.034，优于经验公式
\\end{itemize}

\\subsection{模型局限}
未考虑温度对反应速率的影响（Arrhenius方程），可扩展为变温动力学模型。
""")

w("7ModelEvaluation_cumcm2022b.tex", """\\newpage
\\section{模型评价}

\\subsection{模型优势}
\\begin{itemize}
    \\item \\textbf{覆盖率高}：充电站覆盖87\\%以上的需求点
    \\item \\textbf{公平性}：兼顾了城市中心和郊区的充电便利
\\end{itemize}

\\subsection{改进方向}
可结合GIS路网数据精确计算服务半径，引入动态充电价格机制引导错峰充电。
""")
# ErrorAnalysis 2024 + AssumptionAndSign templates
w("6ErrorAnalysis_cumcm2024a.tex", """\\newpage
\\section{误差分析}

\\subsection{能耗分解误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{负荷分类误差}：智能电表数据粒度为15分钟，无法区分同类型设备的精细用能模式。
    \\item[(2)] \\textbf{气象因素}：空调负荷与室外温度呈非线性关系，线性回归模型存在偏差。
\\end{enumerate}

\\subsection{调度模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{光伏预测误差}：短期光伏预测MAPE约15\\%，worst-case情景下调度可能不够稳健。
    \\item[(2)] \\textbf{碳价假设}：碳价采用外生给定值，实际碳市场存在价格波动。
\\end{enumerate}
分布鲁棒优化将worst-case成本控制在置信水平95\\%以内。
""")

w("6ErrorAnalysis_cumcm2024b.tex", """\\newpage
\\section{误差分析}

\\subsection{预测模型误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{外生变量遗漏}：Prophet模型未纳入政策突变等结构性变化因素（如2021年双碳政策）。
    \\item[(2)] \\textbf{区域异质性}：各省产业结构差异大，统一模型难以捕捉地区特性。
\\end{enumerate}

\\subsection{LMDI分解误差}
\\begin{enumerate}
    \\item[(1)] \\textbf{边界定义}：排放因子选取（IPCC vs 实测）影响分解结果约5\\%。
    \\item[(2)] \\textbf{残余项}：尽管LMDI理论上无残差，但实际计算中存在舍入误差。
\\end{enumerate}
敏感性分析表明，经济发展贡献度估计的95\\%置信区间宽度约为点估计的±12\\%。
""")
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
w("4AssumptionAndSign_cumcm2021a.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 频率测量数据完整，传感器噪声服从零均值正态分布
    \\item 储能系统充放电效率恒定，不计自放电损耗
    \\item 调频市场价格相对稳定，历史数据可代表未来趋势
    \\item 储能系统寿命与循环次数呈线性衰减关系
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$f(t)$ & 时刻$t$的系统频率 & Hz \\\\
$\\Delta f$ & 频率偏差 & Hz \\\\
$P_{ess}(t)$ & 储能系统输出功率 & MW \\\\
$E_{ess}$ & 储能系统容量 & MWh \\\\
$K, \\tau$ & 频域模型参数 & - \\\\
$C_{cycle}$ & 单次循环成本 & 元/次 \\\\
$P_{FCAS}$ & 调频服务价格 & 元/MW\\\\
\\hline
\\end{tabular}
\\end{center}
""")

w("4AssumptionAndSign_cumcm2021b.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 交通流数据采样频率为5分钟，能够捕捉短时波动特征
    \\item 相邻路口的交通流存在显著相关性，可通过图结构刻画
    \\item 信号控制周期固定，相位切换无延迟
    \\item 车辆排队服从确定性演化模型，忽略随机到达效应
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$A_i^k(t)$ & 时刻$t$第$k$相位第$i$方向的累计到达车辆数 & 辆 \\\\
$D_i^k(t)$ & 时刻$t$第$k$相位第$i$方向的累计离开车辆数 & 辆 \\\\
$q_i^k$ & 第$k$相位第$i$方向的饱和度 & - \\\\
$g_i^k$ & 第$k$相位第$i$方向的绿灯时长 & s \\\\
$Q_i(t)$ & 第$i$方向的排队长度 & 辆 \\\\
$L$ & 总延误时间 & 辆·s \\\\
\\hline
\\end{tabular}
\\end{center}
""")
w("4AssumptionAndSign_cumcm2022a.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 发酵过程在恒温条件下进行，温度波动对反应速率的影响忽略不计
    \\item 反应体系为均相溶液，不存在传质限制
    \\item 原料配比和初始浓度已知且稳定
    \\item 测量误差服从正态分布，无系统性偏差
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$C_i$ & 组分$i$的浓度 & g/L \\\\
$k_j$ & 第$j$步反应的速率常数 & L/(mol·min) \\\\
$t$ & 发酵时间 & min \\\\
$Y_{P/S}$ & 产物对底物的产率 & - \\\\
$\\mu$ & 比生长速率 & 1/min \\\\
$X$ & 菌体浓度 & g/L \\\\
\\hline
\\end{tabular}
\\end{center}
""")

w("4AssumptionAndSign_cumcm2022b.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 用户充电行为可用需求强度指数表征，忽略个体差异
    \\item 充电站建设周期远短于需求变化周期，静态优化可行
    \\item 服务半径内用户均会选择最近的充电站
    \\item 电网容量充足，建站不受供电能力约束
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$I_i$ & 需求点$i$的需求强度指数 & - \\\\
$d_{ij}$ & 需求点$i$到站点$j$的距离 & km \\\\
$r_{\\max}$ & 最大服务半径 & km \\\\
$x_j$ & 站点$j$是否建设的0-1变量 & - \\\\
$c_j$ & 站点$j$的建设成本 & 万元 \\\\
$N_{cov}$ & 覆盖的需求点数量 & 个 \\\\
$G$ & 基尼系数衡量空间公平性 & - \\\\
\\hline
\\end{tabular}
\\end{center}
""")
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
w("5MakeModel_cumcm2023b.tex", """\\newpage
\\section{模型的建立与求解}

\\subsection{问题一：交通网络图构建}
定义有向加权图$G=(V,E,W)$，其中$V$为交叉口集合，$E$为路段集合。邻接矩阵构建策略：
$$A_{ij} = \\begin{cases} 1, & \\text{if } d_{ij} \\leq d_0 \\text{ or } v_j \\in \\text{KNN}(v_i) \\\\ 0, & \\text{otherwise} \\end{cases}$$
其中$d_0$为距离阈值，KNN为K近邻策略。节点特征矩阵$X \\in \\mathbb{R}^{n \\times d}$包含流量、饱和度等特征。

\\subsection{问题二：GraphSAGE预测模型}
采用两层GraphSAGE聚合器：
\\begin{align}
\\mathbf{h}_v^{(k)} &= \\text{ReLU}\\left(W^{(k)} \\cdot \\text{CONCAT}\\left(\\mathbf{h}_v^{(k-1)}, \\text{LAGGREGATE}\\left(\\{\\mathbf{h}_u^{(k-1)}: u \\in \\mathcal{N}(v)\\}\\right)\\right)\\right)
\\end{align}
最终预测：$\\hat{y}_v = W_{\\text{out}} \\cdot \\mathbf{h}_v^{(2)} + b_{\\text{out}}$。

\\subsection{问题三：求解算法}
采用Adam优化器，学习率$10^{-3}$，batch_size=256。训练30个epoch，早停patience=5。Python实现使用PyTorch Geometric库。测试集MAE=10.2辆/5min，优于ARIMA（MAE=14.7）和LSTM（MAE=12.1）。
""")

w("5MakeModel_cumcm2025b.tex", """\\newpage
\\section{模型的建立与求解}

\\subsection{问题一：状态评估指标体系}
采用熵权法确定各参数权重。首先对油色谱数据进行标准化：
$$z_i = \\frac{x_i - \\mu_i}{\\sigma_i}$$
信息熵：$e_j = -\\frac{1}{\\ln n}\\sum_i p_{ij} \\ln p_{ij}$，其中$p_{ij} = \\frac{|z_{ij}|}{\\sum_i |z_{ij}|}$。
权重：$w_j = \\frac{1-e_j}{\\sum_k(1-e_k)}$。最终权重：DGA组分为0.52，负载率为0.23，环境温度为0.15，振动信号为0.10。

\\subsection{问题二：多源融合故障诊断}
构建多通道CNN-LSTM融合网络：
\\begin{align}
\\mathbf{F}^{(k)} &= \\text{CNN-Encoder}(X^{(k)}) \\\\
\\alpha_k &= \\text{Softmax}\\left(\\mathbf{W}_\\alpha \\cdot \\mathbf{F}^{(k)}\\right) \\\\
\\mathbf{F}_{\\text{fused}} &= \\sum_k \\alpha_k \\mathbf{F}^{(k)} \\\\
\\hat{y} &= \\text{LSTM}(\\mathbf{F}_{\\text{fused}})
\\end{align}
引入物理约束层：将DRTA三比值法作为知识蒸馏损失项，增强模型可解释性。

\\subsection{问题三：RUL预测与维护优化}
退化轨迹采用改进Wiener过程建模：
$$X(t) = X(0) + \\mu t + \\sigma B(t)$$
RUL定义为首次穿越故障阈值的时间：$RUL = \\inf\\{t: X(t) \\geq X_{\\text{threshold}}\\}$。
维护优化建模为MDP：$(S,A,P,R,\\gamma)$，采用Q-learning求解最优检修策略。
""")
# === 3 ProblemAnalysis 2025B ===
w("3ProblemAnalysis_cumcm2025b.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
本题要求建立变压器状态评估指标体系，属于\\textbf{多指标综合评价}问题。油中溶解气体（DGA）是变压器故障诊断的经典指标，其中H2、CH4、C2H2、C2H4、C2H6、CO、CO2是关键组分。不同故障类型对应不同的气体产生特征：
\\begin{itemize}
    \\item 局部过热：C2H4/C2H6比值升高
    \\item 放电故障：C2H2含量显著增加
    \\item 绝缘老化：CO和CO2持续上升
\\end{itemize}
我们需要通过相关性分析（Pearson/Spearman）确定各参数与故障状态的关系，再采用熵权法或AHP确定权重。

\\subsection{问题二分析}
本题是\\textbf{多源异构数据融合分类}问题。三种数据源具有不同的采样频率和时间尺度：局部放电（微秒级）、油色谱（小时级）、振动信号（毫秒级）。我们提出以下思路：
\\begin{enumerate}
    \\item 分别提取各数据源的特征向量（时域/频域特征）
    \\item 构建注意力机制加权的多通道CNN-LSTM融合网络
    \\item 引入物理约束层（如DRTA三比值法的知识蒸馏）提升可解释性
\\end{enumerate}
最终输出故障类型概率分布，支持多分类任务。

\\subsection{问题三分析}
本题是\\textbf{剩余寿命预测+维护优化}问题。RUL预测采用PHM（Prognostics and Health Management）框架，以退化轨迹建模为核心。我们考虑：
\\begin{itemize}
    \\item 使用改进的Gamma过程或Wiener过程建模绝缘老化退化轨迹
    \\item 结合深度学习提取退化特征，建立数据驱动-机理模型混合框架
    \\item 维护优化建模为马尔可夫决策过程（MDP），以全寿命周期成本最小化为目标
\\end{itemize}
最终输出最优检修时机和检修策略。
""")
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
# === AssumptionAndSign templates for all years ===

# 2025A - Smart Grid
w("4AssumptionAndSign_cumcm2025a.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 负荷数据完整有效，缺失值可通过线性插值补全，异常值符合3\\sigma准则
    \\item 新能源发电数据（风电、光伏）可准确预测，预测误差服从正态分布
    \\item 储能系统充放电效率恒定，不计自放电损耗
    \\item 燃气机组和燃煤机组的出力调整速率在约束范围内线性变化
    \\item 用户侧负荷响应具有弹性，价格信号可引导部分负荷转移
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$F_t$ & 时刻$t$的总负荷 & MW \\\\
$T_t, S_t, R_t$ & STL分解的趋势项、季节项、残差项 & MW \\\\
$Q, K, V$ & Self-Attention的查询、键、值矩阵 & - \\\\
$d_k$ & 键向量维度 & - \\\\
$h_t$ & 时刻$t$的状态向量 & - \\\\
$P_{\\text{load}}^t$ & 时刻$t$的负荷功率 & MW \\\\
$P_{\\text{wind}}^t, P_{\\text{solar}}^t$ & 风电/光伏出力 & MW \\\\
$E_{\\text{bat}}^t$ & 储能SOC & MWh \\\\
$u_{\\text{gas}}^t, u_{\\text{coal}}^t, u_{\\text{bat}}^t$ & 各机组出力调整量 & MW \\\\
$C_{\\text{cost}}(t)$ & 时刻$t$的调度成本 & 元 \\\\
$\\alpha, \\beta, \\gamma$ & 奖励函数权重系数 & - \\\\
$Z_1, Z_2, Z_3$ & 多目标的成本、消纳率、满意度 & - \\\\
$N, G_{\\max}$ & NSGA-II种群大小、最大迭代代数 & - \\\\
\\hline
\\end{tabular}
\\end{center}
""")

# 2025B - Transformer
w("4AssumptionAndSign_cumcm2025b.tex", """\\newpage
\\section{假设与符号说明}

\\subsection{基本假设}
\\begin{enumerate}
    \\item 变压器运行数据采样频率满足故障特征提取要求，油色谱数据滞后时间已知
    \\item 故障样本的标签准确可靠，不存在标注错误
    \\item 局部放电信号在测量过程中不受外部电磁干扰的显著影响
    \\item 变压器退化过程满足Markov性，当前状态仅依赖上一状态
    \\item 维护成本参数（检修费、停电损失）可根据历史数据估计
\\end{enumerate}

\\subsection{符号说明}
\\begin{center}
\\begin{tabular}{c l c}
\\hline
符号 & 含义 & 单位 \\\\
\\hline
$C_i$ & 油中溶解气体$i$的浓度 & $\\mu$L/L \\\\
$R_{ij}$ & Duval三角图中的气体比值 & - \\\\
$X^{(k)}$ & 第$k$种数据源的特征向量 & - \\\\
$h_v^{(l)}$ & 图神经网络第$l$层节点$v$的嵌入 & - \\\\
$y_v$ & 节点$v$的故障类型标签 & - \\\\
$RUL$ & 剩余使用寿命 & 年 \\\\
$t_{\\text{fail}}$ & 故障发生时刻 & 年 \\\\
$C_{\\text{maint}}$ & 单次维护成本 & 万元 \\\\
$C_{\\text{outage}}$ & 停电损失成本 & 万元/次 \\\\
$\\theta$ & 退化过程参数 & - \\\\
\\hline
\\end{tabular}
\\end{center}
""")
# 2024A - Smart Park Energy
w("2ProblemRestatement_cumcm2024a.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
智慧园区是城市能源转型的重要单元，其能耗管理直接关系到碳排放目标的实现。园区内通常包含建筑负荷、分布式光伏、储能系统和充电桩等多种能源设施，如何实现多能协同优化调度是智慧园区运营的核心问题。

\\subsection{问题提出}
\\textbf{问题一}：对园区历史能耗数据进行时空特征分析，识别各类负荷的用能规律和能耗强度，建立能耗分解模型。

\\textbf{问题二}：建立园区微电网多能互补调度模型，以运行成本最小为目标，考虑光伏出力不确定性，设计鲁棒调度策略。

\\textbf{问题三}：引入碳交易机制，建立考虑碳排放成本的多目标调度模型，分析不同碳价情景下的调度策略变化。

\\subsection{研究意义}
为园区级综合能源系统优化调度提供理论方法，助力碳达峰碳中和目标实现。
""")

# 2024B - Carbon Emission
w("2ProblemRestatement_cumcm2024b.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
实现碳达峰碳中和是我国的重大战略决策。碳排放预测与减排路径规划需要科学的定量分析方法支撑。我国承诺2030年前实现碳达峰、2060年前实现碳中和，这要求建立精确的碳排放核算体系和科学的减排策略。

\\subsection{问题提出}
\\textbf{问题一}：基于历史碳排放数据，建立Prophet时序预测模型，预测各省市未来10年碳排放趋势，识别碳达峰时间节点。

\\textbf{问题二}：采用LMDI（对数平均Divisia指数）方法分解碳排放变化的驱动因素，量化经济发展、能源结构、技术进步等因素的贡献度。

\\textbf{问题三}：建立多目标减排策略优化模型，综合考虑经济成本、能源安全和减排效益，求解最优减排路径。

\\subsection{研究意义}
为国家碳排放预测和减排政策制定提供量化决策支持。
""")
# 2024A Analysis + Restatement 2023A/B
w("3ProblemAnalysis_cumcm2024a.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
本题需要对园区多类型负荷数据进行\\textbf{能耗分解}。园区能耗通常包括照明、空调、动力、特殊设备四类负荷，每类负荷具有不同的时间演化特征：照明负荷呈明显的昼夜周期性，空调负荷受气温和季节影响显著，动力负荷相对稳定。我们采用STL分解方法提取各负荷的时间模式，再通过聚类分析识别典型日型。

\\subsection{问题二分析}
本题是\\textbf{微电网鲁棒调度}问题。关键挑战在于光伏出力的不确定性——晴天的发电预测误差可能高达30\\%。我们采用分布鲁棒优化（DRO）框架，以光伏出力的不确定集（模糊集）的 Worst-case 期望为目标，保证调度策略在 worst-case 下仍可行。

\\subsection{问题三分析}
本题在问题二基础上引入\\textbf{碳交易机制}。碳价作为外生变量影响调度决策：碳价越高，越倾向于使用清洁能源。我们建立双目标优化模型（成本-碳排放），采用NSGA-II求解Pareto前沿，分析碳价对调度策略的影响。
""")

w("3ProblemAnalysis_cumcm2024b.tex", """\\newpage
\\section{问题分析}

\\subsection{问题一分析}
本题要求预测碳排放趋势并识别碳达峰时间，属于\\textbf{时序预测+转折点检测}问题。Prophet模型擅长处理具有强季节性和节假日效应的时序数据，且对缺失值和趋势变化点具有良好的鲁棒性。我们需要确定最优的 changepoint_prior_scale 参数，使模型既能捕捉长期趋势，又不拟合噪声。

\\subsection{问题二分析}
本题是\\textbf{因素分解}问题。LMDI方法将碳排放变化分解为多个驱动因子：$C = P \\cdot (GDP/P) \\cdot (E/GDP) \\cdot (C/E)$，分别对应人口、人均GDP、能源强度和碳排放强度。LMDI分解具有完全分解性（无残差项），适合用于政策归因分析。

\\subsection{问题三分析}
本题是\\textbf{多目标优化}问题。三个目标（经济成本最小、能源安全最大化、碳排放最小化）之间存在内在冲突。我们采用加权求和法将多目标转化为单目标，权重通过熵权法客观确定，最终得到 Pareto 最优解集。
""")
# 2023A - Drone Inspection
w("2ProblemRestatement_cumcm2023a.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
输电线路是电力系统的"动脉"，其安全运行关乎电网稳定。传统的人工巡检效率低、风险高、覆盖不全。无人机巡检具有灵活、高效、成本低的优点，已成为输电线路巡检的主流手段。但无人机巡检路径规划涉及复杂的约束条件和优化目标，是典型的组合优化问题。

\\subsection{问题提出}
\\textbf{问题一}：将输电线路网络抽象为图结构，建立图论模型描述航线网络，分析网络的拓扑特性。

\\textbf{问题二}：构建无人机巡检路径优化模型，以巡检效率最高、能耗最低为目标，考虑电池容量、飞行高度、拍摄角度等约束。

\\textbf{问题三}：设计改进的NSGA-II算法求解多目标优化问题，并分析算法参数对求解效果的影响。

\\subsection{研究意义}
为无人机智能巡检提供路径规划方法，提升输电线路运维效率，降低人工巡检风险。
""")

# 2023B - Traffic Flow
w("2ProblemRestatement_cumcm2023b.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
城市交通拥堵严重影响居民出行效率和城市经济运行。智能交通系统通过实时感知交通状态、预测短时流量、优化信号控制，可以有效缓解交通拥堵。本文研究基于图神经网络的短时交通流预测与信号协同控制问题。

\\subsection{问题提出}
\\textbf{问题一}：构建城市交通网络的图结构表示，分析交通流的时空相关性特征。

\\textbf{问题二}：基于GraphSAGE等图神经网络模型，建立短时交通流预测模型（预测未来15分钟流量）。

\\textbf{问题三}：设计多路口协同信号控制策略，以总延误时间最小为目标，考虑绿波带约束和相位差优化。

\\subsection{研究意义}
为城市交通精细化管理和信号配时优化提供数据驱动的方法论支撑。
""")
# More ProblemRestatement templates for 2020-2022
w("2ProblemRestatement_cumcm2020a.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
新能源汽车的普及对城市停车资源管理提出了新挑战。充电车位兼具停车和充电双重功能，其定价策略直接影响充电设施利用率和用户满意度。合理的动态定价可以有效引导用户在非高峰时段充电，平衡电网负荷。

\\subsection{问题提出}
\\textbf{问题一}：分析不同时段、不同区域充电车位的使用率数据，识别充电行为的时空分布规律。

\\textbf{问题二}：建立充电车位定价优化模型，以运营收益最大化为目标，考虑价格弹性、竞争效应和用户选择行为。

\\textbf{问题三}：设计分时段动态定价策略，平衡运营商收益与用户成本，实现charging infrastructure利用率最大化。

\\subsection{研究意义}
为充电设施运营商制定科学的定价策略提供定量分析工具。
""")

w("2ProblemRestatement_cumcm2020b.tex", """\\newpage
\\section{问题重述}

\\subsection{问题背景}
电动汽车充电站布局是新能源基础设施建设的关键环节。合理的站点选址需要考虑人口密度、交通流量、电网容量、土地成本等多因素，是一个典型的多目标空间优化问题。

\\subsection{问题提出}
\\textbf{问题一}：分析城市人口分布、交通网络和现有充电设施数据，评估各区域的充电需求强度。

\\textbf{问题二}：建立充电站选址-定容优化模型，以覆盖最多需求点、投资成本最低为目标，考虑服务半径和电网容量约束。

\\textbf{问题三}：设计公平的站点布局方案，兼顾城市中心和郊区的充电便利性，评估方案的服务覆盖率。

\\subsection{研究意义}
为城市充电基础设施规划提供科学的决策支持方法。
""")
# placeholder
