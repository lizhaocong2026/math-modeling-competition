"""
VIKOR 多准则决策方法
用于解决冲突性多准则决策问题，寻找妥协解
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class VIKOR:
    """
    VIKOR多准则决策方法
    
    VIKOR通过计算正负理想解的距离，为决策者提供妥协排序。
    核心思想：最大化群体效用，最小化个体遗憾。
    
    公式：
        S_i = sum_j [w_j * (f_j* - f_ij) / (f_j* - f_j-)]
        R_i = max_j [w_j * (f_j* - f_ij) / (f_j* - f_j-)]
        Q_i = v * (S_i - S*) / (S* - S-) + (1-v) * (R_i - R*) / (R* - R-)
    
    其中：
        f_j*: 第j个准则的正理想解
        f_j-: 第j个准则的负理想解
        v: 决策者权重（通常取0.5）
    """
    
    def __init__(self, v: float = 0.5, method: str = 'max'):
        """
        初始化VIKOR
        
        参数:
            v: 策略权重 (0 <= v <= 1)，v=0.5表示最大群体效用和最小个体遗憾的平衡
            method: 归一化方法 ('max', 'euclidean')
        """
        self.v = v
        self.method = method
        self.results = None
    
    def solve(self, decision_matrix: np.ndarray, 
              weights: np.ndarray, 
              criteria_type: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        求解VIKOR
        
        参数:
            decision_matrix: 决策矩阵 (m个方案 x n个准则)
            weights: 准则权重向量 (长度为n)
            criteria_type: 准则类型列表 ('benefit'或'cost')
        
        返回:
            包含排序结果的字典
        """
        m, n = decision_matrix.shape
        
        # 确定正负理想解
        if criteria_type is None:
            criteria_type = ['benefit'] * n
        
        f_star = np.zeros(n)
        f_minus = np.zeros(n)
        
        for j in range(n):
            if criteria_type[j] == 'benefit':
                f_star[j] = decision_matrix[:, j].max()
                f_minus[j] = decision_matrix[:, j].min()
            else:  # cost
                f_star[j] = decision_matrix[:, j].min()
                f_minus[j] = decision_matrix[:, j].max()
        
        # 归一化
        if self.method == 'max':
            denom = np.abs(f_star - f_minus)
            denom[denom == 0] = 1
        else:
            denom = np.sqrt(np.sum((decision_matrix - f_minus) ** 2, axis=0))
            denom[denom == 0] = 1
        
        # 计算S和R
        S = np.zeros(m)
        R = np.zeros(m)
        
        for i in range(m):
            for j in range(n):
                ratio = (f_star[j] - decision_matrix[i, j]) / denom[j]
                S[i] += weights[j] * ratio
                R[i] = max(R[i], weights[j] * ratio)
        
        # 计算Q值
        S_star = S.min()
        S_minus = S.max()
        R_star = R.min()
        R_minus = R.max()
        
        denom_S = S_minus - S_star
        denom_R = R_minus - R_star
        
        if denom_S == 0:
            Q = (1 - self.v) * (R - R_star) / (denom_R + 1e-10)
        elif denom_R == 0:
            Q = self.v * (S - S_star) / (denom_S + 1e-10)
        else:
            Q = self.v * (S - S_star) / denom_S + (1 - self.v) * (R - R_star) / denom_R
        
        # 排序
        S_rank = np.argsort(S)
        R_rank = np.argsort(R)
        Q_rank = np.argsort(Q)
        
        # 检查稳定条件
        C1_accepted = False
        C2_accepted = False
        
        if len(Q_rank) >= 2:
            delta_Q = Q[Q_rank[1]] - Q[Q_rank[0]]
            if delta_Q >= 1e-10:
                C1_accepted = (delta_Q >= 1 / (m - 1))
            
            delta_R = R[Q_rank[1]] - R[Q_rank[0]]
            C2_accepted = (Q[Q_rank[0]] - Q[Q_rank[1]]) / (Q[Q_rank[-1]] - Q[Q_rank[0]] + 1e-10) <= 0.5
        
        stable_solution = C1_accepted and C2_accepted
        
        self.results = {
            'S': S,
            'R': R,
            'Q': Q,
            'S_rank': S_rank,
            'R_rank': R_rank,
            'Q_rank': Q_rank,
            'f_star': f_star,
            'f_minus': f_minus,
            'C1_accepted': C1_accepted,
            'C2_accepted': C2_accepted,
            'stable_solution': stable_solution,
            'best_solution': Q_rank[0]
        }
        
        return self.results
    
    def get_recommendation(self) -> str:
        """获取推荐方案"""
        if self.results is None:
            return "请先调用solve()方法"
        
        best = self.results['best_solution']
        if self.results['stable_solution']:
            return f"推荐方案: 方案{best + 1} (稳定解)"
        else:
            return f"无稳定解，推荐方案1: 方案{best + 1}, 方案2: 方案{self.results['Q_rank'][1] + 1}"


class PROMETHEE:
    """
    PROMETHEE多准则决策方法
    
    PROMETHEE I和II是最常用的偏好rank顺序方法。
    
    核心概念：
        - 偏好函数：量化两个方案间的偏好程度
        - 净流：综合所有准则的偏好强度
    """
    
    def __init__(self, preference_type: str = 'linear', preference_params: Optional[Dict] = None):
        """
        初始化PROMETHEE
        
        参数:
            preference_type: 偏好函数类型
                - 'usual': 通常偏好
                - 'u-shaped': U形偏好
                - 'level': 水平偏好
                - 'linear': 线性偏好
                - 'V-shaped': V形偏好
                - ' gaussian': 高斯偏好
            preference_params: 偏好函数参数
        """
        self.preference_type = preference_type
        self.preference_params = preference_params or {}
        self.results = None
    
    def _preference_function(self, d: float, q: float = 0, p: float = None, 
                              sigma: float = None) -> float:
        """
        偏好函数
        
        参数:
            d: 差异值
            q: 无差异阈值
            p: 严格偏好阈值
            sigma: 高斯分布参数
        """
        if p is None:
            p = q * 2
        
        if self.preference_type == 'usual':
            return 1 if d > 0 else 0
        
        elif self.preference_type == 'u-shaped':
            return 1 if d > p else 0
        
        elif self.preference_type == 'level':
            if d <= q:
                return 0
            elif d <= p:
                return 0.5
            else:
                return 1
        
        elif self.preference_type == 'linear':
            if d <= q:
                return 0
            elif d <= p:
                return (d - q) / (p - q)
            else:
                return 1
        
        elif self.preference_type == 'V-shaped':
            return 0 if d <= q else min(1, d / p)
        
        elif self.preference_type == 'gaussian':
            if sigma is None:
                sigma = p
            return 1 - np.exp(-(d ** 2) / (2 * sigma ** 2))
        
        return 0
    
    def solve(self, decision_matrix: np.ndarray, 
              weights: np.ndarray,
              criteria_type: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        求解PROMETHEE II
        
        参数:
            decision_matrix: 决策矩阵
            weights: 准则权重
            criteria_type: 准则类型
        """
        m, n = decision_matrix.shape
        
        if criteria_type is None:
            criteria_type = ['benefit'] * n
        
        # 计算成对偏好指数
        PI = np.zeros((m, m))
        
        for i in range(m):
            for k in range(m):
                if i == k:
                    continue
                
                pi_ik = 0
                for j in range(n):
                    d = decision_matrix[k, j] - decision_matrix[i, j]
                    if criteria_type[j] == 'cost':
                        d = -d
                    
                    p = weights[j] * self._preference_function(d)
                    pi_ik += p
                
                PI[i, k] = pi_ik / np.sum(weights)
        
        # 计算净流
        phi_plus = np.sum(PI, axis=1)  # 正流
        phi_minus = np.sum(PI, axis=0)  # 负流
        phi_net = phi_plus - phi_minus  # 净流
        
        # 排序
        ranking = np.argsort(-phi_net)
        
        self.results = {
            'preference_index': PI,
            'positive_flow': phi_plus,
            'negative_flow': phi_minus,
            'net_flow': phi_net,
            'ranking': ranking,
            'best_alternative': ranking[0]
        }
        
        return self.results
    
    def get_recommendation(self) -> str:
        if self.results is None:
            return "请先调用solve()方法"
        
        best = self.results['best_alternative']
        return f"推荐方案: 方案{best + 1} (净流 = {self.results['net_flow'][best]:.4f})"


class ELECTRE:
    """
    ELECTRE多准则决策方法
    
    ELECTRE I用于产生 outranking 关系，ELECTRE III/IV用于排序。
    这里实现ELECTRE I。
    """
    
    def __init__(self, concordance_threshold: float = 0.65, 
                 discordance_threshold: float = 0.8):
        """
        初始化ELECTRE
        
        参数:
            concordance_threshold: 一致阈值
            discordance_threshold: 不一致阈值
        """
        self.concordance_threshold = concordance_threshold
        self.discordance_threshold = discordance_threshold
        self.results = None
    
    def solve(self, decision_matrix: np.ndarray, 
              weights: np.ndarray) -> Dict[str, Any]:
        """
        求解ELECTRE I
        
        参数:
            decision_matrix: 决策矩阵
            weights: 准则权重
        """
        m, n = decision_matrix.shape
        
        # 归一化
        norm_matrix = decision_matrix / np.sqrt(np.sum(decision_matrix ** 2, axis=0))
        
        # 计算加权归一化矩阵
        weighted_matrix = norm_matrix * weights
        
        # 计算一致和不一致集合
        concordance = np.zeros((m, m))
        discordance = np.zeros((m, m))
        
        for i in range(m):
            for k in range(m):
                if i == k:
                    continue
                
                # 一致集合
                c_indices = []
                for j in range(n):
                    if weighted_matrix[i, j] >= weighted_matrix[k, j]:
                        c_indices.append(j)
                
                concordance[i, k] = np.sum(weights[c_indices]) if c_indices else 0
                
                # 不一致集合
                d_values = []
                for j in range(n):
                    if weighted_matrix[k, j] - weighted_matrix[i, j] > 0:
                        d_values.append((weighted_matrix[k, j] - weighted_matrix[i, j]) / 
                                       (np.max(decision_matrix[:, j]) - np.min(decision_matrix[:, j]) + 1e-10))
                
                discordance[i, k] = max(d_values) if d_values else 0
        
        # 确定 outranking 关系
        outranking = np.zeros((m, m))
        for i in range(m):
            for k in range(m):
                if i != k:
                    if concordance[i, k] >= self.concordance_threshold and \
                       discordance[i, k] <= self.discordance_threshold:
                        outranking[i, k] = 1
        
        # 计算净流出
        net_flow = np.sum(outranking, axis=1) - np.sum(outranking, axis=0)
        ranking = np.argsort(-net_flow)
        
        self.results = {
            'concordance': concordance,
            'discordance': discordance,
            'outranking': outranking,
            'net_flow': net_flow,
            'ranking': ranking,
            'best_alternative': ranking[0]
        }
        
        return self.results


if __name__ == "__main__":
    np.random.seed(42)
    
    # 决策矩阵 (5个方案 x 4个准则)
    matrix = np.array([
        [85, 90, 75, 88],
        [90, 80, 85, 82],
        [78, 95, 80, 75],
        [88, 85, 90, 80],
        [82, 88, 85, 90]
    ])
    
    weights = np.array([0.3, 0.25, 0.25, 0.2])
    criteria_type = ['benefit', 'benefit', 'benefit', 'benefit']
    
    # VIKOR
    print("=" * 50)
    print("VIKOR Results:")
    vikor = VIKOR(v=0.5)
    vikor_results = vikor.solve(matrix, weights, criteria_type)
    print(vikor.get_recommendation())
    print(f"S values: {vikor_results['S']}")
    print(f"R values: {vikor_results['R']}")
    print(f"Q values: {vikor_results['Q']}")
    
    # PROMETHEE
    print("\n" + "=" * 50)
    print("PROMETHEE Results:")
    promethee = PROMETHEE(preference_type='linear')
    prom_results = promethee.solve(matrix, weights, criteria_type)
    print(promethee.get_recommendation())
    print(f"Net flow: {prom_results['net_flow']}")
    print(f"Ranking: {prom_results['ranking'] + 1}")
    
    # ELECTRE
    print("\n" + "=" * 50)
    print("ELECTRE Results:")
    electre = ELECTRE(concordance_threshold=0.65, discordance_threshold=0.8)
    el_results = electre.solve(matrix, weights)
    print(electre.get_recommendation() if hasattr(electre, 'get_recommendation') else "Done")
    print(f"Net flow: {el_results['net_flow']}")
    print(f"Ranking: {el_results['ranking'] + 1}")
