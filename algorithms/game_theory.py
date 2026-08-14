"""
博弈论算法
用于竞争决策分析问题
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.optimize import linprog


class GameTheory:
    """博弈论求解器"""
    
    @staticmethod
    def find_nash_equilibrium(payoff_matrix_a: np.ndarray, 
                             payoff_matrix_b: np.ndarray = None) -> Dict[str, Any]:
        """
        求解纳什均衡
        
        参数:
            payoff_matrix_a: 玩家A的收益矩阵
            payoff_matrix_b: 玩家B的收益矩阵（零和博弈可为None）
            
        返回:
            纳什均衡策略
        """
        n = payoff_matrix_a.shape[0]
        m = payoff_matrix_a.shape[1]
        
        if payoff_matrix_b is None:
            # 零和博弈
            # 使用线性规划求解
            # max v s.t. A^T p >= v
            c = np.zeros(n)
            c[-1] = -1  # 最大化v
            
            # 约束
            A_ub = -payoff_matrix_a.T
            b_ub = np.zeros(n)
            
            bounds = [(0, 1)] * n
            bounds.append((None, None))  # v无界
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                return {
                    "player_a_strategy": result.x[:n].tolist(),
                    "value": -result.x[-1],
                    "method": "linear_programming"
                }
        
        # 一般双人博弈的纳什均衡求解
        # 使用补变换方法
        return GameTheory._find_nash_general(payoff_matrix_a, payoff_matrix_b)
    
    @staticmethod
    def _find_nash_general(A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
        """一般双人博弈纳什均衡"""
        n, m = A.shape
        
        # 简化版：枚举纯策略纳什均衡
        pure_nash = []
        for i in range(n):
            for j in range(m):
                # 检查是否是纳什均衡
                is_nash = True
                # 玩家A不会单方面偏离
                for i2 in range(n):
                    if B[i2, j] > B[i, j]:
                        is_nash = False
                        break
                # 玩家B不会单方面偏离
                for j2 in range(m):
                    if A[i, j2] > A[i, j]:
                        is_nash = False
                        break
                if is_nash:
                    pure_nash.append((i, j))
        
        # 混合策略纳什均衡（简化版）
        # 对于2x2博弈
        if n == 2 and m == 2:
            mix_nash = GameTheory._solve_2x2(A, B)
            if mix_nash:
                return {"pure": pure_nash, "mixed": mix_nash}
        
        return {"pure": pure_nash, "mixed": None}
    
    @staticmethod
    def _solve_2x2(A: np.ndarray, B: np.ndarray) -> Optional[Dict[str, Any]]:
        """求解2x2博弈的混合策略均衡"""
        # 玩家1的混合策略
        # p*A[j1] + (1-p)*A[j2] = p*B[j1] + (1-p)*B[j2]
        # 对于玩家1，选择p使得玩家2在j1和j2之间无差异
        # 对于玩家2，选择q使得玩家1在i1和i2之间无差异
        
        try:
            # 玩家1以概率q选择行1
            # 玩家2的期望收益相等
            # q*A[0,0] + (1-q)*A[1,0] = q*A[0,1] + (1-q)*A[1,1]
            q = (A[1,0] - A[1,1]) / (A[0,0] - A[0,1] - A[1,0] + A[1,1] + 1e-10)
            q = max(0, min(1, q))
            
            # 玩家2以概率p选择列1
            p = (B[1,0] - B[1,1]) / (B[0,0] - B[0,1] - B[1,0] + B[1,1] + 1e-10)
            p = max(0, min(1, p))
            
            if 0 <= q <= 1 and 0 <= p <= 1:
                return {
                    "player1": [p, 1-p],
                    "player2": [q, 1-q]
                }
        except:
            pass
        
        return None
    
    @staticmethod
    def shapley_value(values: Dict[Tuple[int, ...], float], 
                      n_players: int) -> np.ndarray:
        """
        Shapley值计算
        
        参数:
            values: 联盟值函数 {tuple: value}
            n_players: 玩家数量
            
        返回:
            每个玩家的Shapley值
        """
        n = n_players
        shapley = np.zeros(n)
        
        # 枚举所有排列
        from itertools import permutations
        for perm in permutations(range(n)):
            # position tracking
            # 简化计算
            pass
        
        # 简化版：使用公式计算
        for i in range(n):
            total = 0
            for k in range(n):
                # 选择k个其他玩家
                from math import comb
                for S in itertools.combinations([j for j in range(n) if j != i], k):
                    S_set = set(S)
                    S_i = S_set | {i}
                    
                    # 边际贡献
                    v_S = values.get(tuple(sorted(S_set)), 0)
                    v_S_i = values.get(tuple(sorted(S_i)), 0)
                    
                    total += (v_S_i - v_S) * comb(n-1, k)
            
            shapley[i] = total / (n * comb(n-1, n-1))
        
        # 归一化
        if np.sum(shapley) > 0:
            shapley = shapley / np.sum(shapley)
        
        return shapley


class AuctionTheory:
    """拍卖理论"""
    
    @staticmethod
    def first_price_auction(n_bidders: int, valuations: np.ndarray) -> Dict[str, Any]:
        """
        第一价格密封拍卖
        贝叶斯纳什均衡：bid = (n-1)/n * valuation
        """
        bids = (n_bidders - 1) / n_bidders * valuations
        winner = np.argmax(bids)
        
        return {
            "bids": bids.tolist(),
            "winner": winner,
            "winning_bid": bids[winner],
            "seller_revenue": bids[winner]
        }
    
    @staticmethod
    def second_price_auction(n_bidders: int, valuations: np.ndarray) -> Dict[str, Any]:
        """
        第二价格密封拍卖（Vickrey拍卖）
        优势策略：bid = valuation
        """
        bids = valuations.copy()
        sorted_bids = np.sort(bids)[::-1]
        winner = np.argmax(bids)
        
        return {
            "bids": bids.tolist(),
            "winner": winner,
            "winning_bid": sorted_bids[0],
            "payment": sorted_bids[1] if len(sorted_bids) > 1 else 0,
            "seller_revenue": sorted_bids[1] if len(sorted_bids) > 1 else 0
        }


class CooperativeGame:
    """合作博弈"""
    
    @staticmethod
    def core_check(values: Dict[Tuple[int, ...], float], 
                   allocation: np.ndarray) -> Dict[str, Any]:
        """
        核心存在性检验
        核心：所有联盟都不会有动机偏离
        """
        n = len(allocation)
        core_conditions = []
        
        from itertools import combinations
        
        for k in range(1, n):
            for S in combinations(range(n), k):
                S = tuple(sorted(S))
                coalition_value = values.get(S, 0)
                allocation_S = np.sum(allocation[list(S)])
                
                if allocation_S < coalition_value:
                    core_conditions.append({
                        "coalition": S,
                        "value": coalition_value,
                        "allocation": allocation_S,
                        "violated": True
                    })
        
        is_in_core = len(core_conditions) == 0
        
        return {
            "allocation": allocation.tolist(),
            "is_in_core": is_in_core,
            "violations": core_conditions
        }
    
    @staticmethod
    def nucleolus(values: Dict[Tuple[int, ...], float], 
                  n_players: int) -> np.ndarray:
        """
        Nucleolus求解（简化版）
        """
        # 简化：使用Shapley值作为近似
        return GameTheory.shapley_value(values, n_players)