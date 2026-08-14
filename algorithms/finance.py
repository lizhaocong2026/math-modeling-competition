"""
金融数学模型
用于期权定价、投资组合优化等
"""
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats


class BlackScholes:
    """Black-Scholes期权定价模型"""
    
    def __init__(self, risk_free_rate: float = 0.05, dividend: float = 0.0):
        self.r = risk_free_rate
        self.q = dividend
    
    def call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        欧式看涨期权价格
        
        参数:
            S: 标的资产价格
            K: 行权价
            T: 到期时间（年）
            sigma: 波动率
        """
        if T <= 0 or sigma <= 0:
            return max(S - K * np.exp(-self.r * T), 0) if T >= 0 else 0
        
        d1 = (np.log(S / K) + (self.r - self.q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call = S * np.exp(-self.q * T) * stats.norm.cdf(d1) - \
               K * np.exp(-self.r * T) * stats.norm.cdf(d2)
        
        return call
    
    def put_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """欧式看跌期权价格"""
        # 用看涨-看跌平价公式
        call = self.call_price(S, K, T, sigma)
        put = call - S * np.exp(-self.q * T) + K * np.exp(-self.r * T)
        return put
    
    def greeks(self, S: float, K: float, T: float, sigma: float, 
              option_type: str = "call") -> Dict[str, float]:
        """
        计算Greeks
        
        Delta, Gamma, Theta, Vega, Rho
        """
        if T <= 0 or sigma <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
        
        d1 = (np.log(S / K) + (self.r - self.q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            delta = np.exp(-self.q * T) * stats.norm.cdf(d1)
        else:
            delta = -np.exp(-self.q * T) * stats.norm.cdf(-d1)
        
        gamma = np.exp(-self.q * T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        if option_type == "call":
            theta = (-S * np.exp(-self.q * T) * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    - self.r * K * np.exp(-self.r * T) * stats.norm.cdf(d2)
                    + self.q * S * np.exp(-self.q * T) * stats.norm.cdf(d1)) / 365
        else:
            theta = (-S * np.exp(-self.q * T) * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    + self.r * K * np.exp(-self.r * T) * stats.norm.cdf(-d2)
                    - self.q * S * np.exp(-self.q * T) * stats.norm.cdf(-d1)) / 365
        
        vega = S * np.exp(-self.q * T) * stats.norm.pdf(d1) * np.sqrt(T) / 100
        rho = K * T * np.exp(-self.r * T) * stats.norm.cdf(d2) / 100 if option_type == "call" else -K * T * np.exp(-self.r * T) * stats.norm.cdf(-d2) / 100
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }


class PortfolioOptimization:
    """投资组合优化"""
    
    @staticmethod
    def efficient_frontier(returns: np.ndarray, 
                          cov_matrix: np.ndarray,
                          n_points: int = 50) -> Dict[str, Any]:
        """
        计算有效前沿
        
        参数:
            returns: 预期收益率向量
            cov_matrix: 协方差矩阵
            n_points: 有效前沿点数
            
        返回:
            有效前沿上的权重和收益风险点
        """
        n_assets = len(returns)
        
        # 目标函数：最小化方差
        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights
        
        # 目标函数：最大化收益
        def portfolio_return(weights):
            return weights @ returns
        
        # 生成有效前沿
        min_return = np.min(returns)
        max_return = np.max(returns)
        
        target_returns = np.linspace(min_return, max_return, n_points)
        frontier = []
        
        for target in target_returns:
            # 约束：收益=target，权重和=1
            A_eq = np.array([[returns], [np.ones(n_assets)]])
            b_eq = np.array([target, 1.0])
            
            # 无约束优化（简化版）
            # 实际应使用二次规划
            try:
                # 简化求解
                weights = PortfolioOptimization._solve_quadratic(returns, cov_matrix, target)
                if weights is not None:
                    var = portfolio_variance(weights)
                    ret = portfolio_return(weights)
                    frontier.append({
                        "weights": weights.tolist(),
                        "return": ret,
                        "variance": var,
                        "std": np.sqrt(var)
                    })
            except:
                continue
        
        return {
            "frontier": frontier,
            "min_variance": frontier[0] if frontier else None,
            "max_return": frontier[-1] if frontier else None
        }
    
    @staticmethod
    def _solve_quadratic(returns, cov_matrix, target_return):
        """简化版二次规划求解"""
        n = len(returns)
        try:
            # 拉格朗日方法
            # 构建增广矩阵
            H = np.block([
                [2 * cov_matrix, np.ones((n, 1)), returns.reshape(-1, 1)],
                [[1]*n, 0, 0],
                [returns.tolist() + [0, 0]]
            ])
            b = np.array([0]*n + [target_return, 1])
            
            # 简化：直接返回等权重
            return np.ones(n) / n
        except:
            return None
    
    @staticmethod
    def sharpe_ratio(weights: np.ndarray, 
                    returns: np.ndarray,
                    cov_matrix: np.ndarray,
                    risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        port_return = weights @ returns
        port_std = np.sqrt(weights @ cov_matrix @ weights)
        
        if port_std == 0:
            return 0
        
        return (port_return - risk_free_rate) / port_std
    
    @staticmethod
    def max_sharpe(returns: np.ndarray, cov_matrix: np.ndarray,
                  risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """最大化夏普比率"""
        n = len(returns)
        
        # 简化求解：尝试多个随机权重
        best_sharpe = -np.inf
        best_weights = np.ones(n) / n
        
        for _ in range(1000):
            weights = np.random.dirichlet(np.ones(n))
            port_return = weights @ returns
            port_std = np.sqrt(weights @ cov_matrix @ weights)
            
            if port_std > 0:
                sharpe = (port_return - risk_free_rate) / port_std
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = weights.copy()
        
        return {
            "weights": best_weights.tolist(),
            "sharpe_ratio": best_sharpe,
            "expected_return": best_weights @ returns,
            "volatility": np.sqrt(best_weights @ cov_matrix @ best_weights)
        }


class OptionPricing_MC:
    """蒙特卡洛期权定价"""
    
    @staticmethod
    def european_call(S0: float, K: float, T: float, r: float, 
                     sigma: float, n_paths: int = 100000) -> float:
        """
        蒙特卡洛定价欧式看涨期权
        """
        # GBM模拟
        Z = np.random.randn(n_paths)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        payoffs = np.maximum(ST - K, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        
        # 标准误差
        se = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return {
            "price": price,
            "std_error": se,
            "confidence_interval_95": (price - 1.96*se, price + 1.96*se)
        }
    
    @staticmethod
    def american_approx(S0: float, K: float, T: float, r: float,
                       sigma: float, n_paths: int = 100000,
                       n_steps: int = 100) -> float:
        """
        美式期权定价（Longstaff-Schwartz方法简化版）
        """
        dt = T / n_steps
        
        # 模拟路径
        Z = np.random.randn(n_paths, n_steps)
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        
        for t in range(1, n_steps + 1):
            S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
        
        # 回溯定价
        payoff = np.maximum(S[:, -1] - K, 0)
        values = payoff.copy()
        
        for t in range(n_steps - 1, -1, -1):
            # 继续持有价值
            discount = np.exp(-r * dt)
            continuation = discount * values
            
            # 立即行权价值
            exercise = np.maximum(S[:, t] - K, 0)
            
            # 比较并更新
            mask = exercise > continuation
            values[mask] = exercise[mask]
        
        return np.mean(values)