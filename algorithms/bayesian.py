"""
贝叶斯统计推断
用于参数估计和不确定性量化
"""
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from scipy import stats


class BayesianInference:
    """贝叶斯推断类"""
    
    def __init__(self, prior: Callable, likelihood: Callable):
        """
        初始化
        
        参数:
            prior: 先验分布函数 p(theta)
            likelihood: 似然函数 p(data|theta)
        """
        self.prior = prior
        self.likelihood = likelihood
        self.posterior_samples = None
        
    def posterior(self, data: np.ndarray, n_samples: int = 10000) -> np.ndarray:
        """
        计算后验分布（简化版，使用MCMC）
        
        参数:
            data: 观测数据
            n_samples: 采样数
            
        返回:
            后验样本
        """
        # 使用Metropolis-Hastings采样
        samples = []
        
        # 初始值
        theta_current = np.mean(data) if len(data) > 0 else 0
        log_prior_current = self.prior(theta_current)
        log_likelihood_current = self.likelihood(data, theta_current)
        
        for i in range(n_samples):
            # Proposal
            theta_proposed = theta_current + np.random.randn() * 0.1
            
            log_prior_proposed = self.prior(theta_proposed)
            log_likelihood_proposed = self.likelihood(data, theta_proposed)
            
            # Acceptance probability
            log_alpha = (log_likelihood_proposed + log_prior_proposed) - \
                        (log_likelihood_current + log_prior_current)
            
            if np.log(np.random.random()) < log_alpha:
                theta_current = theta_proposed
                log_likelihood_current = log_likelihood_proposed
                log_prior_current = log_prior_proposed
            
            samples.append(theta_current)
        
        self.posterior_samples = np.array(samples)
        return self.posterior_samples
    
    def summary(self) -> Dict[str, Any]:
        """后验分布摘要"""
        if self.posterior_samples is None:
            return {}
        
        return {
            "mean": float(np.mean(self.posterior_samples)),
            "std": float(np.std(self.posterior_samples)),
            "median": float(np.median(self.posterior_samples)),
            "ci_95": (float(np.percentile(self.posterior_samples, 2.5)),
                     float(np.percentile(self.posterior_samples, 97.5))),
            "n_samples": len(self.posterior_samples)
        }


class ConjugateBayes:
    """共轭先验贝叶斯推断"""
    
    @staticmethod
    def normal_normal(data: np.ndarray, 
                      mu_prior_mean: float = 0,
                      mu_prior_var: float = 1,
                      sigma_known: float = 1) -> Dict[str, Any]:
        """
        正态-正态共轭模型
        
        参数:
            data: 观测数据
            mu_prior_mean: 先验均值
            mu_prior_var: 先验方差
            sigma_known: 已知标准差
            
        返回:
            后验分布参数
        """
        n = len(data)
        data_mean = np.mean(data)
        
        # 后验参数
        precision_prior = 1 / mu_prior_var
        precision_data = n / (sigma_known ** 2)
        
        mu_post_mean = (precision_prior * mu_prior_mean + precision_data * data_mean) / \
                       (precision_prior + precision_data)
        mu_post_var = 1 / (precision_prior + precision_data)
        
        return {
            "posterior_mean": mu_post_mean,
            "posterior_var": mu_post_var,
            "posterior_std": np.sqrt(mu_post_var),
            "credible_interval_95": (mu_post_mean - 1.96 * np.sqrt(mu_post_var),
                                      mu_post_mean + 1.96 * np.sqrt(mu_post_var))
        }
    
    @staticmethod
    def beta_binomial(successes: int, trials: int,
                      alpha_prior: float = 1, beta_prior: float = 1) -> Dict[str, Any]:
        """
        Beta-二项式共轭模型
        
        参数:
            successes: 成功次数
            trials: 试验次数
            alpha_prior: Beta先验alpha
            beta_prior: Beta先验beta
            
        返回:
            后验分布参数
        """
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + trials - successes
        
        mean = alpha_post / (alpha_post + beta_post)
        var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
        
        return {
            "posterior_alpha": alpha_post,
            "posterior_beta": beta_post,
            "mean": mean,
            "variance": var,
            "std": np.sqrt(var)
        }
    
    @staticmethod
    def poisson_gamma(data: np.ndarray,
                      alpha_prior: float = 1,
                      beta_prior: float = 1) -> Dict[str, Any]:
        """
        Gamma-泊松共轭模型
        
        参数:
            data: 计数数据
            alpha_prior: Gamma先验alpha
            beta_prior: Gamma先验beta
            
        返回:
            后验分布参数
        """
        n = len(data)
        sum_data = np.sum(data)
        
        alpha_post = alpha_prior + sum_data
        beta_post = beta_prior + n
        
        mean = alpha_post / beta_post
        var = alpha_post / (beta_post**2)
        
        return {
            "posterior_alpha": alpha_post,
            "posterior_beta": beta_post,
            "mean": mean,
            "variance": var,
            "std": np.sqrt(var)
        }


class BayesFactor:
    """贝叶斯因子计算"""
    
    @staticmethod
    def compute(model1_likelihood: float, model2_likelihood: float,
                prior_model1: float = 0.5, prior_model2: float = 0.5) -> Dict[str, Any]:
        """
        计算贝叶斯因子
        
        参数:
            model1_likelihood: 模型1的边际似然
            model2_likelihood: 模型2的边际似然
            prior_model1: 模型1的先验概率
            prior_model2: 模型2的先验概率
            
        返回:
            贝叶斯因子和后验概率
        """
        bf = model1_likelihood / model2_likelihood if model2_likelihood > 0 else float('inf')
        
        # 后验概率
        post_model1 = (prior_model1 * model1_likelihood) / \
                      (prior_model1 * model1_likelihood + prior_model2 * model2_likelihood)
        
        return {
            "bayes_factor": bf,
            "posterior_model1": post_model1,
            "posterior_model2": 1 - post_model1,
            "interpretation": BayesFactor._interpret_bf(bf)
        }
    
    @staticmethod
    def _interpret_bf(bf: float) -> str:
        """解释贝叶斯因子"""
        if bf < 1/3:
            return "支持模型2"
        elif bf < 1:
            return " anecdotal evidence for model2"
        elif bf <= 3:
            return "anecdotal evidence for model1"
        elif bf <= 10:
            return " moderate evidence for model1"
        else:
            return " strong evidence for model1"