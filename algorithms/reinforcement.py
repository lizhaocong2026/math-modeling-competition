"""
Reinforcement Learning for optimization and scheduling problems
Q-Learning and SARSA implementations for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List, Tuple
import collections


class QLearningAgent:
    """
    Q-Learning agent for discrete optimization problems
    
    Suitable for: A题优化调度、路径规划、资源分配
    """
    
    def __init__(self, state_dim=3, action_dim=2, 
                 lr=0.1, gamma=0.95, epsilon=0.1, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = collections.defaultdict(lambda: np.zeros(action_dim))
        self.episode_rewards = []
        
    def _state_to_key(self, state):
        return tuple(np.round(state, decimals=2))
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_key = self._state_to_key(state)
        return int(np.argmax(self.q_table[state_key]))
    
    def update(self, state, action, reward, next_state, done):
        state_key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_key])
        current = self.q_table[state_key][action]
        new_val = current + self.lr * (target - current)
        self.q_table[state_key][action] = new_val
        return abs(target - current)
    
    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
    
    def fit(self, env, episodes=1000):
        self.episode_rewards = []
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            while True:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            self.decay_epsilon()
            self.episode_rewards.append(total_reward)
        return {
            "status": "success",
            "final_epsilon": self.epsilon,
            "avg_reward_last100": float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 100 else float(np.mean(self.episode_rewards)),
            "q_table_size": len(self.q_table),
        }
    
    def get_best_policy(self):
        policy = {}
        for state_key, q_values in self.q_table.items():
            policy[state_key] = int(np.argmax(q_values))
        return policy
    
    def get_params(self):
        return {"state_dim": self.state_dim, "action_dim": self.action_dim,
                "lr": self.lr, "gamma": self.gamma, "epsilon": self.epsilon}


class SARSAgent:
    """SARSA agent (on-policy version)"""
    
    def __init__(self, state_dim=3, action_dim=2, lr=0.1, gamma=0.95, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = collections.defaultdict(lambda: np.zeros(action_dim))
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_key = tuple(np.round(state, decimals=2))
        return int(np.argmax(self.q_table[state_key]))
    
    def fit(self, env, episodes=500):
        for ep in range(episodes):
            state = env.reset()
            action = self.choose_action(state)
            total_reward = 0
            while True:
                next_state, reward, done, _ = env.step(action)
                next_action = self.choose_action(next_state)
                state_key = tuple(np.round(state, decimals=2))
                next_key = tuple(np.round(next_state, decimals=2))
                target = reward + self.gamma * self.q_table[next_key][next_action]
                self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])
                state, action = next_state, next_action
                total_reward += reward
                if done:
                    break
        return {"status": "success", "episodes": episodes}


class SimpleEnv:
    """Simple environment for testing RL agents"""
    
    def __init__(self, n_resources=5, n_actions=3):
        self.n_resources = n_resources
        self.n_actions = n_actions
        self.state_dim = n_resources
        self.action_dim = n_actions
        
    def reset(self):
        return np.random.rand(self.state_dim) * 10
    
    def step(self, action):
        state = np.random.rand(self.state_dim) * 10
        reward = -np.sum((state - action) ** 2)
        done = np.random.random() < 0.1
        return state, reward, done, {}
