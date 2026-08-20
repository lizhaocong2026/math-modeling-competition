"""
Proximal Policy Optimization (PPO) for discrete control
Simplified PPO implementation for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List, Tuple
import collections


class SimplePolicy:
    """Linear policy network: action = W * state + b"""
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.W = np.random.randn(action_dim, state_dim) * 0.01
        self.b = np.zeros(action_dim)
        self.log_probs = []
        self.states = []
        self.actions = []
    
    def select_action(self, state, epsilon=0.1):
        state = np.asarray(state).flatten()
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        mean = self.W @ state + self.b
        return int(np.argmax(mean))
    
    def log_prob(self, state, action):
        state = np.asarray(state).flatten()
        mean = self.W @ state + self.b
        one_hot = np.eye(self.action_dim)[action]
        diff = one_hot - mean
        return float(-0.5 * diff @ diff)
    
    def update(self, advantages, lr=0.001, epochs=1):
        if len(self.states) == 0:
            return 0.0
        states = np.array(self.states)
        actions = np.array(self.actions)
        adv = np.array(advantages)
        actions_oh = np.eye(self.action_dim)[actions]
        
        for ep in range(epochs):
            means = states @ self.W.T + self.b
            diff = actions_oh - means
            loss_grad_W = -(diff.T @ states) / max(len(states), 1)
            loss_grad_b = -adv.mean()
            self.W -= lr * loss_grad_W
            self.b -= lr * loss_grad_b
        
        self.states, self.actions = [], []
        return float(np.mean(advantages**2))


class PPOMDPEnv:
    """Simple MDP environment for PPO testing"""
    
    def __init__(self, n_states=4, n_actions=2, seed=42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)
        self.transition = self.rng.rand(n_states, n_actions, n_states).astype(np.float32)
        self.transition /= self.transition.sum(axis=-1, keepdims=True)
        self.reward = self.rng.rand(n_states, n_actions).astype(np.float32) * 2 - 1
        self.state = None
    
    def reset(self):
        self.state = self.rng.randint(0, self.n_states)
        return self._get_obs()
    
    def _get_obs(self):
        obs = np.zeros(self.n_states)
        obs[self.state] = 1.0
        return obs
    
    def step(self, action):
        probs = np.asarray(self.transition[self.state, action])
        if probs.ndim == 0:
            probs = np.array([probs, 1.0-probs])
        elif probs.size != self.n_states:
            probs = probs.flatten()[:self.n_states]
        self.state = int(self.rng.choice(self.n_states, p=probs))
        reward = float(self.reward[self.state, action])
        done = False
        return self._get_obs(), reward, done, {}
    
    def get_optimal_policy(self):
        return np.argmax(self.reward, axis=1)


class PPOAgent:
    """PPO agent for discrete control problems"""
    
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.95, 
                 clip_epsilon=0.2, seed=42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.policy = SimplePolicy(state_dim, action_dim)
        self.episode_rewards = []
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def choose_action(self, state, epsilon=0.05):
        if self.rng.random() < epsilon:
            return self.rng.randint(0, self.action_dim)
        return self.policy.select_action(state, epsilon=0)
    
    def train(self, env, episodes=500, batch_size=64):
        self.episode_rewards = []
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0.0
            traj_s, traj_a, traj_r = [], [], []
            done = False
            eps = max(0.05, 0.3 * (1 - ep / (episodes + 1)))
            while not done:
                action = self.choose_action(state, epsilon=eps)
                next_state, reward, done, info = env.step(action)
                traj_s.append(state)
                traj_a.append(action)
                traj_r.append(reward)
                state = next_state
                total_reward += reward
                if done or len(traj_s) >= batch_size:
                    if len(traj_s) >= 8:
                        rewards = np.array(traj_r)
                        advantages = np.zeros(len(rewards))
                        for t in range(len(rewards)):
                            advantages[t] = rewards[t:] @ (self.gamma ** np.arange(len(rewards) - t))
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        actions_oh = np.eye(self.action_dim)[traj_a]
                        means = np.array(traj_s) @ self.policy.W.T + self.policy.b
                        diff = actions_oh - means
                        grad_W = -(diff.T @ np.array(traj_s)) / max(len(traj_s), 1)
                        self.policy.W -= self.lr * grad_W
                        self.policy.b -= self.lr * advantages.mean(axis=0)
                    traj_s, traj_a, traj_r = [], [], []
            self.episode_rewards.append(total_reward)
            if (ep + 1) % 50 == 0:
                avg = np.mean(self.episode_rewards[-50:])
                print(f"  Episode {ep+1}/{episodes}, Avg: {avg:.3f}")
        return self
    
    def get_policy_value(self, state):
        return self.policy.select_action(state, epsilon=0)
