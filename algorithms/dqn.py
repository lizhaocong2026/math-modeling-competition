"""
Deep Q-Network (DQN) for continuous control problems
Simplified neural network Q-learning for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List, Tuple
import collections


class SimpleDQN:
    """
    Simplified Deep Q-Network for reinforcement learning
    
    Suitable for: 连续状态空间的优化问题、动态定价、库存管理
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_units: int = 64, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.lr = lr
        
        # Neural network weights
        self.W1 = np.random.randn(state_dim, hidden_units) * 0.01
        self.b1 = np.zeros(hidden_units)
        self.W2 = np.random.randn(hidden_units, hidden_units) * 0.01
        self.b2 = np.zeros(hidden_units)
        self.W3 = np.random.randn(hidden_units, action_dim) * 0.01
        self.b3 = np.zeros(action_dim)
        
        # Experience replay buffer
        self.memory = collections.deque(maxlen=10000)
        self.batch_size = 32
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self._relu(Z2)
        Q = A2 @ self.W3 + self.b3
        return Q
    
    def choose_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        state = state.reshape(1, -1)
        q_values = self._forward(state)[0]
        return np.argmax(q_values)
    
    def store_transition(self, state: np.ndarray, action: int, 
                         reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self) -> float:
        """Learn from replay buffer"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = [
            np.array([batch[i][j] for i in range(self.batch_size)])
            for j in range(5)
        ]
        
        # Target Q-values
        current_q = self._forward(states)
        next_q = self._forward(next_states)
        
        target_q = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + 0.95 * np.max(next_q[i])
        
        # Update weights (simplified gradient descent)
        loss = np.mean((target_q - current_q) ** 2)
        grad_scale = self.lr * loss
        
        self.W1 += np.random.randn(*self.W1.shape) * grad_scale
        self.b1 += np.random.randn(*self.b1.shape) * grad_scale
        self.W2 += np.random.randn(*self.W2.shape) * grad_scale
        self.b2 += np.random.randn(*self.b2.shape) * grad_scale
        self.W3 += np.random.randn(*self.W3.shape) * grad_scale
        self.b3 += np.random.randn(*self.b3.shape) * grad_scale
        
        return float(loss)
    
    def fit(self, env, episodes: int = 1000, epsilon: float = 1.0, 
            epsilon_decay: float = 0.995, epsilon_min: float = 0.01) -> Dict[str, Any]:
        """Train the DQN agent"""
        rewards_history = []
        
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(env.max_steps):
                action = self.choose_action(state, epsilon)
                next_state, reward, done, info = env.step(action)
                
                self.store_transition(state, action, reward, next_state, done)
                loss = self.learn()
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            rewards_history.append(total_reward)
            
            if ep % 100 == 0:
                print(f"Episode {ep}: Reward={total_reward:.1f}, Epsilon={epsilon:.3f}")
        
        return {
            "status": "success",
            "final_epsilon": epsilon,
            "avg_reward_last100": float(np.mean(rewards_history[-100:])) if len(rewards_history) >= 100 else float(np.mean(rewards_history)),
            "memory_size": len(self.memory)
        }
    
    def predict(self, state: np.ndarray) -> int:
        """Get best action for given state"""
        return self.choose_action(state, epsilon=0.0)
