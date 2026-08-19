# CUMCM 2025 C - Reinforcement Learning for Energy Management
"""
CUMCM 2025 C: Multi-Agent Reinforcement Learning for Building Energy Management

Problem Background:
- Smart buildings need dynamic energy management
- Traditional rule-based control is suboptimal
- RL can learn optimal scheduling policies

RL Formulation:
- State: Current temperature, occupancy, energy price, time of day
- Action: HVAC setpoint adjustment
- Reward: Comfort + Energy cost penalty
- Algorithm: Q-Learning / SARSA for discrete actions
"""
import numpy as np
from typing import Dict, Any, Tuple
from algorithms.reinforcement import QLearningAgent


class BuildingEnergyEnv:
    """
    Simplified building energy management environment
    """
    
    def __init__(self, n_temperature_bins=5, n_occupancy=3, 
                 n_price=3, n_time=4, n_actions=4):
        self.n_temperature_bins = n_temperature_bins
        self.n_occupancy = n_occupancy
        self.n_price = n_price
        self.n_time = n_time
        self.n_actions = n_actions
        
        self.state_dims = (n_temperature_bins, n_occupancy, n_price, n_time)
        self.state_size = int(np.prod(self.state_dims))
        self.action_size = n_actions
        
    def reset(self) -> Tuple[int, ...]:
        np.random.seed()
        temp_error = np.random.uniform(-1.5, 1.5)
        occupancy = np.random.randint(0, self.n_occupancy)
        price_level = np.random.randint(0, self.n_price)
        time_block = np.random.randint(0, self.n_time)
        temp_bin = int(np.clip(temp_error + 2, 0, self.n_temperature_bins - 1))
        return (temp_bin, occupancy, price_level, time_block)
    
    def step(self, state: Tuple[int, ...], action: int) -> Tuple[Tuple[int, ...], float, bool, Dict]:
        temp_bin, occupancy, price_level, time_block = state
        
        power_levels = [0, 0.3, 0.6, 1.0]
        power = power_levels[action]
        
        temp_change = power * 2.0 - 0.5 * occupancy / self.n_occupancy
        new_temp_error = temp_bin - 2 + temp_change + np.random.normal(0, 0.1)
        
        comfort_penalty = abs(new_temp_error) * 10
        price_factors = [0.5, 1.0, 1.5]
        energy_cost = power * price_factors[price_level] * 100
        occupancy_bonus = occupancy * 5 if new_temp_error < 1 else 0
        
        reward = -comfort_penalty - energy_cost + occupancy_bonus
        
        new_temp_bin = int(np.clip(new_temp_error + 2, 0, self.n_temperature_bins - 1))
        new_occupancy = np.random.randint(0, self.n_occupancy)
        new_price = np.random.randint(0, self.n_price)
        new_time = (time_block + 1) % self.n_time
        
        new_state = (new_temp_bin, new_occupancy, new_price, new_time)
        done = time_block == self.n_time - 1
        
        info = {"power_used": power, "temp_error": new_temp_error}
        return new_state, reward, done, info
    
    def get_state_index(self, state: Tuple[int, ...]) -> int:
        return (state[0] * self.n_occupancy + state[1]) * self.n_price * self.n_time + \
               state[2] * self.n_time + state[3]


def run_rl_energy_case():
    """Run RL-based energy management case study"""
    print("=" * 60)
    print("CUMCM 2025 C - RL Building Energy Management")
    print("=" * 60)
    
    env = BuildingEnergyEnv()
    
    # Use continuous state representation
    agent = QLearningAgent(
        state_dim=4,  # 4 state features
        action_dim=env.action_size,
        lr=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    n_episodes = 500
    steps_per_episode = env.n_time
    
    print(f"\nTraining Q-Learning Agent...")
    print(f"  Episodes: {n_episodes}")
    print(f"  State dims: {env.state_size} discrete states")
    print(f"  Actions: {env.action_size}")
    
    # Custom training loop
    for episode in range(n_episodes):
        state_tuple = env.reset()
        state = np.array(state_tuple, dtype=float)
        total_reward = 0
        
        for step in range(steps_per_episode):
            action = agent.choose_action(state)
            new_state_tuple, reward, done, info = env.step(state_tuple, action)
            new_state = np.array(new_state_tuple, dtype=float)
            
            agent.update(state, action, reward, new_state, done)
            
            total_reward += reward
            state = new_state
            state_tuple = new_state_tuple
            
            if done:
                break
        
        agent.decay_epsilon()
        
        if episode % 100 == 0:
            print(f"  Episode {episode}: Reward={total_reward:.1f}, Epsilon={agent.epsilon:.3f}")
    
    # Evaluate
    print(f"\nEvaluating trained policy...")
    eval_rewards = []
    eval_energies = []
    
    for _ in range(50):
        state_tuple = env.reset()
        state = np.array(state_tuple, dtype=float)
        total_reward = 0
        total_energy = 0
        
        for step in range(steps_per_episode):
            action = agent.choose_action(state)
            new_state_tuple, reward, done, info = env.step(state_tuple, action)
            new_state = np.array(new_state_tuple, dtype=float)
            
            total_reward += reward
            total_energy += info["power_used"]
            state = new_state
            state_tuple = new_state_tuple
            
            if done:
                break
        
        eval_rewards.append(total_reward)
        eval_energies.append(total_energy)
    
    avg_reward = np.mean(eval_rewards)
    avg_energy = np.mean(eval_energies)
    
    print(f"\nEvaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Energy: {avg_energy:.2f} kWh")
    print(f"  Energy Variance: {np.var(eval_energies):.2f}")
    
    print("\n" + "=" * 60)
    print("RL Energy Management Case Study Completed!")
    print("=" * 60)
    
    return {
        "training_episodes": n_episodes,
        "avg_reward": float(avg_reward),
        "avg_energy": float(avg_energy),
        "final_epsilon": agent.epsilon,
        "q_table_size": len(agent.q_table)
    }


if __name__ == "__main__":
    result = run_rl_energy_case()
