import os
BASE = r"D:\本地的知识库构建\math-modeling-competition"

# Fix transformer_ensemble.py
te_path = os.path.join(BASE, "algorithms/transformer_ensemble.py")
te = open(te_path, "r", encoding="utf-8").read()
te = te.replace('from transformer import SimpleTransformer', 
    "import sys, os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)))); from transformer import SimpleTransformer")
te = te.replace('"seq_len": min(24, X.shape[1])', '"seq_len": X.shape[1]')
te = te.replace('"seq_len": min(48, X.shape[1])', '"seq_len": X.shape[1]')
te = te.replace('"seq_len": min(36, X.shape[1])', '"seq_len": X.shape[1]')
open(te_path, "w", encoding="utf-8").write(te)
print(f"Fixed transformer_ensemble.py: {len(te)} chars")

# Fix ppo_agent.py  
ppo_path = os.path.join(BASE, "algorithms/ppo_agent.py")
ppo = open(ppo_path, "r", encoding="utf-8").read()

# Fix update method
ppo = ppo.replace(
    "loss_grad = -(actions - means) @ adv[:, np.newaxis] / len(states)",
    "actions_oh = np.eye(self.action_dim)[actions]\n            loss_grad = -(actions_oh - means) * adv[:, np.newaxis] / len(states)"
)

# Add PPOAgent class if missing
if "class PPOAgent" not in ppo:
    ppo += '''

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.95, clip_epsilon=0.2, seed=42):
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
                        grad_W = -((actions_oh - means) * advantages[:, np.newaxis]).mean(axis=0).T
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
'''
open(ppo_path, "w", encoding="utf-8").write(ppo)
print(f"Fixed ppo_agent.py: {len(ppo)} chars")
