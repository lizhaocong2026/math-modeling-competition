import numpy as np
from typing import Dict, Any, List, Tuple

class RLHFTrainer:
    def __init__(self, num_actions=10, lr=1e-3):
        self.num_actions = num_actions
        self.lr = lr
        self.reward_model_w = np.random.randn(num_actions, 1) * 0.01
        self.policy_logits = np.zeros(num_actions)
        self.history = []

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-10)

    def reward_model(self, action_idx):
        return float(self.reward_model_w[action_idx % self.num_actions, 0])

    def policy(self, state=None):
        logits = self.policy_logits + np.random.randn(self.num_actions) * 0.01
        probs = self._softmax(logits)
        return np.random.choice(self.num_actions, p=probs)

    def train_step(self, reference_action, chosen_action, rejected_action):
        r_ref = self.reward_model(reference_action)
        r_chosen = self.reward_model(chosen_action)
        r_rejected = self.reward_model(rejected_action)
        log_prob_chosen = np.log(self._softmax(self.policy_logits)[chosen_action] + 1e-10)
        log_prob_rejected = np.log(self._softmax(self.policy_logits)[rejected_action] + 1e-10)
        loss = -r_chosen * log_prob_chosen + r_rejected * log_prob_rejected
        grad = -r_chosen * (self._softmax(self.policy_logits) == 1).astype(float) * chosen_action
        self.policy_logits -= self.lr * grad
        self.history.append(loss)
        return float(loss)

    def fit(self, preferences: List[Tuple[int, int, int]], epochs=50, verbose=False):
        losses = []
        for ep in range(epochs):
            total_loss = 0.0
            for ref, chosen, rejected in preferences:
                total_loss += self.train_step(ref, chosen, rejected)
            avg_loss = total_loss / max(len(preferences), 1)
            losses.append(avg_loss)
            if verbose and (ep % 10 == 0 or ep == epochs-1):
                print("Epoch %d/%d loss=%.6f", ep, epochs, avg_loss)
        self.history = losses
        return {"status": "success", "final_loss": losses[-1] if losses else None}

    def generate_response(self, prompt_idx):
        return self.policy()