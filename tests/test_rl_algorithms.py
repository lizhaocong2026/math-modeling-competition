"""Tests for Reinforcement Learning algorithms (QLearning, SARSA, RLHF)"""
import unittest
import numpy as np
import sys
sys.path.insert(0, r'D:\\本地的知识库构建\\math-modeling-competition')


class TestQLearningAgent(unittest.TestCase):
    def test_init(self):
        from algorithms.reinforcement import QLearningAgent
        agent = QLearningAgent(state_dim=3, action_dim=2)
        self.assertEqual(agent.state_dim, 3)
        self.assertEqual(agent.action_dim, 2)

    def test_choose_action_deterministic(self):
        from algorithms.reinforcement import QLearningAgent
        agent = QLearningAgent(state_dim=2, action_dim=2, epsilon=0.0)
        state = np.array([1.0, 0.5])
        action = agent.choose_action(state)
        self.assertIn(action, [0, 1])

    def test_choose_action_exploration(self):
        from algorithms.reinforcement import QLearningAgent
        agent = QLearningAgent(state_dim=2, action_dim=2, epsilon=1.0)
        actions = [agent.choose_action(np.zeros(2)) for _ in range(100)]
        self.assertIn(0, actions)
        self.assertIn(1, actions)

    def test_update(self):
        from algorithms.reinforcement import QLearningAgent
        agent = QLearningAgent(state_dim=2, action_dim=2, lr=0.1, gamma=0.95)
        state = np.array([1.0, 0.5])
        action = 0
        reward = 1.0
        next_state = np.array([0.5, 1.0])
        done = False
        td_error = agent.update(state, action, reward, next_state, done)
        self.assertIsInstance(td_error, float)
        self.assertTrue(td_error >= 0)

    def test_decay_epsilon(self):
        from algorithms.reinforcement import QLearningAgent
        agent = QLearningAgent(state_dim=2, action_dim=2, epsilon=0.5, epsilon_decay=0.99)
        old_eps = agent.epsilon
        agent.decay_epsilon()
        self.assertLess(agent.epsilon, old_eps)
        self.assertGreaterEqual(agent.epsilon, 0.01)

    def test_fit(self):
        from algorithms.reinforcement import QLearningAgent, SimpleEnv
        np.random.seed(42)
        agent = QLearningAgent(state_dim=3, action_dim=2, epsilon=0.3)
        env = SimpleEnv(n_resources=3, n_actions=2)
        result = agent.fit(env, episodes=20)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(agent.episode_rewards), 20)

    def test_get_best_policy(self):
        from algorithms.reinforcement import QLearningAgent
        agent = QLearningAgent(state_dim=2, action_dim=2, epsilon=0.0)
        policy = agent.get_best_policy()
        self.assertIsInstance(policy, dict)


class TestSARSAgent(unittest.TestCase):
    def test_init(self):
        from algorithms.reinforcement import SARSAgent
        agent = SARSAgent(state_dim=3, action_dim=2)
        self.assertEqual(agent.state_dim, 3)

    def test_choose_action(self):
        from algorithms.reinforcement import SARSAgent
        agent = SARSAgent(state_dim=2, action_dim=2, epsilon=0.0)
        action = agent.choose_action(np.array([1.0, 0.5]))
        self.assertIn(action, [0, 1])

    def test_fit(self):
        from algorithms.reinforcement import SARSAgent, SimpleEnv
        np.random.seed(42)
        agent = SARSAgent(state_dim=3, action_dim=2, epsilon=0.3)
        env = SimpleEnv(n_resources=3, n_actions=2)
        result = agent.fit(env, episodes=20)
        self.assertEqual(result["status"], "success")


class TestRLHFTrainer(unittest.TestCase):
    def test_init(self):
        from algorithms.rlhf import RLHFTrainer
        trainer = RLHFTrainer(num_actions=5)
        self.assertEqual(trainer.num_actions, 5)

    def test_train_step(self):
        from algorithms.rlhf import RLHFTrainer
        trainer = RLHFTrainer(num_actions=5)
        loss = trainer.train_step(reference_action=0, chosen_action=1, rejected_action=2)
        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0)

    def test_fit(self):
        from algorithms.rlhf import RLHFTrainer
        trainer = RLHFTrainer(num_actions=5)
        preferences = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        result = trainer.fit(preferences, epochs=5)
        self.assertEqual(result["status"], "success")

    def test_generate_response(self):
        from algorithms.rlhf import RLHFTrainer
        trainer = RLHFTrainer(num_actions=5)
        response = trainer.generate_response(prompt_idx=0)
        self.assertIn(response, range(5))

    def test_reward_model(self):
        from algorithms.rlhf import RLHFTrainer
        trainer = RLHFTrainer(num_actions=5)
        reward = trainer.reward_model(action_idx=0)
        self.assertIsInstance(reward, float)


if __name__ == "__main__":
    unittest.main()