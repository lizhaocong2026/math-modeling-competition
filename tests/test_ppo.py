"""Tests for PPO Agent"""
import os
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algorithms'))


class TestSimplePolicy:
    def test_select_action_deterministic(self):
        from ppo_agent import SimplePolicy
        policy = SimplePolicy(state_dim=3, action_dim=2)
        state = np.array([1.0, 0.5, 0.3])
        action = policy.select_action(state, epsilon=0.0)
        assert action in [0, 1]

    def test_select_action_exploration(self):
        from ppo_agent import SimplePolicy
        policy = SimplePolicy(state_dim=3, action_dim=2)
        actions = [policy.select_action(np.zeros(3), epsilon=1.0) for _ in range(100)]
        assert 0 in actions and 1 in actions

    def test_log_prob_shape(self):
        from ppo_agent import SimplePolicy
        policy = SimplePolicy(state_dim=3, action_dim=2)
        state = np.array([1.0, 0.5, 0.3])
        lp = policy.log_prob(state, 0)
        assert isinstance(lp, float)
        assert np.isfinite(lp)

    def test_update_no_crash(self):
        from ppo_agent import SimplePolicy
        policy = SimplePolicy(state_dim=3, action_dim=2)
        for _ in range(20):
            state = np.random.randn(3)
            action = np.random.randint(2)
            policy.states.append(state)
            policy.actions.append(action)
        loss = policy.update(advantages=np.random.randn(20), lr=0.001, epochs=1)
        assert isinstance(loss, float)
        assert np.isfinite(loss)


class TestPPOMDPEnv:
    def test_reset(self):
        from ppo_agent import PPOMDPEnv
        env = PPOMDPEnv(n_states=4, n_actions=2, seed=42)
        obs = env.reset()
        assert len(obs) == 4
        assert obs.sum() == 1.0

    def test_step(self):
        from ppo_agent import PPOMDPEnv
        env = PPOMDPEnv(n_states=4, n_actions=2, seed=42)
        obs, reward, done, info = env.step(0)
        assert len(obs) == 4
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_optimal_policy(self):
        from ppo_agent import PPOMDPEnv
        env = PPOMDPEnv(n_states=4, n_actions=2, seed=42)
        optimal = env.get_optimal_policy()
        assert len(optimal) == 4
        assert all(a in [0, 1] for a in optimal)


class TestPPOAgent:
    def test_choose_action(self):
        from ppo_agent import PPOAgent, PPOMDPEnv
        env = PPOMDPEnv(n_states=4, n_actions=2, seed=42)
        agent = PPOAgent(state_dim=4, action_dim=2, seed=42)
        state = env.reset()
        action = agent.choose_action(state, epsilon=0.0)
        assert action in [0, 1]

    def test_train_short(self):
        from ppo_agent import PPOAgent, PPOMDPEnv
        env = PPOMDPEnv(n_states=4, n_actions=2, seed=42)
        agent = PPOAgent(state_dim=4, action_dim=2, seed=42)
        agent.train(env, episodes=10)
        assert len(agent.episode_rewards) == 10
        assert all(isinstance(r, float) for r in agent.episode_rewards)

    def test_policy_value_returns_int(self):
        from ppo_agent import PPOAgent, PPOMDPEnv
        env = PPOMDPEnv(n_states=4, n_actions=2, seed=42)
        agent = PPOAgent(state_dim=4, action_dim=2, seed=42)
        state = env.reset()
        val = agent.get_policy_value(state)
        assert isinstance(val, int)
        assert val in [0, 1]
