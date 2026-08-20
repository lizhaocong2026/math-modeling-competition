# CUMCM 2025 A - Smart Grid: Deepened Case Study with PINN and Extended Analysis
import numpy as np
from typing import Dict, Any, List, Optional
import math
from algorithms.transformer import SimpleTransformer
from algorithms.transformer_ensemble import TransformerEnsemble
from algorithms.nsga2 import NSGAII
from algorithms.grey_model import GM11
from algorithms.stl_decompose import STLDecomposer
from algorithms.reinforcement import QLearningAgent, SARSAgent, SimpleEnv
from algorithms.pinn import PINN


class SmartGridDeepOptimizer:
    def __init__(self, horizon=24, seed=2025):
        self.horizon = horizon
        self.seed = seed
        np.random.seed(seed)

    def generate_synthetic_data(self, days=30):
        t = np.arange(0, days * 24, 1)
        base_load = 500
        daily_pattern = 100 * np.sin(2 * np.pi * (t - 14) / 24)
        weekly_pattern = 30 * np.sin(2 * np.pi * t / 168)
        weekend_dip = 50 * np.where((t % 168) >= 120, 1, 0)
        noise = np.random.normal(0, 15, len(t))
        return np.maximum(base_load + daily_pattern + weekly_pattern - weekend_dip + noise, 100)

    def forecast_transformer(self, historical_data, seq_len=48):
        actual_seq = min(seq_len, len(historical_data))
        X = historical_data[-actual_seq:].reshape(1, actual_seq, 1)
        y = historical_data[-actual_seq:]
        ensemble = TransformerEnsemble(n_estimators=3)
        ensemble.fit(X, y, epochs=15)
        forecast = ensemble.predict_steps(X, steps=self.horizon)
        pred_last = ensemble.predict(X)
        return {'method': 'TransformerEnsemble', 'forecast': forecast,
                'last_pred': float(pred_last[-1]) if len(pred_last) > 0 else None,
                'input_seq_len': actual_seq}

    def forecast_arima_equivalent(self, historical_data, p=2, d=1, q=1):
        from algorithms.arima import ARIMA
        n = min(48, len(historical_data))
        arima = ARIMA(p=p, d=d, q=q)
        return arima.fit(historical_data[-n:]).forecast(steps=self.horizon)

    def forecast_lstm(self, historical_data, seq_len=48):
        from algorithms.lstm import SimpleLSTM
        actual_seq = min(seq_len, len(historical_data))
        X = historical_data[-actual_seq:].reshape(1, actual_seq, 1)
        y = historical_data[-actual_seq:]
        lstm = SimpleLSTM(input_dim=1, lstm_units=16, output_dim=1, seed=self.seed)
        lstm.fit(X, y, epochs=15)
        pred = lstm.predict(X)
        forecast = lstm.predict_steps(X, steps=self.horizon)
        return {'method': 'SimpleLSTM', 'forecast': forecast, 'last_pred': float(pred[-1])}

    def forecast_pinn(self, historical_data, seq_len=48):
        n = min(seq_len, len(historical_data))
        X_train = np.linspace(0, 1, n).reshape(-1, 1)
        y_train = historical_data[-n:].reshape(-1, 1)
        pinn = PINN(input_dim=1, output_dim=1, hidden=[32, 32], lr=1e-3)
        pinn.fit(X_train, y_train, epochs=50, lam=1.0)
        X_test = np.linspace(0, 1, n + self.horizon).reshape(-1, 1)
        forecast = pinn.predict(X_test[n:]).flatten()
        last_pred = pinn.predict(X_train[-1:]).flatten()[0]
        return {'method': 'PINN', 'forecast': forecast, 'last_pred': float(last_pred)}

    def full_forecast_comparison(self, historical_data):
        results = {}
        results['transformer'] = self.forecast_transformer(historical_data)
        results['arima'] = self.forecast_arima_equivalent(historical_data)
        results['lstm'] = self.forecast_lstm(historical_data)
        results['pinn'] = self.forecast_pinn(historical_data)
        return results

    def build_rl_environment(self, forecast_data):
        class SmartGridEnv:
            def __init__(self, load_forecast, horizon=24):
                self.horizon = horizon
                self.load_forecast = load_forecast
                self.time_step = 0
                self.battery_energy = 50.0
            def reset(self):
                self.time_step = 0
                self.battery_energy = 50.0
                return self._get_state()
            def _get_state(self):
                load_norm = min(self.load_forecast[self.time_step] / 800.0, 1.0)
                bat_norm = self.battery_energy / 100.0
                return np.array([load_norm, bat_norm, self.time_step / self.horizon])
            def step(self, action):
                action_map = {0: (-1,-1), 1: (-1,0), 2: (-1,1), 3: (0,-1), 4: (0,0), 5: (0,1), 6: (1,0)}
                u_bat, u_gen = action_map[action]
                self.battery_energy = np.clip(self.battery_energy + u_bat * 5, 0, 100)
                load = self.load_forecast[self.time_step]
                supply = 500 + u_gen * 50
                cost = abs(supply - load) * 0.01 + abs(u_bat) * 2
                satisfaction = max(0, 100 - abs(supply - load))
                reward = satisfaction - cost
                self.time_step += 1
                done = self.time_step >= self.horizon
                return self._get_state(), reward, done, {'cost': cost}
        return SmartGridEnv(forecast_data, horizon=self.horizon)

    def solve_rl_scheduling(self, forecast_data):
        env = self.build_rl_environment(forecast_data)
        agent = QLearningAgent(state_dim=3, action_dim=7, lr=0.1, gamma=0.95, epsilon=0.3)
        result = agent.fit(env, episodes=100)
        total_reward = sum(agent.episode_rewards[-10:]) / 10
        return {'method': 'Q-Learning', 'avg_reward_last10': float(total_reward),
                'q_table_size': result['q_table_size'], 'final_epsilon': result['final_epsilon']}

    def solve_multi_objective(self, forecast_data):
        def objectives(x):
            cost = np.sum(x[:3] * np.array([0.3, 0.5, 0.8])) * 1000
            carbon = np.sum(x[:3] * np.array([0.1, 0.3, 0.9])) * 100
            reliability = 1.0 - np.abs(np.mean(x[3:]) - 0.5) * 100
            return np.array([cost, carbon, -reliability])
        nsga = NSGAII(n_objectives=3, pop_size=40, max_gen=60)
        bounds = [(0, 1)] * 6
        result = nsga.optimize(objective_func=objectives, bounds=bounds)
        pareto_objs = result.get('pareto_objectives', [])
        return {'method': 'NSGA-II', 'pareto_solutions': result.get('n_solutions', 0),
                'best_cost': float(pareto_objs[0][0]) if pareto_objs else None,
                'best_carbon': float(pareto_objs[0][1]) if pareto_objs else None,
                'best_reliability': float(-pareto_objs[0][2] / 100) if pareto_objs else None}

    def full_pipeline(self, historical_data=None):
        if historical_data is None:
            historical_data = self.generate_synthetic_data(days=30)
        forecast_comparison = self.full_forecast_comparison(historical_data)
        rl_result = self.solve_rl_scheduling(forecast_comparison['transformer']['forecast'])
        mo_result = self.solve_multi_objective(forecast_comparison['transformer']['forecast'])
        return {'forecast_comparison': forecast_comparison, 'rl_scheduling': rl_result,
                'multi_objective': mo_result, 'data_points': len(historical_data)}


if __name__ == '__main__':
    np.random.seed(42)
    optimizer = SmartGridDeepOptimizer(horizon=24, seed=2025)
    result = optimizer.full_pipeline()
    print('CUMCM 2025 A - Smart Grid Deepened')
    print('Forecast methods:', list(result['forecast_comparison'].keys()))
    for method, res in result['forecast_comparison'].items():
        last = res.get('last_pred', 'N/A')
        print(f'  {method}: last_pred={last:.2f}')
    print('RL:', result['rl_scheduling']['method'])
    print('MO solutions:', result['multi_objective']['pareto_solutions'])
