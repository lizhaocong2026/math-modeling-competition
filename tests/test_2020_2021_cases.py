
import os
import unittest
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')


class Test2020ACase(unittest.TestCase):
    def test_parking_pricing(self):
        from cases.cumcm_2020a_parking_pricing import ParkingPricing
        model = ParkingPricing(n_parking=5)
        result = model.full_analysis()
        self.assertEqual(result['method'], 'Stackelberg Game')
        self.assertEqual(len(result['optimal_prices']), 5)
        self.assertGreater(result['total_profit'], 0)


class Test2020BCase(unittest.TestCase):
    def test_ev_station_planning(self):
        from cases.cumcm_2020b_ev_station import EVStationPlanner
        planner = EVStationPlanner(n_candidates=10, n_users=50, budget=500000)
        result = planner.solve_multi_objective()
        self.assertEqual(result['method'], 'NSGA-II')
        self.assertGreater(result['pareto_solutions'], 0)


class Test2021ACase(unittest.TestCase):
    def test_power_network_optimization(self):
        from cases.cumcm_2021a_power_network import PowerNetworkOptimizer
        model = PowerNetworkOptimizer(n_generators=6, horizon=24)
        result = model.solve_ed_problem()
        self.assertEqual(result['method'], 'Particle Swarm Optimization')
        self.assertEqual(len(result['optimal_power']), 6)
        self.assertGreater(result['min_cost'], 0)


class Test2021BCase(unittest.TestCase):
    def test_traffic_flow_prediction(self):
        from cases.cumcm_2021b_traffic_flow import TrafficFlowPredictor
        predictor = TrafficFlowPredictor(seq_len=24, horizon=12)
        data = predictor.generate_traffic_data()
        result = predictor.forecast_transformer(data)
        self.assertEqual(result['method'], 'Transformer')
        self.assertGreater(len(result['predictions']), 0)


class TestNewAlgorithms(unittest.TestCase):
    def test_transformer_forecast(self):
        from algorithms.transformer import SimpleTransformer
        model = SimpleTransformer(d_model=16, nhead=2, num_layers=1, seq_len=12)
        X = np.random.rand(30, 12, 1)
        y = np.random.rand(30)
        result = model.fit(X, y, epochs=5)
        self.assertEqual(result['status'], 'success')
        pred = model.predict(X)
        self.assertEqual(len(pred), 30)

    def test_nsga2_multiobjective(self):
        from algorithms.nsga2 import NSGAII
        def objectives(x):
            return np.array([x[0]**2, x[1]**2])
        nsga = NSGAII(n_objectives=2, pop_size=20, max_gen=20)
        result = nsga.optimize(objectives, bounds=[(-10, 10), (-10, 10)])
        self.assertGreater(result['n_solutions'], 0)
        self.assertEqual(len(result['pareto_objectives'][0]), 2)

    def test_pso_optimize(self):
        from algorithms.pso import ParticleSwarm
        ps = ParticleSwarm(n_particles=20, max_iter=100)
        result = ps.optimize(lambda x: (x[0] - 0)**2, bounds=[(-10, 10)])
        self.assertEqual(result['success'], True)
        self.assertIsInstance(result['optimal_value'], (int, float))


if __name__ == '__main__':
    unittest.main()
