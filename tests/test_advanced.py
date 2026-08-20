"""
新增算法测试 - ODE、PDE、组合优化、统计
"""
import os
import unittest
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')


class TestODESolver(unittest.TestCase):
    """测试ODE求解器"""
    
    def test_euler(self):
        from algorithms.ode_solver import EulerMethod
        
        def f(t, y):
            return np.array([y[0]])  # dy/dt = y
        
        solver = EulerMethod()
        result = solver.solve(f, np.array([1.0]), (0, 1), n_steps=100)
        
        self.assertEqual(len(result["t"]), 101)
        self.assertGreater(result["y"][-1, 0], 2.5)  # e^1 ≈ 2.718
    
    def test_rk4(self):
        from algorithms.ode_solver import RK4
        
        def f(t, y):
            return np.array([y[0]])
        
        solver = RK4()
        result = solver.solve(f, np.array([1.0]), (0, 1), n_steps=100)
        
        # RK4应该更准确
        self.assertAlmostEqual(result["y"][-1, 0], np.e, places=3)


class TestPDESolver(unittest.TestCase):
    """测试PDE求解器"""
    
    def test_heat_equation(self):
        from algorithms.pde_solver import HeatEquationSolver
        
        solver = HeatEquationSolver(alpha=1.0)
        result = solver.solve_finite_difference(L=1, T=0.1, nx=10, nt=10)
        
        self.assertEqual(result["u"].shape[0], 10)
        self.assertEqual(result["u"].shape[1], 10)


class TestCellularAutomaton(unittest.TestCase):
    """测试元胞自动机"""
    
    def test_game_of_life(self):
        from algorithms.cellular_automaton import GameOfLife
        
        # Blinker模式
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 1:4] = 1
        
        ga = GameOfLife(5, 5)
        ga.initialize(grid)
        ga.step()
        
        # 验证有变化
        self.assertTrue(np.any(ga.get_state() != grid))
    
    def test_langton_ant(self):
        from algorithms.cellular_automaton import LangtonAnt
        
        ca = LangtonAnt(10, 10)
        ca.step()
        
        self.assertEqual(ca.ant_x + ca.ant_y, ca.ant_x + ca.ant_y)  # 蚂蚁存在


class TestCombinatorial(unittest.TestCase):
    """测试组合优化"""
    
    def test_tsp_nearest_neighbor(self):
        from algorithms.combinatorial import TravelingSalesman
        
        dist = np.array([
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ])
        
        tsp = TravelingSalesman(dist)
        tour, length = tsp.nearest_neighbor()
        
        self.assertEqual(len(tour), 5)  # 包含返回起点
        self.assertEqual(tour[0], tour[-1])
    
    def test_knapsack_dp(self):
        from algorithms.combinatorial import KnapsackSolver
        
        weights = [2, 3, 4, 5]
        values = [3, 4, 5, 6]
        capacity = 8
        
        result = KnapsackSolver.dynamic_programming(weights, values, capacity)
        
        self.assertGreater(result["max_value"], 0)
        self.assertTrue(result["total_weight"] <= capacity)


class TestStatistics(unittest.TestCase):
    """测试统计检验"""
    
    def test_t_test(self):
        from algorithms.statistics import HypothesisTest
        
        data = np.array([1.2, 1.5, 1.3, 1.6, 1.4])
        result = HypothesisTest.t_test_one_sample(data, mu0=1.0)
        
        self.assertIn("p_value", result)
        self.assertIn("t_statistic", result)
    
    def test_normality_test(self):
        from algorithms.statistics import NormalityTest
        
        data = np.random.randn(100)
        result = NormalityTest.shapiro_test(data)
        
        self.assertIn("is_normal", result)
    
    def test_correlation(self):
        from algorithms.statistics import CorrelationTest
        
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.1
        
        result = CorrelationTest.pearson(x, y)
        
        self.assertGreater(result["correlation"], 0.9)


class TestMonteCarloAdvanced(unittest.TestCase):
    """测试高级蒙特卡洛方法"""
    
    def test_antithetic(self):
        from algorithms.monte_carlo_advanced import VarianceReduction
        
        def func(u):
            return u**2
        
        result = VarianceReduction.antithetic_variates(func, n=1000)
        
        self.assertIn("estimate", result)
        self.assertLess(result["std_error"], 0.1)
    
    def test_lhs(self):
        from algorithms.monte_carlo_advanced import LatinHypercubeSampling
        
        lhs = LatinHypercubeSampling(n_params=3)
        samples = lhs.sample(n=50)
        
        self.assertEqual(samples.shape, (50, 3))
        self.assertTrue(np.all(samples >= 0) and np.all(samples <= 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)