"""
数学建模竞赛代码库 - 中国大学生数学建模竞赛 (CUMCM)
"""
__version__ = "1.0.0"
__author__ = "lzc18"
__description__ = "数学建模竞赛核心算法库"

from .algorithms.optimization import LinearProgramming, IntegerProgramming, NonlinearProgramming
from .algorithms.ga import GeneticAlgorithm
from .algorithms.pso import ParticleSwarm
from .algorithms.sa import SimulatedAnnealing
from .algorithms.grey_model import GM11
from .algorithms.pca import PCA
from .algorithms.ahp import AHP
from .algorithms.topcis import TOPSIS
from .algorithms.entropy_weight import EntropyWeight
from .algorithms.linear_regression import LinearRegression
from .algorithms.polynomial_regression import PolynomialRegression
from .utils.data_preprocessor import DataPreprocessor
from .utils.helpers import Utils
from .visualizations.model_viz import ModelVisualization

__all__ = [
    'LinearProgramming',
    'IntegerProgramming', 
    'NonlinearProgramming',
    'GeneticAlgorithm',
    'ParticleSwarm',
    'SimulatedAnnealing',
    'GM11',
    'PCA',
    'AHP',
    'TOPSIS',
    'EntropyWeight',
    'LinearRegression',
    'PolynomialRegression',
    'DataPreprocessor',
    'Utils',
    'ModelVisualization'
]
