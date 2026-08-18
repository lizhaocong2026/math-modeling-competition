# Markov Chain Algorithm
import numpy as np
from typing import List

class MarkovChain:
    def __init__(self, states=None):
        self.states = states
        self.transition_matrix = None
        self.initial_distribution = None

    def fit(self, sequence):
        n_states = max(sequence) + 1
        self.states = list(range(n_states))
        count_matrix = np.zeros((n_states, n_states))
        for i in range(len(sequence) - 1):
            count_matrix[sequence[i]][sequence[i+1]] += 1
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = count_matrix / row_sums
        self.initial_distribution = np.zeros(n_states)
        self.initial_distribution[sequence[0]] = 1.0
        return self

    def predict(self, steps=1):
        distribution = self.initial_distribution.copy()
        for _ in range(steps):
            distribution = distribution @ self.transition_matrix
        return distribution

    def predict_sequence(self, steps=10, seed=None):
        if seed is not None:
            np.random.seed(seed)
        n_states = len(self.states)
        sequence = [np.random.choice(n_states, p=self.initial_distribution)]
        for _ in range(steps - 1):
            current = sequence[-1]
            next_state = np.random.choice(n_states, p=self.transition_matrix[current])
            sequence.append(next_state)
        return sequence

    def get_stationary_distribution(self):
        n = self.transition_matrix.shape[0]
        A = self.transition_matrix.T - np.eye(n)
        A[-1, :] = 1
        b = np.zeros(n)
        b[-1] = 1
        try:
            stationary = np.linalg.solve(A, b)
            return np.abs(stationary)
        except np.linalg.LinAlgError:
            dist = np.ones(n) / n
            for _ in range(1000):
                dist = dist @ self.transition_matrix
            return dist
