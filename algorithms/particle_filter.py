"""
Particle Filter and Enhanced Kalman Filter for state estimation
Sequential Monte Carlo methods for nonlinear/non-Gaussian systems
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for nonlinear state estimation
    
    Linearizes nonlinear state space models using Jacobians
    Suitable for: 目标跟踪、导航定位、状态估计
    """
    
    def __init__(self, dim_state: int, dim_measure: int, 
                 process_noise: float = 0.01, measure_noise: float = 0.1):
        self.dim_state = dim_state
        self.dim_measure = dim_measure
        self.process_noise = process_noise
        self.measure_noise = measure_noise
        
        # State transition matrix (identity for now)
        self.F = np.eye(dim_state)
        # Measurement matrix
        self.H = np.zeros((dim_measure, dim_state))
        # Control input matrix
        self.B = np.zeros((dim_state, 1))
        
        # Covariance matrices
        self.P = np.eye(dim_state) * 1.0
        self.Q = np.eye(dim_state) * process_noise
        self.R = np.eye(dim_measure) * measure_noise
        
        # State estimate
        self.x = np.zeros(dim_state)
        
        self.residuals = []
    
    def predict(self, u=0.0) -> Dict[str, Any]:
        """Prediction step"""
        # Predict state
        self.x = self.F @ self.x + self.B * u
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return {"state": self.x.copy(), "covariance": self.P.copy()}
    
    def update(self, z: np.ndarray) -> Dict[str, Any]:
        """Update step with measurement"""
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I_KH = np.eye(self.dim_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        self.residuals.append(float(np.sqrt(np.mean(y ** 2))))
        
        return {
            "state": self.x.copy(),
            "covariance": self.P.copy(),
            "innovation": y,
            "residual": float(np.sqrt(np.mean(y ** 2)))
        }
    
    def filter(self, measurements: np.ndarray, controls: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run full EKF filter"""
        results = []
        controls = controls or np.zeros(len(measurements))
        
        for t in range(len(measurements)):
            self.predict(controls[t] if len(controls) > t else 0.0)
            result = self.update(measurements[t])
            results.append(result)
        
        return {
            "status": "success",
            "states": np.array([r["state"] for r in results]),
            "residuals": self.residuals,
            "final_rmse": float(np.sqrt(np.mean(np.array(self.residuals) ** 2)))
        }
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "dim_state": self.dim_state,
            "dim_measure": self.dim_measure,
            "process_noise": self.process_noise,
            "measure_noise": self.measure_noise
        }


class ParticleFilter:
    """
    Particle Filter (Sequential Monte Carlo) for nonlinear state estimation
    
    Uses importance sampling with resampling for non-Gaussian systems
    Suitable for: 非线性系统状态估计、多模态分布、复杂噪声环境
    """
    
    def __init__(self, n_particles: int = 1000, dim_state: int = 2,
                 process_std: float = 0.1, measure_std: float = 1.0):
        self.n_particles = n_particles
        self.dim_state = dim_state
        self.process_std = process_std
        self.measure_std = measure_std
        
        # Initialize particles
        self.particles = np.random.randn(n_particles, dim_state) * 5.0
        self.weights = np.ones(n_particles) / n_particles
        
        self.trajectory = []
        self.log_weights_history = []
    
    def _resample(self):
        """Systematic resampling"""
        cumulative = np.cumsum(self.weights)
        positions = np.random.uniform(0, cumulative[-1], self.n_particles)
        indices = np.searchsorted(cumulative, positions)
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def predict_step(self, u=0.0):
        """Predict: propagate particles through process model"""
        # Simple random walk with control
        self.particles += np.random.randn(*self.particles.shape) * self.process_std
        self.particles[:, 0] += u * 0.1
    
    def update_step(self, z: np.ndarray):
        """Update: compute weights based on measurement likelihood"""
        # Measurement model: z = H @ x + noise
        H = np.array([[1.0, 0.0]])  # Observe first state component
        predicted_z = H @ self.particles.T  # (1, n_particles)
        
        # Gaussian likelihood
        innovation = z - predicted_z.squeeze()
        log_weights = -0.5 * (innovation ** 2) / (self.measure_std ** 2)
        
        # Normalize weights
        log_weights -= np.max(log_weights)
        self.weights = np.exp(log_weights)
        self.weights /= np.sum(self.weights) + 1e-300
        
        self.log_weights_history.append(float(np.max(log_weights)))
    
    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state estimate from weighted particles"""
        estimate = (self.particles * self.weights[:, np.newaxis]).sum(axis=0)
        mean = estimate
        diff = self.particles - mean
        cov = (diff.T @ (diff * self.weights[:, None])) / (1.0 - np.max(self.weights) + 1e-10) + 1e-10 * np.eye(self.dim_state)
        return estimate, cov
    
    def filter(self, measurements: np.ndarray, controls: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run full particle filter"""
        results = []
        controls = controls or np.zeros(len(measurements))
        
        for t in range(len(measurements)):
            self.predict_step(controls[t] if len(controls) > t else 0.0)
            self.update_step(measurements[t])
            
            # Resample if effective sample size is low
            ESS = 1.0 / np.sum(self.weights ** 2)
            if ESS < self.n_particles / 2:
                self._resample()
            
            est, cov = self.estimate()
            self.trajectory.append(est.copy())
            results.append({
                "estimate": est,
                "covariance": cov,
                "ESS": float(ESS)
            })
        
        return {
            "status": "success",
            "trajectory": np.array(self.trajectory),
            "n_particles": self.n_particles,
            "final_ESS": results[-1]["ESS"] if results else 0.0
        }
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_particles": self.n_particles,
            "dim_state": self.dim_state,
            "process_std": self.process_std,
            "measure_std": self.measure_std
        }
