# CUMCM 2025 B - Transformer-based Time Series Forecasting Case Study
"""
CUMCM 2025 B: Traffic Flow Prediction using Transformer
This case demonstrates how to use Transformer for multi-step time series forecasting.

Problem Background:
- Smart city traffic management requires accurate short-term traffic flow prediction
- Traditional ARIMA models struggle with long-range dependencies
- Transformer captures temporal patterns through self-attention

Data Requirements:
- Historical traffic flow data (e.g., hourly data for 30 days)
- Input: (samples, seq_len, features) where features can include time, weather, etc.
- Output: Multi-step ahead predictions

Solution Approach:
1. Data preprocessing and feature engineering
2. Transformer model training with sliding window
3. Ensemble prediction for uncertainty quantification
4. Performance evaluation (MAE, RMSE, MAPE)
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from algorithms.transformer import SimpleTransformer, TransformerEnsemble


class TrafficFlowTransformer:
    """
    Transformer-based traffic flow forecasting system
    
    Use case: CUMCM B题预测类问题，适合有季节性和周期性的时间序列
    """
    
    def __init__(self, seq_len: int = 48, horizon: int = 24, 
                 d_model: int = 32, nhead: int = 4, n_ensemble: int = 5):
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model
        self.nhead = nhead
        self.n_ensemble = n_ensemble
        self.model = None
        self.ensemble = None
        
    def prepare_data(self, data: np.ndarray, seq_len: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sliding window data for transformer
        
        Args:
            data: 1D array of historical traffic flow
            seq_len: sequence length (default: self.seq_len)
            
        Returns:
            X: (n_samples, seq_len, 1) feature matrix
            y: (n_samples,) target values
        """
        seq_len = seq_len or self.seq_len
        n_samples = len(data) - seq_len
        
        X = np.zeros((n_samples, seq_len, 1))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            X[i, :, 0] = data[i:i + seq_len]
            y[i] = data[i + seq_len]
        
        return X, y
    
    def prepare_multivariate_data(self, data: np.ndarray, seq_len: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare multivariate data for transformer
        
        Args:
            data: 2D array (samples, features)
            seq_len: sequence length
            
        Returns:
            X: (n_samples, seq_len, n_features)
            y: (n_samples,)
        """
        seq_len = seq_len or self.seq_len
        n_samples = len(data) - seq_len
        n_features = data.shape[1]
        
        X = np.zeros((n_samples, seq_len, n_features))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            X[i] = data[i:i + seq_len]
            y[i] = data[i + seq_len, 0]  # Predict first feature
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.001) -> Dict[str, Any]:
        """
        Train the transformer model
        
        Args:
            X: Training features (n_samples, seq_len, n_features)
            y: Training targets (n_samples,)
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Training result dictionary
        """
        n_features = X.shape[-1] if X.ndim == 3 else 1
        
        # Use ensemble for better robustness
        self.ensemble = TransformerEnsemble(n_models=self.n_ensemble)
        result = self.ensemble.fit(X, y, epochs=epochs, lr=lr)
        result["seq_len"] = self.seq_len
        result["n_features"] = n_features
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained ensemble
        
        Args:
            X: Input features
            
        Returns:
            Predictions (n_samples,)
        """
        if self.ensemble is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.ensemble.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty quantification
        
        Returns:
            Dictionary with 'mean' and 'std' predictions
        """
        if self.ensemble is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.ensemble.predict_with_std(X)
    
    def multi_step_predict(self, historical: np.ndarray, steps: int = None) -> np.ndarray:
        """
        Multi-step ahead prediction using recursive forecasting
        
        Args:
            historical: Recent historical data
            steps: Number of steps to predict
            
        Returns:
            Predicted values for next 'steps' time periods
        """
        steps = steps or self.horizon
        predictions = []
        current_data = historical.copy()
        
        for _ in range(steps):
            # Prepare input
            if len(current_data) < self.seq_len:
                pad_len = self.seq_len - len(current_data)
                X_input = np.pad(current_data, (pad_len, 0)).reshape(1, -1, 1)
            else:
                X_input = current_data[-self.seq_len:].reshape(1, -1, 1)
            
            # Predict next step
            pred = self.predict(X_input)
            predictions.append(pred[0])
            
            # Append to current data for next iteration
            current_data = np.append(current_data, pred[0])
        
        return np.array(predictions)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Returns:
            Dictionary with MAE, RMSE, MAPE
        """
        y_pred = self.predict(X_test)
        
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100
        
        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape)
        }


def run_case_study():
    """
    Complete case study with synthetic traffic data
    Demonstrates the full pipeline from data preparation to evaluation
    """
    np.random.seed(42)
    
    # Generate synthetic traffic data (hourly for 30 days = 720 hours)
    t = np.arange(0, 720)
    base_flow = 1000
    daily_pattern = 300 * np.sin(2 * np.pi * (t - 8) / 24)  # Peak at 8 AM
    weekly_pattern = 100 * np.sin(2 * np.pi * t / 168)  # Weekly cycle
    trend = 0.1 * t  # Slight upward trend
    noise = np.random.normal(0, 50, 720)
    
    traffic_flow = np.maximum(base_flow + daily_pattern + weekly_pattern + trend + noise, 0)
    
    # Split data: 80% train, 20% test
    split_idx = int(len(traffic_flow) * 0.8)
    train_data = traffic_flow[:split_idx]
    test_data = traffic_flow[split_idx:]
    
    print("=" * 60)
    print("CUMCM 2025 B - Transformer Traffic Flow Forecasting")
    print("=" * 60)
    
    # Initialize and train model
    model = TrafficFlowTransformer(seq_len=48, horizon=24, n_ensemble=3)
    
    # Prepare data
    X_train, y_train = model.prepare_data(train_data)
    X_test, y_test = model.prepare_data(test_data)
    
    print(f"\nTraining data: {len(train_data)} samples")
    print(f"Sequence length: {model.seq_len}")
    print(f"Training samples: {X_train.shape[0]}")
    
    # Train
    print("\nTraining Transformer Ensemble...")
    train_result = model.train(X_train, y_train, epochs=30, lr=0.001)
    print(f"Training completed: {train_result['status']}")
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest Results:")
    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # Multi-step prediction
    recent_data = train_data[-model.seq_len:]
    forecast = model.multi_step_predict(recent_data, steps=24)
    
    print(f"\n24-hour Forecast (next day):")
    print(f"  Start: {forecast[0]:.1f}")
    print(f"  Peak:  {forecast.max():.1f}")
    print(f"  End:   {forecast[-1]:.1f}")
    
    # Uncertainty quantification
    X_sample = X_test[:5]
    uncertainty = model.predict_with_uncertainty(X_sample)
    print(f"\nUncertainty Analysis (first 5 test samples):")
    print(f"  Mean pred range: [{uncertainty['mean'].min():.1f}, {uncertainty['mean'].max():.1f}]")
    print(f"  Std dev range:   [{uncertainty['std'].min():.2f}, {uncertainty['std'].max():.2f}]")
    
    print("\n" + "=" * 60)
    print("Case study completed successfully!")
    print("=" * 60)
    
    return {
        "metrics": metrics,
        "forecast": forecast,
        "uncertainty": uncertainty
    }


if __name__ == "__main__":
    result = run_case_study()
