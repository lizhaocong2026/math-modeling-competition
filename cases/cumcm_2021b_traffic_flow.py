# CUMCM 2021 B - Traffic Flow Prediction
import sys
sys.path.insert(0, '.')
import numpy as np
from algorithms.transformer import SimpleTransformer


class TrafficFlowPredictor:
    def __init__(self, seq_len=48, horizon=24):
        self.seq_len = seq_len
        self.horizon = horizon
        np.random.seed(2021)
        
    def generate_traffic_data(self):
        t = np.arange(0, 720, 1)
        base_flow = 500
        rush_morning = 200 * np.exp(-((t - 18)**2) / 200)
        rush_evening = 250 * np.exp(-((t - 54)**2) / 200)
        weekly_pattern = 50 * np.sin(2 * np.pi * t / 168)
        noise = np.random.normal(0, 30, 720)
        return np.maximum(base_flow + rush_morning + rush_evening + weekly_pattern + noise, 50)
    
    def forecast_transformer(self, data):
        train_data = data[:500]
        test_data = data[500:600]
        
        seq_len = min(self.seq_len, len(train_data))
        X = train_data[-seq_len:].reshape(1, seq_len, 1)
        y = train_data[-seq_len:]
        
        model = SimpleTransformer(d_model=32, nhead=4, num_layers=2, seq_len=seq_len)
        model.fit(X, y, epochs=30)
        
        test_X = test_data[-seq_len:].reshape(1, seq_len, 1)
        predictions = model.predict(test_X)
        
        mape = np.mean(np.abs((test_data - predictions) / (test_data + 1e-8))) * 100
        return {'method': 'Transformer', 'mape': mape, 'predictions': predictions}


if __name__ == '__main__':
    predictor = TrafficFlowPredictor(seq_len=48, horizon=24)
    data = predictor.generate_traffic_data()
    results = predictor.forecast_transformer(data)
    print('CUMCM 2021 B - Traffic Flow Prediction')
    print('Transformer: MAPE = %.2f%%' % results['mape'])
