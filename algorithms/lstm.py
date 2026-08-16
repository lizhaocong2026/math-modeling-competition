"""
LSTM (长短期记忆网络) - 纯NumPy实现
用于时间序列预测和序列建模
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class LSTM:
    """长短期记忆网络实现"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 learning_rate: float = 0.01, verbose: bool = False):
        """
        初始化LSTM
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            learning_rate: 学习率
            verbose: 是否打印训练信息
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        # 初始化权重
        scale = 1.0 / np.sqrt(hidden_size)
        self.W_f = np.random.randn(hidden_size, hidden_size + input_size) * scale
        self.W_i = np.random.randn(hidden_size, hidden_size + input_size) * scale
        self.W_o = np.random.randn(hidden_size, hidden_size + input_size) * scale
        self.W_c = np.random.randn(hidden_size, hidden_size + input_size) * scale
        
        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))
        
        # 输出层权重
        self.W_y = np.random.randn(output_size, hidden_size) * scale
        self.b_y = np.zeros((output_size, 1))
        
        self.loss_history = []
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """前向传播"""
        T = X.shape[0]
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        gates = {'forget': [], 'input': [], 'output': [], 'cell': [],
                 'h': [h.copy()], 'c': [c.copy()]}
        
        for t in range(T):
            x_t = X[t].reshape(-1, 1)
            concat = np.vstack((h, x_t))
            
            f_t = self._sigmoid(self.W_f @ concat + self.b_f)
            i_t = self._sigmoid(self.W_i @ concat + self.b_i)
            o_t = self._sigmoid(self.W_o @ concat + self.b_o)
            c_tilde = self._tanh(self.W_c @ concat + self.b_c)
            
            c = f_t * c + i_t * c_tilde
            h = o_t * self._tanh(c)
            
            gates['forget'].append(f_t)
            gates['input'].append(i_t)
            gates['output'].append(o_t)
            gates['cell'].append(c_tilde)
            gates['h'].append(h.copy())
            gates['c'].append(c.copy())
        
        y = self.W_y @ h + self.b_y
        return {'y': y, 'h': h, 'c': c, 'gates': gates}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        result = self.forward(X)
        return result['y'].flatten()
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            verbose: Optional[bool] = None) -> Dict[str, List]:
        """训练LSTM (简化版使用数值梯度)"""
        if verbose is None:
            verbose = self.verbose
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            result = self.forward(X)
            y_pred = result['y']
            loss = np.mean((y_pred - y) ** 2)
            history['loss'].append(loss)
            
            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}: loss = {loss:.6f}")
            
            self._backprop(X, y, result)
        
        return history
    
    def _backprop(self, X: np.ndarray, y: np.ndarray, result: Dict):
        """反向传播 (简化实现)"""
        gates = result['gates']
        h = result['h']
        c = result['c']
        y_pred = result['y']
        
        dy = 2 * (y_pred - y) / y.shape[1]
        dW_y = dy @ h.T
        db_y = dy
        
        dh = dy @ self.W_y
        dc = dh * (1 - self._tanh(c) ** 2) * gates['output'][-1]
        
        self.W_y -= self.learning_rate * dW_y
        self.b_y -= self.learning_rate * db_y
    
    def predict_sequence(self, X: np.ndarray, steps: int = 10) -> np.ndarray:
        """多步预测"""
        predictions = []
        current_input = X.copy()
        
        for _ in range(steps):
            pred = self.predict(current_input)
            predictions.append(pred)
            current_input = np.roll(current_input, -1, axis=0)
            current_input[-1] = pred
        
        return np.array(predictions)


class GRU(LSTM):
    """GRU (门控循环单元) - LSTM的简化版本"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.01, verbose: bool = False):
        super().__init__(input_size, hidden_size, output_size, learning_rate, verbose)
        
        scale = 1.0 / np.sqrt(hidden_size)
        self.W_z = np.random.randn(hidden_size, hidden_size + input_size) * scale
        self.W_r = np.random.randn(hidden_size, hidden_size + input_size) * scale
        self.W_h = np.random.randn(hidden_size, hidden_size + input_size) * scale
        
        self.b_z = np.zeros((hidden_size, 1))
        self.b_r = np.zeros((hidden_size, 1))
        self.b_h = np.zeros((hidden_size, 1))
    
    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """GRU前向传播"""
        T = X.shape[0]
        h = np.zeros((self.hidden_size, 1))
        
        gates = {'update': [], 'reset': [], 'h': [h.copy()]}
        
        for t in range(T):
            x_t = X[t].reshape(-1, 1)
            concat = np.vstack((h, x_t))
            
            z_t = self._sigmoid(self.W_z @ concat + self.b_z)
            r_t = self._sigmoid(self.W_r @ concat + self.b_r)
            
            concat_r = np.vstack((r_t * h, x_t))
            h_tilde = self._tanh(self.W_h @ concat_r + self.b_h)
            
            h = (1 - z_t) * h + z_t * h_tilde
            
            gates['update'].append(z_t)
            gates['reset'].append(r_t)
            gates['h'].append(h.copy())
        
        y = self.W_y @ h + self.b_y
        return {'y': y, 'h': h, 'gates': gates}


if __name__ == "__main__":
    np.random.seed(42)
    
    T = 100
    X = np.linspace(0, 4*np.pi, T).reshape(-1, 1)
    y = np.sin(X).flatten()
    
    lstm = LSTM(input_size=1, hidden_size=16, output_size=1, learning_rate=0.01)
    history = lstm.fit(X, y, epochs=200, verbose=True)
    
    predictions = lstm.predict(X)
    print(f"Final loss: {history['loss'][-1]:.6f}")
    
    future = lstm.predict_sequence(X[-10:], steps=20)
    print(f"Future predictions shape: {future.shape}")
