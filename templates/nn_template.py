"""
神经网络模板 - 模式识别与分类
"""
import numpy as np
from typing import Dict, Any, List, Optional
import sys
sys.path.insert(0, '..')

from algorithms.nn import NeuralNetwork
from utils.data_preprocessor import DataPreprocessor
from visualizations.model_viz import ModelVisualization


class NeuralNetworkTemplate:
    """神经网络模板 - 用于CUMCM图像识别、模式分类题目"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.viz = ModelVisualization()
        
    def build_mlp(
        self,
        input_dim: int,
        hidden_layers: List[int] = None,
        output_dim: int = 1,
        activation: str = "relu",
        learning_rate: float = 0.01
    ) -> NeuralNetwork:
        """
        构建多层感知机(MLP)
        
        参数:
            input_dim: 输入维度
            hidden_layers: 隐藏层神经元数量列表
            output_dim: 输出维度
            activation: 激活函数
            learning_rate: 学习率
            
        返回:
            神经网络实例
        """
        if hidden_layers is None:
            hidden_layers = [64, 32]
        
        layer_sizes = [input_dim] + hidden_layers + [output_dim]
        
        return NeuralNetwork(
            layer_sizes=layer_sizes,
            activation=activation,
            learning_rate=learning_rate
        )
    
    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 500,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        训练并评估神经网络
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            epochs: 训练轮数
            batch_size: 批次大小
            
        返回:
            包含训练历史和评估指标的字典
        """
        # 预处理
        X_train_norm = self.preprocessor.normalize(X_train)
        X_test_norm = self.preprocessor.normalize(X_test)
        
        # 构建模型
        input_dim = X_train.shape[1]
        output_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        model = self.build_mlp(input_dim, [64, 32], output_dim)
        
        # 训练
        history = model.fit(X_train_norm, y_train, epochs=epochs, batch_size=batch_size)
        
        # 预测
        y_pred = model.predict(X_test_norm)
        
        # 计算误差
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        return {
            "training_history": history,
            "predictions": y_pred,
            "mse": float(mse),
            "mae": float(mae),
            "model": model
        }
    
    def plot_training_curve(
        self,
        loss_history: List[float],
        title: str = "训练损失曲线"
    ):
        """绘制训练损失曲线"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
