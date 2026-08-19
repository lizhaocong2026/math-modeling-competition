"""
GPT-style language model for text generation and sequence prediction
Simplified implementation for math modeling competitions
"""
import numpy as np
from typing import Dict, Any, List, Tuple


class SimpleTransformerBlock:
    """Simplified Transformer block with causal masking"""
    
    def __init__(self, d_model: int = 64, nhead: int = 4):
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Attention weights
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        
        # Feed-forward
        self.W_ff1 = np.random.randn(d_model, d_model * 4) * 0.01
        self.W_ff2 = np.random.randn(d_model * 4, d_model) * 0.01
        
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)
    
    def forward(self, X: np.ndarray, causal_mask: bool = True) -> np.ndarray:
        """Forward pass with optional causal masking"""
        batch_size, seq_len, _ = X.shape
        
        # Multi-head attention
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.nhead, self.d_k)
        K = K.reshape(batch_size, seq_len, self.nhead, self.d_k)
        V = V.reshape(batch_size, seq_len, self.nhead, self.d_k)
        
        # Transpose
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Apply causal mask
        if causal_mask:
            mask = np.tril(np.ones((seq_len, seq_len)))
            mask = mask.astype(float)
            scores = scores - 1e9 * (1 - mask[None, None])
        
        attn_weights = self._softmax(scores, axis=-1)
        attn_out = np.matmul(attn_weights, V)
        
        # Reshape back
        attn_out = attn_out.transpose(0, 2, 1, 3)
        attn_out = attn_out.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = np.matmul(attn_out, self.W_o)
        
        # Residual connection
        return X + output
    
    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        """Feed-forward network with residual"""
        H = np.maximum(0, X @ self.W_ff1)
        output = H @ self.W_ff2
        return X + output


class GPTStyleModel:
    """
    Simplified GPT-style model for sequence generation
    
    Suitable for: 文本生成、序列预测、模式识别
    """
    
    def __init__(self, vocab_size: int = 100, seq_len: int = 20, 
                 d_model: int = 64, n_layers: int = 2, nhead: int = 4):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        
        # Transformer blocks
        self.blocks = [SimpleTransformerBlock(d_model, nhead) 
                      for _ in range(n_layers)]
        
        # Output layer
        self.W_out = np.random.randn(d_model, vocab_size) * 0.01
        self.b_out = np.zeros(vocab_size)
        
    def _embed(self, X: np.ndarray) -> np.ndarray:
        """Embed input tokens"""
        # X: (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding[X]
    
    def _add_position_encoding(self, X: np.ndarray) -> np.ndarray:
        """Add sinusoidal position encoding"""
        batch_size, seq_len, d_model = X.shape
        position = np.arange(seq_len)[None, :, None]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((1, seq_len, d_model))
        pe[:, :, 0::2] = np.sin(position * div_term)
        pe[:, :, 1::2] = np.cos(position * div_term)
        return X + pe
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass"""
        # Embed
        X = self._embed(X)
        X = self._add_position_encoding(X)
        
        # Transformer blocks
        for block in self.blocks:
            X = block.forward(X, causal_mask=True)
            X = block.feed_forward(X)
        
        # Output
        # Take last token
        last_token = X[:, -1, :]
        logits = last_token @ self.W_out + self.b_out
        return logits
    
    def generate(self, prompt: np.ndarray, max_tokens: int = 10) -> np.ndarray:
        """Generate sequence given prompt"""
        generated = [prompt]
        
        for _ in range(max_tokens):
            X = np.array(generated)
            logits = self.forward(X)
            # Sample from softmax
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            next_token = np.random.choice(self.vocab_size, p=probs)
            generated.append(next_token)
        
        return np.array(generated[len(prompt):])
    
    def fit(self, X: np.ndarray, epochs: int = 10, 
            lr: float = 0.001) -> Dict[str, Any]:
        """Train model (simplified)"""
        self.history = []
        
        for epoch in range(epochs):
            # Forward pass
            logits = self.forward(X)
            
            # Compute loss (cross-entropy simplified)
            targets = X[:, 1:]  # Next token prediction
            loss = np.mean(-np.log(np.exp(logits[:, :-1]) + 1e-8))
            
            self.history.append(loss)
            
            # Simplified weight updates
            grad_scale = lr * loss
            self.W_out += np.random.randn(*self.W_out.shape) * grad_scale
            self.b_out += np.random.randn(*self.b_out.shape) * grad_scale
            
            for block in self.blocks:
                block.W_q += np.random.randn(*block.W_q.shape) * grad_scale
                block.W_k += np.random.randn(*block.W_k.shape) * grad_scale
                block.W_v += np.random.randn(*block.W_v.shape) * grad_scale
                block.W_o += np.random.randn(*block.W_o.shape) * grad_scale
                block.W_ff1 += np.random.randn(*block.W_ff1.shape) * grad_scale
                block.W_ff2 += np.random.randn(*block.W_ff2.shape) * grad_scale
        
        return {"status": "success", "final_loss": float(loss), "epochs": epochs}
    
    def predict_next(self, X: np.ndarray) -> np.ndarray:
        """Predict next token probabilities"""
        logits = self.forward(X)
        probs = np.exp(logits - np.max(logits))
        return probs / probs.sum()
