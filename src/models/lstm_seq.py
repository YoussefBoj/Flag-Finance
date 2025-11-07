"""
LSTM/GRU sequence models for temporal fraud detection
Implements BiLSTM+Attention, ResidualGRU, and LSTM-CNN hybrid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with attention mechanism.
    
    Args:
        input_size: Feature dimension per time step
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        num_classes: Output classes
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x, return_embeddings=False):
        # x: (batch, seq_len, features)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        
        embeddings = context
        
        if return_embeddings:
            return embeddings
        
        # Classification
        out = self.classifier(context)
        
        return out, embeddings


class ResidualGRU(nn.Module):
    """
    GRU with residual connections between layers.
    More efficient than LSTM with similar performance.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # GRU layers
        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
        # Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x, return_embeddings=False):
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, hidden)
        
        # Residual GRU layers
        for i, (gru, ln) in enumerate(zip(self.gru_layers, self.layer_norms)):
            residual = x
            
            x, _ = gru(x)
            x = ln(x)
            x = self.dropout(x)
            
            # Residual connection (after first layer)
            if i > 0:
                x = x + residual
        
        # Take last hidden state
        embeddings = x[:, -1, :]  # (batch, hidden)
        
        if return_embeddings:
            return embeddings
        
        # Classification
        out = self.classifier(embeddings)
        
        return out, embeddings


class LSTMCNNHybrid(nn.Module):
    """
    Hybrid model combining LSTM for temporal patterns and CNN for local features.
    
    Architecture:
    - CNN extracts local patterns from sequences
    - LSTM captures long-term dependencies
    - Concatenate both representations
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # CNN branch (1D convolutions over time)
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x, return_embeddings=False):
        # x: (batch, seq_len, features)
        
        # LSTM branch
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]  # Last hidden state (batch, hidden*2)
        
        # CNN branch
        x_cnn = x.transpose(1, 2)  # (batch, features, seq_len)
        x_cnn = F.relu(self.conv1(x_cnn))
        x_cnn = F.relu(self.conv2(x_cnn))
        cnn_feat = self.pool(x_cnn).squeeze(-1)  # (batch, hidden)
        
        # Concatenate
        embeddings = torch.cat([lstm_feat, cnn_feat], dim=1)
        
        if return_embeddings:
            return embeddings
        
        # Classification
        out = self.fusion(embeddings)
        
        return out, embeddings


def create_lstm_model(
    model_name: str,
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 3,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function for LSTM models.
    
    Args:
        model_name: 'bilstm', 'resgru', or 'lstm_cnn'
        input_size: Feature dimension per time step
        hidden_size: Hidden layer size
        num_layers: Number of recurrent layers
        num_classes: Output classes
        **kwargs: Additional arguments
        
    Returns:
        LSTM model instance
    """
    models = {
        'bilstm': BiLSTMAttention,
        'resgru': ResidualGRU,
        'lstm_cnn': LSTMCNNHybrid
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    model_class = models[model_name.lower()]
    
    return model_class(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == '__main__':
    # Test model creation
    model = create_lstm_model(
        model_name='bilstm',
        input_size=20,
        hidden_size=128,
        num_layers=3
    )
    
    print(f'✅ Model created: {model.__class__.__name__}')
    print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test forward pass
    x = torch.randn(32, 10, 20)  # (batch, seq_len, features)
    
    with torch.no_grad():
        out, emb = model(x)
        print(f'   Output shape: {out.shape}')
        print(f'   Embedding shape: {emb.shape}')
