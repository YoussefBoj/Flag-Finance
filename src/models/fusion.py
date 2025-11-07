"""
Fusion models combining GNN and LSTM embeddings
Implements 4 fusion strategies: Simple, Gated, Attention, CrossModal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFusion(nn.Module):
    """
    Simple concatenation fusion with MLP classifier.
    Baseline fusion approach.
    """
    
    def __init__(
        self,
        gnn_dim: int,
        lstm_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(gnn_dim + lstm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, gnn_emb, lstm_emb):
        # Concatenate embeddings
        x = torch.cat([gnn_emb, lstm_emb], dim=1)
        
        # Classification
        out = self.fusion(x)
        
        return out


class GatedFusion(nn.Module):
    """
    Gated fusion with learnable modality weights.
    Learns to weight GNN vs LSTM importance dynamically.
    """
    
    def __init__(
        self,
        gnn_dim: int,
        lstm_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Projection layers
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        self.lstm_proj = nn.Linear(lstm_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 weights for GNN and LSTM
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, gnn_emb, lstm_emb):
        # Project to same dimension
        gnn_proj = F.relu(self.gnn_proj(gnn_emb))
        lstm_proj = F.relu(self.lstm_proj(lstm_emb))
        
        # Compute gate weights
        concat = torch.cat([gnn_proj, lstm_proj], dim=1)
        weights = self.gate(concat)  # (batch, 2)
        
        # Weighted fusion
        fused = weights[:, 0:1] * gnn_proj + weights[:, 1:2] * lstm_proj
        
        # Classification
        out = self.classifier(fused)
        
        return out


class AttentionFusion(nn.Module):
    """
    Multi-head attention fusion between GNN and LSTM modalities.
    Learns fine-grained cross-modal interactions.
    """
    
    def __init__(
        self,
        gnn_dim: int,
        lstm_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Projection layers
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        self.lstm_proj = nn.Linear(lstm_dim, hidden_dim)
        
        # Multi-head attention
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, gnn_emb, lstm_emb):
        batch_size = gnn_emb.size(0)
        
        # Project embeddings
        gnn_proj = self.gnn_proj(gnn_emb)
        lstm_proj = self.lstm_proj(lstm_emb)
        
        # Stack as sequence: [GNN, LSTM]
        x = torch.stack([gnn_proj, lstm_proj], dim=1)  # (batch, 2, hidden)
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 2, self.hidden_dim)
        
        attn_output = self.out_proj(attn_output)
        
        # Concatenate attended features
        fused = torch.cat([attn_output[:, 0], attn_output[:, 1]], dim=1)
        
        # Classification
        out = self.classifier(fused)
        
        return out


class CrossModalFusion(nn.Module):
    """
    Advanced cross-modal fusion with bilinear pooling and residual connections.
    Captures complex interactions between GNN and LSTM features.
    """
    
    def __init__(
        self,
        gnn_dim: int,
        lstm_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Projection layers
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        self.lstm_proj = nn.Linear(lstm_dim, hidden_dim)
        
        # Bilinear pooling
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        
        # Cross-attention
        self.cross_attn_gnn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.cross_attn_lstm = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, gnn_emb, lstm_emb):
        # Project to same dimension
        gnn_proj = F.relu(self.gnn_proj(gnn_emb))
        lstm_proj = F.relu(self.lstm_proj(lstm_emb))
        
        # Bilinear interaction
        bilinear_feat = self.bilinear(gnn_proj, lstm_proj)
        
        # Cross-attention
        concat = torch.cat([gnn_proj, lstm_proj], dim=1)
        
        gnn_weight = self.cross_attn_gnn(concat)
        lstm_weight = self.cross_attn_lstm(concat)
        
        gnn_attended = gnn_proj * gnn_weight
        lstm_attended = lstm_proj * lstm_weight
        
        # Concatenate all features
        fused = torch.cat([gnn_attended, lstm_attended, bilinear_feat], dim=1)
        
        # Classification
        out = self.fusion(fused)
        
        return out


def create_fusion_model(
    model_name: str,
    gnn_dim: int,
    lstm_dim: int,
    hidden_dim: int = 256,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function for fusion models.
    
    Args:
        model_name: 'simple', 'gated', 'attention', or 'crossmodal'
        gnn_dim: GNN embedding dimension
        lstm_dim: LSTM embedding dimension
        hidden_dim: Hidden layer size
        num_classes: Output classes
        **kwargs: Additional arguments
        
    Returns:
        Fusion model instance
    """
    models = {
        'simple': SimpleFusion,
        'gated': GatedFusion,
        'attention': AttentionFusion,
        'crossmodal': CrossModalFusion
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    model_class = models[model_name.lower()]
    
    return model_class(
        gnn_dim=gnn_dim,
        lstm_dim=lstm_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == '__main__':
    # Test all fusion models
    gnn_emb = torch.randn(32, 384)
    lstm_emb = torch.randn(32, 256)
    
    for model_name in ['simple', 'gated', 'attention', 'crossmodal']:
        model = create_fusion_model(
            model_name=model_name,
            gnn_dim=384,
            lstm_dim=256,
            hidden_dim=256
        )
        
        with torch.no_grad():
            out = model(gnn_emb, lstm_emb)
        
        print(f'✅ {model_name.upper()}: {out.shape} | Params: {sum(p.numel() for p in model.parameters()):,}')
