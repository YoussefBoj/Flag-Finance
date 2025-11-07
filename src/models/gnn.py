"""
Graph Neural Network architectures for fraud detection
Implements GraphSAGE, GAT, and Hybrid models with deep architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm, global_mean_pool
from torch_geometric.data import Data


class GraphSAGE(nn.Module):
    """
    GraphSAGE model with multiple layers and residual connections.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        num_layers: Number of GraphSAGE layers
        dropout: Dropout probability
        num_classes: Output classes (2 for binary)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Output head
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)
        
    def forward(self, x, edge_index, return_embeddings=False):
        # Input projection
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GraphSAGE layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:
                x = x + x_residual
        
        # Store embeddings
        embeddings = x
        
        if return_embeddings:
            return embeddings
        
        # Classification head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x, embeddings


class DeepSAGE(nn.Module):
    """
    Very deep GraphSAGE with 6-8 layers, skip connections, and LayerNorm.
    Optimized for large graphs with complex patterns.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 384,
        num_layers: int = 6,
        dropout: float = 0.4,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_bn = BatchNorm(hidden_channels)
        
        # Deep SAGE layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Skip connection projections (every 2 layers)
        self.skip_projs = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(num_layers // 2)
        ])
        
        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, num_classes)
        )
        
    def forward(self, x, edge_index, return_embeddings=False):
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Deep layers with skip connections
        skip_idx = 0
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            identity = x
            
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Skip connection every 2 layers
            if (i + 1) % 2 == 0 and skip_idx < len(self.skip_projs):
                x = x + self.skip_projs[skip_idx](identity)
                skip_idx += 1
        
        embeddings = x
        
        if return_embeddings:
            return embeddings
        
        # Classification
        out = self.mlp(x)
        
        return out, embeddings


class HybridGNN(nn.Module):
    """
    Hybrid GNN combining GraphSAGE and GAT layers.
    Uses attention for important patterns, aggregation for neighborhoods.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 320,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.35,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Hybrid layers (alternate SAGE and GAT)
        self.sage_convs = nn.ModuleList()
        self.gat_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            if i % 2 == 0:
                self.sage_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.gat_convs.append(
                    GATConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True)
                )
            self.bns.append(BatchNorm(hidden_channels))
        
        # Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
    def forward(self, x, edge_index, return_embeddings=False):
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        sage_idx = 0
        gat_idx = 0
        
        for i, bn in enumerate(self.bns):
            if i % 2 == 0:
                x = self.sage_convs[sage_idx](x, edge_index)
                sage_idx += 1
            else:
                x = self.gat_convs[gat_idx](x, edge_index)
                gat_idx += 1
            
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        embeddings = x
        
        if return_embeddings:
            return embeddings
        
        out = self.classifier(x)
        
        return out, embeddings


def create_gnn_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int = 256,
    num_layers: int = 3,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating GNN models.
    
    Args:
        model_name: 'graphsage', 'deepsage', or 'hybrid'
        in_channels: Input feature dimension
        hidden_channels: Hidden layer size
        num_layers: Number of GNN layers
        num_classes: Output classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        GNN model instance
    """
    models = {
        'graphsage': GraphSAGE,
        'deepsage': DeepSAGE,
        'hybrid': HybridGNN
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    model_class = models[model_name.lower()]
    
    return model_class(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == '__main__':
    # Test model creation
    model = create_gnn_model(
        model_name='deepsage',
        in_channels=100,
        hidden_channels=384,
        num_layers=6
    )
    
    print(f'✅ Model created: {model.__class__.__name__}')
    print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test forward pass
    x = torch.randn(1000, 100)
    edge_index = torch.randint(0, 1000, (2, 5000))
    
    with torch.no_grad():
        out, emb = model(x, edge_index)
        print(f'   Output shape: {out.shape}')
        print(f'   Embedding shape: {emb.shape}')
