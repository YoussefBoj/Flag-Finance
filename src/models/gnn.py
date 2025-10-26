# src/models/gnn.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, conv="sage", num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        if conv == "sage":
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers-2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, out_dim))
        elif conv == "gat":
            self.convs.append(GATConv(in_dim, hidden_dim, heads=4))
            for _ in range(num_layers-2):
                self.convs.append(GATConv(hidden_dim*4, hidden_dim, heads=4))
            self.convs.append(GATConv(hidden_dim*4, out_dim, heads=1))
        elif conv == "gcn":
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))
        else:
            raise ValueError("Unknown conv type")
    def forward(self, x, edge_index):
        for i,conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x  # node embeddings
