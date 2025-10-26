# src/models/fusion.py
import torch.nn as nn
import torch.nn.functional as F

class FusionClassifier(nn.Module):
    def __init__(self, gnn_dim, seq_dim, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(gnn_dim + seq_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)
    def forward(self, gnn_emb, seq_emb):
        x = torch.cat([gnn_emb, seq_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        logits = self.fc2(x)
        return logits
