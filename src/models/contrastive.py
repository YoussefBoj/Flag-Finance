# src/models/contrastive.py
import torch
import torch.nn.functional as F
from torch import nn
from src.models.gnn import GNNEncoder

class GraphContrastiveLearner:
    def __init__(self, encoder: GNNEncoder, optimizer, device="cpu"):
        self.encoder = encoder
        self.opt = optimizer
        self.device = device
    def augment(self, x, edge_index, drop_node_p=0.1):
        x_aug = x.clone()
        mask = (torch.rand(x.shape[0]) > drop_node_p).float().unsqueeze(1).to(x.device)
        return x_aug * mask
    def info_nce(self, z1, z2, temp=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.matmul(z1, z2.T) / temp
        labels = torch.arange(z1.size(0)).to(z1.device)
        loss = F.cross_entropy(sim, labels)
        return loss
    def train_step(self, data):
        self.encoder.train()
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        x1 = self.augment(x, edge_index)
        x2 = self.augment(x, edge_index)
        z1 = self.encoder(x1, edge_index)
        z2 = self.encoder(x2, edge_index)
        loss = self.info_nce(z1, z2)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()
