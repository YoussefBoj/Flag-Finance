# src/training/train_fusion.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class AccountSeqDataset(Dataset):
    """
    Expects:
      items: list of dicts with keys:
        - node_idx (int): target node index in graph
        - seq (np.array): sequence features [seq_len, feat]
        - label (int)
    """
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        return {"node_idx":it["node_idx"], "seq":torch.tensor(it["seq"], dtype=torch.float), "label":torch.tensor(it["label"], dtype=torch.long)}

def train_epoch(gnn, seq_enc, clf, data_obj, dataset, optimizers, device):
    gnn.train(); seq_enc.train(); clf.train()
    opt_g, opt_s, opt_c = optimizers
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
    losses = []
    for batch in loader:
        node_idx = batch["node_idx"].to(device)
        seq = batch["seq"].to(device)
        label = batch["label"].to(device)
        # compute gnn embeddings for the whole graph (or use stored precomputed embeddings)
        node_embs = gnn(data_obj.x.to(device), data_obj.edge_index.to(device))
        batch_node_emb = node_embs[node_idx]
        seq_emb = seq_enc(seq)
        logits = clf(batch_node_emb, seq_emb)
        loss = F.cross_entropy(logits, label)
        opt_g.zero_grad(); opt_s.zero_grad(); opt_c.zero_grad()
        loss.backward()
        opt_g.step(); opt_s.step(); opt_c.step()
        losses.append(loss.item())
    return sum(losses)/len(losses)
