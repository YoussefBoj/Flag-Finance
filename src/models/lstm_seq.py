# src/models/lstm_seq.py
import torch
import torch.nn as nn

class SeqEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
    def forward(self, seq_batch):
        # seq_batch: [B, seq_len, features]
        out, (h_n, c_n) = self.lstm(seq_batch)
        # return last hidden state
        return h_n[-1]  # [B, hidden_dim]
