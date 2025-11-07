
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class CrossModalFusion(nn.Module):
    def __init__(self, gnn_dim, lstm_dim, hidden_dim=256, num_heads=4, num_classes=2, dropout=0.4):
        super().__init__()
        self.gnn_proj = nn.Sequential(nn.Linear(gnn_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.lstm_proj = nn.Sequential(nn.Linear(lstm_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.gnn_to_lstm_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.lstm_to_gnn_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, gnn_emb, lstm_emb):
        gnn_proj = self.gnn_proj(gnn_emb).unsqueeze(1)
        lstm_proj = self.lstm_proj(lstm_emb).unsqueeze(1)
        gnn_attended, _ = self.gnn_to_lstm_attn(gnn_proj, lstm_proj, lstm_proj)
        gnn_attended = self.norm1(gnn_attended.squeeze(1) + gnn_proj.squeeze(1))
        lstm_attended, _ = self.lstm_to_gnn_attn(lstm_proj, gnn_proj, gnn_proj)
        lstm_attended = self.norm2(lstm_attended.squeeze(1) + lstm_proj.squeeze(1))
        fused = torch.cat([gnn_attended, lstm_attended], dim=1)
        return self.classifier(fused)

app = FastAPI(title="FLAG-Finance Fraud Detection API", version="1.0")

class PredictionRequest(BaseModel):
    gnn_embedding: List[float]
    lstm_embedding: List[float]

class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int
    confidence: str

@app.on_event('startup')
def load_model():
    global model, device, meta
    device = torch.device('cpu')
    meta_path = Path('/models/fusion_meta.json')
    checkpoint_path = Path('/models/best.pt')

    if not meta_path.exists() or not checkpoint_path.exists():
        raise RuntimeError('Model artefacts missing inside container.')

    meta = json.loads(meta_path.read_text())
    model = CrossModalFusion(
        gnn_dim=int(meta['GNN_DIM']),
        lstm_dim=int(meta['LSTM_DIM']),
        hidden_dim=256,
        num_heads=4
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print('✅ Fusion model loaded successfully')

@app.get('/')
def root():
    return {
        'service': 'FLAG-Finance Fraud Detection',
        'version': '1.0',
        'model': meta.get('arch', 'CrossModalFusion'),
        'status': 'operational'
    }

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        gnn_emb = torch.tensor([request.gnn_embedding], dtype=torch.float32)
        lstm_emb = torch.tensor([request.lstm_embedding], dtype=torch.float32)

        if gnn_emb.shape[1] != meta['GNN_DIM']:
            raise HTTPException(400, f"GNN embedding must be {meta['GNN_DIM']} dimensions")
        if lstm_emb.shape[1] != meta['LSTM_DIM']:
            raise HTTPException(400, f"LSTM embedding must be {meta['LSTM_DIM']} dimensions")

        with torch.no_grad():
            logits = model(gnn_emb, lstm_emb)
            prob = torch.softmax(logits, dim=1)[0, 1].item()

        prediction = 1 if prob >= 0.5 else 0
        confidence = 'high' if prob >= 0.8 or prob <= 0.2 else ('medium' if prob >= 0.6 or prob <= 0.4 else 'low')

        return PredictionResponse(
            fraud_probability=prob,
            prediction=prediction,
            confidence=confidence
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f'Prediction error: {exc}')

@app.get('/health')
def health():
    return {'status': 'healthy'}
