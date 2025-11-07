# FLAG-Finance Fraud Detection Service

## Quick Start

### Build and Run
```bash
cd C:\Users\youss\Downloads\Flag_finance\e2e_pipeline
docker build -t flag-finance:latest .
docker run --rm -p 8000:8000 flag-finance:latest
```

### Or use Docker Compose
```bash
cd C:\Users\youss\Downloads\Flag_finance\e2e_pipeline
docker-compose up -d
```

### Test the API
```bash
# Health check
curl http://localhost:8000/health

# Prediction (adjust dimensions)
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"gnn_embedding": [/* 320 values */], "lstm_embedding": [/* 256 values */]}'
```

## Model Performance

- Fusion architecture: GatedFusion

- GNN embedding dimension: 320

- LSTM embedding dimension: 256
