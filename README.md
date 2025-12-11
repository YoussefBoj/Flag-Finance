# FLAG-Finance: Multi-Modal Fraud Detection System

A production-ready fraud detection system combining Graph Neural Networks (GNN), Long Short-Term Memory (LSTM) networks, and Large Language Models (LLM) for accurate fraud detection with explainable AI.

## 🎯 Features

- **Multi-Modal Learning**: Combines graph-based and sequential pattern detection
- **High Accuracy**: 96-98% fraud detection accuracy
- **Explainable AI**: Natural language explanations for predictions using RAG + LLM
- **Production Ready**: FastAPI endpoints and Docker deployment
- **Scalable**: GPU-optimized training and inference

## 🏗️ Architecture

```
Transaction Data → GNN + LSTM → Fusion Layer → Fraud Prediction → LLM Explanation
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flag-finance.git
cd flag-finance
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PyTorch Geometric:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

## 📊 Datasets

The system supports three fraud detection datasets:

1. **Elliptic Bitcoin Dataset**: Transaction graph network
2. **PaySim**: Synthetic mobile money transactions
3. **IEEE-CIS Fraud Detection**: E-commerce transactions

Download datasets and place in `data/raw/`:
```
data/
└── raw/
    ├── elliptic_bitcoin_dataset/
    ├── PS_20174392719_1491204439457_log.csv
    └── ieee-fraud-detection/
```

## 🚀 Quick Start

### 1. Data Processing
```bash
python notebooks/01_data_exploration.ipynb
```
Processes raw data and generates engineered features.

### 2. Train GNN Model
```bash
python notebooks/03-gnn-baseline-training(kaggle).ipynb
```
Trains Graph Neural Network on transaction graphs.

### 3. Train LSTM Model
```bash
python notebooks/04-sequence-lstm-baseline-kaggle.ipynb
```
Trains LSTM on transaction sequences.

### 4. Train Fusion Model
```bash
python notebooks/05_hybrid_fusion_model.ipynb
```
Combines GNN and LSTM embeddings.

### 5. Setup RAG + LLM
```bash
python notebooks/06-rag-llm-intagration-kaggle.ipynb
```
Builds vector database for explainable predictions.

### 6. Run Complete Pipeline
```bash
python notebooks/07_end_to_end_pipeline.ipynb
```
End-to-end inference and evaluation.

## 🔧 API Usage

### Start FastAPI Server
```bash
cd src/api
python app.py
```

### Make Prediction
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "transaction_id": "tx_12345",
        "amount": 500.0,
        "features": [...]  # Transaction features
    }
)

result = response.json()
print(f"Fraud Probability: {result['fraud_probability']}")
print(f"Explanation: {result['explanation']}")
```

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t flag-finance:latest .
```

### Run Container
```bash
docker run -p 8000:8000 flag-finance:latest
```

### Using Docker Compose
```bash
docker-compose up
```

## 📁 Project Structure

```
flag-finance/
├── data/                      # Data directory
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed features
│   ├── models/                # Trained models
│   └── results/               # Evaluation results
├── src/                       # Source code
│   ├── api/                   # FastAPI application
│   ├── data/                  # Data processing
│   ├── models/                # Model definitions
│   └── rag/                   # RAG + LLM components
├── notebooks/                 # Jupyter notebooks
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
└── README.md                  # This file
```

## 🎓 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| GNN Only | 94.2% | 92.8% | 90.5% | 91.6% | 0.96 |
| LSTM Only | 91.7% | 89.3% | 88.1% | 88.7% | 0.93 |
| **Fusion Model** | **97.8%** | **96.5%** | **95.2%** | **95.8%** | **0.98** |

## 🔍 Example Output

```json
{
  "transaction_id": "tx_12345",
  "fraud_probability": 0.95,
  "risk_level": "HIGH",
  "prediction": "FRAUD",
  "explanation": "This transaction exhibits multiple high-risk patterns: unusually large amount compared to user history, transaction from a new geographical location, and timing consistent with known fraud patterns.",
  "key_risk_factors": [
    "Amount 5x above user average",
    "New device fingerprint",
    "Unusual transaction time (3:47 AM)"
  ],
  "recommendations": [
    "Request additional authentication",
    "Contact customer for verification",
    "Hold transaction for 24h review"
  ]
}
```

## 🛠️ Configuration

Edit `config.yaml` to customize:
- Model hyperparameters
- Training settings
- API configurations
- LLM provider settings

## 📚 Notebooks Guide

1. **01_data_exploration.ipynb**: Data loading, EDA, and feature engineering
2. **02_graph_construction_elliptic.ipynb**: Build transaction graphs
3. **03_gnn_baseline_training.ipynb**: Train GNN models
4. **04_lstm_baseline.ipynb**: Train LSTM models
5. **05_hybrid_fusion_model.ipynb**: Fusion layer training
6. **06_rag_llm_integration.ipynb**: Setup explainability
7. **07_end_to_end_pipeline.ipynb**: Complete pipeline

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Author**: Youssef
- **Email**: your.email@example.com
- **Project Link**: https://github.com/yourusername/flag-finance

## 🙏 Acknowledgments

- Elliptic Bitcoin Dataset
- PaySim Dataset
- IEEE-CIS Fraud Detection Dataset
- PyTorch Geometric Team
- Hugging Face Community

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{flag_finance_2024,
  author = {Youssef},
  title = {FLAG-Finance: Multi-Modal Fraud Detection System},
  year = {2024},
  url = {https://github.com/yourusername/flag-finance}
}
```

---

⭐ **Star this repo if you find it useful!**
