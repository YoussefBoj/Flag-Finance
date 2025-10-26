# 🚀 FLAG-Finance  
**Hybrid GNN + LSTM + LLM/RAG System for Financial Fraud Detection and Explanation**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)  
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.6-green.svg)](https://pytorch-geometric.readthedocs.io/)  
[![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20ECS%20%7C%20Neptune-orange.svg)](https://aws.amazon.com/)  
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)  

---

## 📖 Overview
**FLAG-Finance** is a research-grade **fraud detection platform** combining:

- **Graph Neural Networks (GNNs)** → model financial transaction networks  
- **LSTM/Transformers** → capture sequential behavior of accounts  
- **Hybrid Fusion Models** → combine graph + sequence embeddings  
- **Retrieval-Augmented Generation (RAG) with LLMs** → generate **human-readable explanations**  
- **Cloud-native deployment** → Docker + AWS (SageMaker, ECS, Neptune, Bedrock, Kendra)  

It addresses the $42B annual global fraud loss problem by providing an **accurate, interpretable, and scalable AI pipeline**.

---

## ⚙️ Project Structure
FLAG-Finance/
├─ data/ # raw & processed datasets
├─ notebooks/ # Colab-friendly Jupyter notebooks
│ ├─ 01_data_exploration.ipynb
│ ├─ 02_graph_construction_elliptic.ipynb
│ ├─ 03_gnn_baseline_training.ipynb
│ ├─ 04_contrastive_pretraining.ipynb
│ ├─ 05_sequence_lstm_baseline.ipynb
│ ├─ 06_fusion_models.ipynb
│ ├─ 07_evaluation_analysis.ipynb
│ ├─ 08_rag_llm_integration.ipynb
│ └─ 09_deployment_example.ipynb
├─ src/
│ ├─ data/ # ingestion & preprocessing
│ ├─ models/ # GNN, LSTM, Fusion, Contrastive pretraining
│ ├─ training/ # training loops
│ ├─ api/ # FastAPI service for inference
│ └─ rag/ # retriever + LLM integration
├─ infra/ # ECS/K8s manifests, Terraform (optional)
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
└─ README.md

---

## 📊 Datasets
- **[Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)**  
- **[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)**  
- **[PaySim Mobile Money](https://www.kaggle.com/datasets/ealaxi/paysim1)**  

---

## 🛠 Installation

### 🔹 Local
```bash
git clone https://github.com/<your-username>/FLAG-Finance.git
cd FLAG-Finance

python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

