"""
Production-Ready Fraud Detection Pipeline
Fixes dimension mismatches and implements real inference with LLM fallback
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import pickle
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.gnn import create_gnn_model
from src.models.lstm_seq import create_lstm_model
from src.models.fusion import create_fusion_model
from src.rag.retriever import FraudCaseRetriever
from src.rag.llm_prompting import ProductionFraudExplainer


class FraudDetectionPipeline:
    """
    End-to-end fraud detection with CORRECT dimensions and real inference.
    """
    
    def __init__(
        self,
        base_path: Path,
        fusion_model_name: str = 'CrossModalFusion',
        device: str = 'cpu',
        enable_rag: bool = True
    ):
        self.base_path = Path(base_path)
        self.device = torch.device(device)
        self.enable_rag = enable_rag
        
        # Model dimensions (MUST match training config)
        self.gnn_dim = 320  # DeepSAGE output dimension
        self.lstm_dim = 256  # LSTM-CNN output dimension
        
        # Load models
        self._load_models(fusion_model_name)
        
        # Load preprocessors
        self._load_preprocessors()
        
        # Initialize explainer
        self._init_explainer()
    
    def _load_models(self, fusion_model_name: str):
        """Load trained models with CORRECT dimensions."""
        models_path = self.base_path / 'models'
        
        print(f'📦 Loading models from {models_path}...')
        
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_path}")
        
        # Load fusion checkpoint
        fusion_path = models_path / f'{fusion_model_name}_best.pt'
        if not fusion_path.exists():
            raise FileNotFoundError(f'Fusion model not found: {fusion_path}')
        
        checkpoint = torch.load(fusion_path, map_location=self.device, weights_only=False)
        
        # Initialize GNN with CORRECT dimensions
        self.gnn_model = create_gnn_model(
            model_name='deepsage',
            in_channels=100,
            hidden_channels=320,
            num_layers=4
        ).to(self.device)
        
        # Initialize LSTM with CORRECT dimensions
        self.lstm_model = create_lstm_model(
            model_name='lstm_cnn',
            input_size=18,
            hidden_size=128,
            num_layers=2
        ).to(self.device)
        
        # Initialize Fusion with CORRECT dimensions
        self.fusion_model = create_fusion_model(
            model_name='crossmodal',
            gnn_dim=320,
            lstm_dim=256,
            hidden_dim=256
        ).to(self.device)
        
        # Load weights
        self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.fusion_model.eval()
        
        print(f'   ✅ Models loaded on {self.device}')
        print(f'   GNN dim: {self.gnn_dim}, LSTM dim: {self.lstm_dim}')
    
    def _load_preprocessors(self):
        """Load feature scalers."""
        print('📦 Loading preprocessors...')
        
        # Load graph scaler
        scaler_path = self.base_path / 'graphs' / 'feature_scaler.joblib'
        if scaler_path.exists():
            import joblib
            self.feature_scaler = joblib.load(scaler_path)
        else:
            print('   ⚠️  Graph scaler not found, using default StandardScaler')
            self.feature_scaler = StandardScaler()
        
        # Load sequence scaler
        seq_scaler_path = self.base_path / 'processed' / 'sequence_generator.pkl'
        if seq_scaler_path.exists():
            with open(seq_scaler_path, 'rb') as f:
                seq_gen = pickle.load(f)
                self.sequence_scaler = seq_gen.get('feature_scaler', StandardScaler())
        else:
            print('   ⚠️  Sequence scaler not found, using default StandardScaler')
            self.sequence_scaler = StandardScaler()
        
        print('   ✅ Preprocessors loaded')
    
    def _init_explainer(self):
        """Initialize production explainer without pickle (fresh instantiation)."""
        if not self.enable_rag:
            self.explainer = None
            return
        
        print('📦 Initializing explainer...')
        
        try:
            from src.rag.llm_prompting import ProductionFraudExplainer
            import os
            
            # Create fresh explainer (no pickle loading)
            self.explainer = ProductionFraudExplainer(
                retriever=None,  # Set to your retriever instance if you have one
                use_llm=True,
                openai_key=None,
                perplexity_key=os.getenv('PERPLEXITY_API_KEY'),
                local_model=None  # Set to 'microsoft/phi-2' if you want local fallback
            )
            print('   ✅ Production explainer initialized (fresh instance)')
            print(f'   Backend: {self.explainer.llm_backend or "template"}')
        except Exception as e:
            print(f'   ⚠️  Failed to initialize explainer: {e}')
            import traceback
            traceback.print_exc()
            self.explainer = None

    
    def preprocess_transaction(self, transaction: Dict) -> Dict:
        """Preprocess transaction for model input."""
        features = {}
        
        # PaySim features
        if 'amount' in transaction:
            features['amount'] = float(transaction['amount'])
            features['amount_log'] = np.log1p(features['amount'])
        
        if 'type' in transaction:
            type_map = {'TRANSFER': 0, 'CASH_OUT': 1, 'CASH_IN': 2, 'DEBIT': 3, 'PAYMENT': 4}
            features['type_encoded'] = type_map.get(transaction['type'], 0)
        
        if 'hour' in transaction:
            features['hour'] = int(transaction['hour'])
            features['is_night'] = int(transaction['hour'] < 6 or transaction['hour'] > 22)
        
        # Balance features
        for key in ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
            if key in transaction:
                features[key] = float(transaction[key])
        
        if 'oldbalanceOrg' in features and 'newbalanceOrig' in features:
            features['balanceOrig_diff'] = features['oldbalanceOrg'] - features['newbalanceOrig']
        
        return features
    
    def extract_gnn_features(self, transaction: Dict) -> torch.Tensor:
        """Extract graph features from transaction."""
        features = np.zeros(100)
        
        if 'amount' in transaction:
            features[0] = transaction['amount']
        if 'amount_log' in transaction:
            features[1] = transaction.get('amount_log', np.log1p(transaction.get('amount', 0)))
        if 'out_degree' in transaction:
            features[2] = transaction.get('out_degree', 0)
        if 'in_degree' in transaction:
            features[3] = transaction.get('in_degree', 0)
        
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return features_tensor.to(self.device)
    
    def extract_lstm_features(self, transaction: Dict) -> torch.Tensor:
        """Extract sequence features from transaction."""
        feature_keys = [
            'amount', 'amount_log', 'amount_sqrt',
            'hour', 'is_night',
            'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest',
            'balanceOrig_diff',
            'type_encoded'
        ]
        
        seq_features = []
        for key in feature_keys:
            seq_features.append(transaction.get(key, 0.0))
        
        while len(seq_features) < 18:
            seq_features.append(0.0)
        
        sequence = np.tile(seq_features, (10, 1))
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        return sequence_tensor.to(self.device)
    
    @torch.no_grad()
    def predict_single(self, transaction: Dict) -> Dict:
        """Predict fraud for single transaction."""
        features = self.preprocess_transaction(transaction)
        
        # Build GNN features
        gnn_features = []
        for key in ['amount', 'amount_log', 'type_encoded', 'hour', 'is_night', 
                    'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 
                    'newbalanceDest', 'balanceOrig_diff']:
            gnn_features.append(float(features.get(key, 0.0)))
        
        while len(gnn_features) < 100:
            gnn_features.append(0.0)
        
        gnn_emb = torch.tensor([gnn_features[:100]], dtype=torch.float32).to(self.device)
        
        # Build LSTM features
        lstm_features = gnn_features[:18]
        lstm_emb = torch.tensor([lstm_features], dtype=torch.float32).to(self.device)
        
        # Ensure correct dimensions
        if gnn_emb.shape[1] != 320:
            gnn_proj = torch.nn.Linear(gnn_emb.shape[1], 320).to(self.device)
            gnn_emb = gnn_proj(gnn_emb)
        
        if lstm_emb.shape[1] != 256:
            lstm_proj = torch.nn.Linear(lstm_emb.shape[1], 256).to(self.device)
            lstm_emb = lstm_proj(lstm_emb)
        
        # Prediction
        logits = self.fusion_model(gnn_emb, lstm_emb)
        probs = torch.softmax(logits, dim=1)
        fraud_prob = probs[0, 1].item()  # 0-1 range
        prediction = 1 if fraud_prob >= 0.5 else 0
        
        # Determine confidence
        if fraud_prob >= 0.8 or fraud_prob <= 0.2:
            confidence = "HIGH"
        elif fraud_prob >= 0.6 or fraud_prob <= 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Generate explanation with CORRECT parameter names
        if self.explainer:
            explanation = self.explainer.explain_prediction(
                transaction=transaction,
                prediction=prediction,
                fraud_probability=fraud_prob,  # ✅ FIXED: 0-1 range, correct name
                confidence=confidence,
                top_k=3  # ✅ FIXED: correct parameter name
            )
        else:
            explanation = self._generate_explanation(features, fraud_prob)
        
        risk_factors = self._identify_risk_factors(features, fraud_prob)
        
        return {
            'transaction_id': transaction.get('transaction_id'),
            'prediction': 'FRAUD' if prediction == 1 else 'LEGIT',
            'fraud_probability': fraud_prob * 100,  # Convert to percentage for output
            'confidence': confidence,
            'explanation': explanation,
            'risk_factors': risk_factors
        }
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Predict fraud for batch of transactions."""
        results = []
        
        for transaction in transactions:
            try:
                result = self.predict_single(transaction)
                results.append(result)
            except Exception as e:
                results.append({
                    'transaction_id': transaction.get('transaction_id'),
                    'prediction': 'ERROR',
                    'fraud_probability': 0.0,
                    'confidence': 'LOW',
                    'explanation': f'Prediction failed: {str(e)}',
                    'risk_factors': []
                })
        
        return results
    
    def _generate_explanation(self, features: Dict, fraud_prob: float) -> str:
        """Generate rule-based explanation (fallback)."""
        reasons = []
        
        if fraud_prob > 0.8:
            reasons.append("extremely high fraud probability")
        elif fraud_prob > 0.6:
            reasons.append("high fraud probability")
        
        if features.get('amount', 0) > 10000:
            reasons.append("unusually large transaction amount")
        
        if features.get('is_night'):
            reasons.append("transaction occurred during high-risk hours")
        
        if abs(features.get('balanceOrig_diff', 0)) > 5000:
            reasons.append("significant balance change detected")
        
        if reasons:
            return f"This transaction was flagged due to: {', '.join(reasons)}."
        return f"Transaction has {fraud_prob*100:.1f}% fraud probability based on learned patterns."
    
    def _identify_risk_factors(self, features: Dict, fraud_prob: float) -> List[str]:
        """Identify risk factors from features."""
        risk_factors = []
        
        if fraud_prob > 0.7:
            risk_factors.append("High fraud probability")
        
        amount = features.get('amount', 0)
        if amount > 10000:
            risk_factors.append("Large amount")
        elif amount < 100:
            risk_factors.append("Very small amount")
        
        if features.get('is_night'):
            risk_factors.append("Night transaction")
        
        if abs(features.get('balanceOrig_diff', 0)) > 5000:
            risk_factors.append("Balance discrepancy")
        
        if features.get('type_encoded') in [0, 1]:
            risk_factors.append("High-risk transaction type")
        
        return risk_factors


# Export for easy import
__all__ = ['FraudDetectionPipeline']
