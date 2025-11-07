"""
RAG retrieval system for fraud case similarity search
Uses FAISS vector database with sentence transformers
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


class FraudCaseRetriever:
    """
    Vector-based retrieval system for fraud cases.
    
    Features:
    - FAISS vector database for fast similarity search
    - Sentence transformer embeddings
    - Case indexing with metadata
    - Top-k retrieval with scores
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        dimension: int = 384,
        index_type: str = 'flat'
    ):
        self.embedding_model_name = embedding_model
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize embedding model
        print(f'📦 Loading embedding model: {embedding_model}...')
        self.encoder = SentenceTransformer(embedding_model)
        
        # Initialize FAISS index
        if index_type == 'flat':
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.cases: List[Dict] = []
        self.case_texts: List[str] = []
        
    def create_case_text(self, case: Dict) -> str:
        """
        Convert fraud case metadata to text description.
        
        Args:
            case: Dictionary with transaction metadata
            
        Returns:
            Text description of the case
        """
        text_parts = []
        
        # Amount
        if 'amount' in case:
            text_parts.append(f"Transaction amount: ${case['amount']:.2f}")
        
        # Transaction type
        if 'type' in case:
            text_parts.append(f"Type: {case['type']}")
        
        # Temporal features
        if 'hour' in case:
            text_parts.append(f"Hour: {case['hour']}")
        if 'is_night' in case and case['is_night']:
            text_parts.append("Occurred at night")
        
        # Balance features
        if 'balance_error_orig' in case:
            text_parts.append(f"Balance discrepancy: ${abs(case['balance_error_orig']):.2f}")
        
        # Graph features
        if 'out_degree' in case:
            text_parts.append(f"Outgoing connections: {case['out_degree']}")
        if 'in_degree' in case:
            text_parts.append(f"Incoming connections: {case['in_degree']}")
        
        # Fraud indicator
        if 'is_fraud' in case:
            text_parts.append(f"Fraud status: {'Fraudulent' if case['is_fraud'] else 'Legitimate'}")
        
        # Pattern description
        if 'pattern' in case:
            text_parts.append(f"Pattern: {case['pattern']}")
        
        return ". ".join(text_parts) + "."
    
    def build_index(
        self,
        cases: List[Dict],
        save_path: Optional[Path] = None
    ):
        """
        Build FAISS index from fraud cases.
        
        Args:
            cases: List of case dictionaries
            save_path: Optional path to save index
        """
        print(f'\n🔨 Building FAISS index from {len(cases)} cases...')
        
        self.cases = cases
        
        # Convert cases to text
        self.case_texts = [self.create_case_text(case) for case in tqdm(cases, desc='Creating case texts')]
        
        # Generate embeddings
        print('   Encoding cases...')
        embeddings = self.encoder.encode(
            self.case_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add to FAISS index
        if self.index_type == 'ivf':
            print('   Training IVF index...')
            self.index.train(embeddings)
        
        print('   Adding vectors to index...')
        self.index.add(embeddings)
        
        print(f'   ✅ Index built with {self.index.ntotal} vectors')
        
        # Save if path provided
        if save_path:
            self.save_index(save_path)
    
    def retrieve(
        self,
        query_case: Dict,
        top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieve similar fraud cases.
        
        Args:
            query_case: Query case dictionary
            top_k: Number of similar cases to retrieve
            
        Returns:
            List of (case, similarity_score) tuples
        """
        # Create query text
        query_text = self.create_case_text(query_case)
        
        # Encode query
        query_emb = self.encoder.encode([query_text], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_emb, top_k)
        
        # Retrieve cases with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.cases):
                similarity_score = 1.0 / (1.0 + dist)  # Convert L2 distance to similarity
                results.append((self.cases[idx], similarity_score))
        
        return results
    
    def save_index(self, save_path: Path):
        """Save FAISS index and metadata."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / 'index.faiss'))
        
        # Save metadata
        metadata = {
            'cases': self.cases,
            'case_texts': self.case_texts,
            'embedding_model': self.embedding_model_name,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
        
        with open(save_path / 'index.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f'   💾 Index saved to {save_path}')
    
    def load_index(self, load_path: Path):
        """Load FAISS index and metadata."""
        load_path = Path(load_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / 'index.faiss'))
        
        # Load metadata
        with open(load_path / 'index.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.cases = metadata['cases']
        self.case_texts = metadata['case_texts']
        
        print(f'   ✅ Loaded index with {self.index.ntotal} vectors')


def build_fraud_case_database(
    data_path: Path,
    dataset: str = 'paysim',
    n_cases: int = 1000,
    save: bool = True
) -> FraudCaseRetriever:
    """
    Build fraud case database from processed data.
    
    Args:
        data_path: Base data directory
        dataset: 'paysim' or 'elliptic'
        n_cases: Number of fraud cases to index
        save: Whether to save the index
        
    Returns:
        FraudCaseRetriever instance
    """
    print(f'\n🏗️  Building fraud case database ({dataset})...')
    
    # Load data
    if dataset == 'paysim':
        df = pd.read_csv(data_path / 'processed' / 'paysim_data_enhanced.csv')
        fraud_df = df[df['isFraud'] == 1].head(n_cases)
        
        cases = []
        for _, row in fraud_df.iterrows():
            case = {
                'amount': float(row['amount']),
                'type': str(row['type']),
                'hour': int(row.get('hour', 0)),
                'is_night': bool(row.get('is_night', False)),
                'balance_error_orig': float(row.get('balance_error_orig', 0)),
                'is_fraud': True,
                'pattern': 'suspicious_transaction'
            }
            cases.append(case)
    
    elif dataset == 'elliptic':
        df = pd.read_csv(data_path / 'processed' / 'elliptic_nodes_enhanced.csv')
        fraud_df = df[df['class'] == 2].head(n_cases)
        
        cases = []
        for _, row in fraud_df.iterrows():
            case = {
                'amount': float(row.get('feat_1', 0)),
                'out_degree': int(row.get('out_degree', 0)),
                'in_degree': int(row.get('in_degree', 0)),
                'time_step': int(row.get('time_step', 0)),
                'is_fraud': True,
                'pattern': 'illicit_bitcoin_transaction'
            }
            cases.append(case)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Build retriever
    retriever = FraudCaseRetriever()
    retriever.build_index(cases)
    
    if save:
        save_path = data_path / 'vector_db' / 'fraud_cases_faiss'
        retriever.save_index(save_path)
    
    return retriever


if __name__ == '__main__':
    from pathlib import Path
    
    base_path = Path(__file__).parent.parent.parent / 'data'
    
    # Build database
    retriever = build_fraud_case_database(
        data_path=base_path,
        dataset='paysim',
        n_cases=500
    )
    
    # Test retrieval
    test_case = {
        'amount': 5000.0,
        'type': 'TRANSFER',
        'hour': 3,
        'is_night': True,
        'balance_error_orig': 500.0
    }
    
    results = retriever.retrieve(test_case, top_k=3)
    
    print(f'\n🔍 Retrieved {len(results)} similar cases:')
    for i, (case, score) in enumerate(results, 1):
        print(f'   {i}. Score: {score:.3f} | Amount: ${case["amount"]:.2f}')
