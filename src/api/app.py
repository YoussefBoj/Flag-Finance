"""
FastAPI production service for FLAG-Finance fraud detection
REST API with real-time predictions and explainability
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import numpy as np

from src.api.predict import FraudDetectionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FLAG-Finance Fraud Detection API",
    description="Advanced fraud detection using Graph Neural Networks, LSTM, and LLM explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[FraudDetectionPipeline] = None


# ============================================================================
# Pydantic Models
# ============================================================================

class TransactionInput(BaseModel):
    """Input schema for single transaction prediction."""
    
    # PaySim features
    amount: Optional[float] = Field(None, description="Transaction amount")
    type: Optional[str] = Field(None, description="Transaction type (TRANSFER, CASH_OUT, etc.)")
    oldbalanceOrg: Optional[float] = Field(None, description="Origin account balance before")
    newbalanceOrig: Optional[float] = Field(None, description="Origin account balance after")
    oldbalanceDest: Optional[float] = Field(None, description="Destination account balance before")
    newbalanceDest: Optional[float] = Field(None, description="Destination account balance after")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of transaction (0-23)")
    
    # Elliptic features (for Bitcoin transactions)
    features: Optional[List[float]] = Field(None, description="Raw feature vector for Elliptic dataset")
    out_degree: Optional[int] = Field(None, description="Number of outgoing transactions")
    in_degree: Optional[int] = Field(None, description="Number of incoming transactions")
    
    # Metadata
    transaction_id: Optional[str] = Field(None, description="Unique transaction identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "amount": 5000.0,
                "type": "TRANSFER",
                "oldbalanceOrg": 15000.0,
                "newbalanceOrig": 10000.0,
                "oldbalanceDest": 2000.0,
                "newbalanceDest": 7000.0,
                "hour": 3,
                "transaction_id": "TX_12345"
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for fraud prediction."""
    
    transaction_id: Optional[str]
    prediction: str = Field(..., description="FRAUD or LEGIT")
    fraud_probability: float = Field(..., ge=0.0, le=100.0, description="Fraud probability (0-100%)")
    confidence: str = Field(..., description="LOW, MEDIUM, or HIGH")
    explanation: Optional[str] = Field(None, description="Natural language explanation")
    similar_cases: Optional[List[Dict]] = Field(None, description="Similar historical fraud cases")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TX_12345",
                "prediction": "FRAUD",
                "fraud_probability": 92.5,
                "confidence": "HIGH",
                "explanation": "This transaction was flagged due to: high fraud probability, unusually large transaction amount, transaction occurred during high-risk hours.",
                "risk_factors": ["Large amount", "Night transaction", "Balance discrepancy"]
            }
        }


class BatchPredictionInput(BaseModel):
    """Input schema for batch prediction."""
    transactions: List[TransactionInput]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    
    logger.info("🚀 Starting FLAG-Finance API...")
    
    try:
        # Get model paths
        base_path = Path(__file__).parent.parent.parent / 'data'
        
        pipeline = FraudDetectionPipeline(
            base_path=base_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            enable_rag=True  # ✅ Enable RAG for LLM explanations
        )
        
        logger.info(f"✅ Pipeline initialized on {pipeline.device}")
        logger.info(f"   GNN model: {pipeline.gnn_model.__class__.__name__}")
        logger.info(f"   LSTM model: {pipeline.lstm_model.__class__.__name__}")
        logger.info(f"   Fusion model: {pipeline.fusion_model.__class__.__name__}")
        logger.info(f"   RAG enabled: {pipeline.explainer is not None}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        pipeline = None


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FLAG-Finance Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "models_info": "/models/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        model_loaded=pipeline is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_transaction(transaction: TransactionInput):
    """
    Predict fraud for a single transaction.
    
    Args:
        transaction: Transaction input data
        
    Returns:
        Prediction with explanation
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dictionary
        transaction_dict = transaction.dict(exclude_none=True)
        
        # Predict
        result = pipeline.predict_single(transaction_dict)
        
        return PredictionOutput(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=List[PredictionOutput])
async def predict_batch(
    batch_input: BatchPredictionInput,
    background_tasks: BackgroundTasks
):
    """
    Predict fraud for batch of transactions.
    
    Args:
        batch_input: List of transactions
        background_tasks: FastAPI background tasks
        
    Returns:
        List of predictions
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        transactions = [t.dict(exclude_none=True) for t in batch_input.transactions]
        
        # Predict
        results = pipeline.predict_batch(transactions)
        
        return [PredictionOutput(**r) for r in results]
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "gnn_model": {
            "type": pipeline.gnn_model.__class__.__name__,
            "parameters": sum(p.numel() for p in pipeline.gnn_model.parameters())
        },
        "lstm_model": {
            "type": pipeline.lstm_model.__class__.__name__,
            "parameters": sum(p.numel() for p in pipeline.lstm_model.parameters())
        },
        "fusion_model": {
            "type": pipeline.fusion_model.__class__.__name__,
            "parameters": sum(p.numel() for p in pipeline.fusion_model.parameters())
        },
        "device": str(pipeline.device),
        "rag_enabled": pipeline.explainer is not None,
        "explainer_backend": getattr(pipeline.explainer, 'llm_backend', 'template') if pipeline.explainer else None
    }


@app.get("/statistics")
async def get_statistics():
    """Get API usage statistics."""
    # In production, implement proper tracking with Redis/DB
    return {
        "total_predictions": 0,
        "fraud_detected": 0,
        "average_response_time_ms": 0,
        "api_version": "1.0.0"
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
