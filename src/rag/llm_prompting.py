"""
LLM-based fraud explanation generation using RAG
Supports multiple LLM backends: OpenAI, HuggingFace, Anthropic
"""

from pathlib import Path
from typing import Dict, List, Optional
import os
import json

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import HuggingFaceHub

from src.rag.retriever import FraudCaseRetriever


class FraudExplainer:
    """
    Generate natural language explanations for fraud predictions using RAG.
    """
    
    def __init__(
        self,
        retriever: FraudCaseRetriever,
        llm_provider: str = 'openai',
        model_name: Optional[str] = None,
        temperature: float = 0.3
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = self._init_llm(llm_provider, model_name)
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create chain using LCEL (modern approach)
        if self.llm is not None:
            self.chain = (
                self.prompt_template 
                | self.llm 
                | StrOutputParser()
            )
        else:
            self.chain = None
    
    def _init_llm(self, provider: str, model_name: Optional[str]):
        """Initialize LLM based on provider."""
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print('⚠️  No OpenAI API key found, using fallback template')
                return None
            
            return ChatOpenAI(
                model=model_name or 'gpt-3.5-turbo',
                temperature=self.temperature,
                api_key=api_key
            )
        
        elif provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print('⚠️  No Anthropic API key found, using fallback template')
                return None
            
            return ChatAnthropic(
                model=model_name or 'claude-3-sonnet-20240229',
                temperature=self.temperature,
                api_key=api_key
            )
        
        elif provider == 'huggingface':
            api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
            if not api_key:
                print('⚠️  No HuggingFace API key found, using fallback template')
                return None
            
            return HuggingFaceHub(
                repo_id=model_name or 'mistralai/Mistral-7B-Instruct-v0.2',
                huggingfacehub_api_token=api_key,
                model_kwargs={'temperature': self.temperature, 'max_length': 512}
            )
        
        else:
            print(f'⚠️  Unknown provider: {provider}, using fallback template')
            return None
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for fraud explanation."""
        template = """You are an expert fraud detection analyst. A transaction has been flagged as potentially fraudulent by our AI system.

**Transaction Details:**
{transaction_details}

**Model Prediction:**
- Fraud Probability: {fraud_probability}%
- Prediction: {prediction}

**Similar Historical Cases:**
{similar_cases}

**Task:**
Provide a clear, professional explanation for why this transaction was flagged as fraud. Your explanation should:
1. Reference specific transaction characteristics
2. Compare to similar historical fraud cases
3. Highlight the most suspicious patterns
4. Be concise (2-3 sentences)

**Explanation:**"""

        return PromptTemplate(
            input_variables=['transaction_details', 'fraud_probability', 'prediction', 'similar_cases'],
            template=template
        )
    
    def explain_prediction(
        self,
        transaction: Dict,
        prediction: int,
        fraud_probability: float,  
        confidence: Optional[str] = None,
        top_k: int = 3
    ) -> str:
        """Generate explanation for fraud prediction."""
        
        # Determine confidence if not provided
        if confidence is None:
            if fraud_probability >= 0.8 or fraud_probability <= 0.2:
                confidence = "HIGH"
            elif fraud_probability >= 0.6 or fraud_probability <= 0.4:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        
        # Build basic explanation
        tx_type = transaction.get('type', 'UNKNOWN')
        amount = transaction.get('amount', 0.0)
        
        if prediction == 1:
            # Fraud explanation
            signals = []
            if abs(transaction.get('balance_error_orig', 0)) > 0:
                signals.append("origin balance mismatch")
            if abs(transaction.get('balance_error_dest', 0)) > 0:
                signals.append("destination balance mismatch")
            if transaction.get('isFlaggedFraud', 0) == 1:
                signals.append("system flag")
            if transaction.get('is_night', False):
                signals.append("late-night transaction")
            
            if not signals:
                signals.append("pattern anomalies detected by ML models")
            
            explanation = (
                f"This {tx_type} transaction of ${amount:,.2f} is classified as FRAUD "
                f"with {confidence} confidence ({fraud_probability*100:.1f}% probability). "
                f"Key risk indicators: {', '.join(signals)}."
            )
        else:
            # Legitimate explanation
            explanation = (
                f"This {tx_type} transaction of ${amount:,.2f} appears LEGITIMATE "
                f"with {confidence} confidence ({(1-fraud_probability)*100:.1f}% probability). "
                f"Transaction patterns align with normal behavior."
            )
        
        # Add similar cases if retriever available
        if self.retriever:
            try:
                similar_cases = self.retriever.retrieve(transaction, top_k=top_k)
                if similar_cases:
                    explanation += f"\n\nFound {len(similar_cases)} similar historical fraud cases."
            except Exception as e:
                print(f"Retrieval failed: {e}")
        
        return explanation
        """
        Generate explanation for a fraud prediction.
        
        Args:
            transaction: Transaction metadata dictionary
            fraud_probability: Model's fraud probability (0-100)
            prediction: 'FRAUD' or 'LEGIT'
            top_k: Number of similar cases to retrieve
            
        Returns:
            Dictionary with explanation and metadata
        """
        # Retrieve similar cases
        similar_cases = self.retriever.retrieve(transaction, top_k=top_k)
        
        # Format transaction details
        transaction_details = self._format_transaction(transaction)
        
        # Format similar cases
        similar_cases_text = self._format_similar_cases(similar_cases)
        
        # Generate explanation
        if self.chain is not None:
            try:
                explanation = self.chain.invoke({
                    "transaction_details": transaction_details,
                    "fraud_probability": f"{fraud_probability:.2f}",
                    "prediction": prediction,
                    "similar_cases": similar_cases_text
                })
            except Exception as e:
                print(f'⚠️  LLM error: {e}')
                explanation = self._generate_template_explanation(
                    transaction, fraud_probability, similar_cases
                )
        else:
            # Fallback to template-based explanation
            explanation = self._generate_template_explanation(
                transaction, fraud_probability, similar_cases
            )
        
        return {
            'explanation': explanation.strip(),
            'fraud_probability': fraud_probability,
            'prediction': prediction,
            'similar_cases': [
                {'case': case, 'similarity': score}
                for case, score in similar_cases
            ],
            'transaction': transaction
        }
    
    def _format_transaction(self, transaction: Dict) -> str:
        """Format transaction details as text."""
        parts = []
        
        if 'amount' in transaction:
            parts.append(f"Amount: ${transaction['amount']:.2f}")
        if 'type' in transaction:
            parts.append(f"Type: {transaction['type']}")
        if 'hour' in transaction:
            parts.append(f"Time: {transaction['hour']}:00")
        if 'out_degree' in transaction:
            parts.append(f"Network connections: {transaction['out_degree']}")
        
        return ", ".join(parts)
    
    def _format_similar_cases(self, similar_cases: List) -> str:
        """Format similar cases as text."""
        if not similar_cases:
            return "No similar cases found."
        
        texts = []
        for i, (case, score) in enumerate(similar_cases, 1):
            case_text = f"Case {i} (similarity: {score:.2f}): "
            if 'amount' in case:
                case_text += f"${case['amount']:.2f} transaction"
            if 'type' in case:
                case_text += f", {case['type']}"
            if 'pattern' in case:
                case_text += f" - {case['pattern']}"
            texts.append(case_text)
        
        return "\n".join(texts)
    
    def _generate_template_explanation(
        self,
        transaction: Dict,
        fraud_probability: float,
        similar_cases: List
    ) -> str:
        """Generate rule-based explanation when LLM is unavailable."""
        reasons = []
        
        # High probability
        if fraud_probability > 80:
            reasons.append("extremely high fraud probability")
        elif fraud_probability > 60:
            reasons.append("high fraud probability")
        
        # Amount-based
        if 'amount' in transaction and transaction['amount'] > 10000:
            reasons.append("unusually large transaction amount")
        
        # Temporal
        if transaction.get('is_night', False):
            reasons.append("transaction occurred during high-risk hours")
        
        # Balance discrepancy
        if 'balance_error_orig' in transaction and abs(transaction['balance_error_orig']) > 100:
            reasons.append("significant balance discrepancy detected")
        
        # Network anomaly
        if 'out_degree' in transaction and transaction['out_degree'] > 10:
            reasons.append("abnormal network activity pattern")
        
        # Similar cases
        if similar_cases and len(similar_cases) > 0:
            avg_similarity = sum(score for _, score in similar_cases) / len(similar_cases)
            if avg_similarity > 0.8:
                reasons.append(f"strong similarity to {len(similar_cases)} known fraud cases")
        
        # Construct explanation
        if reasons:
            explanation = f"This transaction was flagged as fraud due to: {', '.join(reasons)}. "
        else:
            explanation = "This transaction exhibits patterns consistent with fraudulent behavior. "
        
        explanation += f"The model assigns a {fraud_probability:.1f}% fraud probability based on learned patterns from historical data."
        
        return explanation


def generate_batch_explanations(
    explainer: FraudExplainer,
    predictions: List[Dict],
    save_path: Optional[Path] = None
) -> List[Dict]:
    """
    Generate explanations for batch of predictions.
    
    Args:
        explainer: FraudExplainer instance
        predictions: List of prediction dictionaries
        save_path: Optional path to save explanations
        
    Returns:
        List of explanation dictionaries
    """
    print(f'\n🔍 Generating explanations for {len(predictions)} predictions...')
    
    explanations = []
    
    for pred in predictions:
        explanation = explainer.explain_prediction(
            transaction=pred['transaction'],
            fraud_probability=pred['fraud_probability'],
            prediction=pred['prediction']
        )
        explanations.append(explanation)
    
    print(f'   ✅ Generated {len(explanations)} explanations')
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(explanations, f, indent=2)
        
        print(f'   💾 Saved to {save_path}')
    
    return explanations


if __name__ == '__main__':
    from pathlib import Path
    from src.rag.retriever import build_fraud_case_database
    
    base_path = Path(__file__).parent.parent.parent / 'data'
    
    # Build retriever
    retriever = build_fraud_case_database(base_path, dataset='paysim', n_cases=100)
    
    # Create explainer
    explainer = FraudExplainer(
        retriever=retriever,
        llm_provider='openai',  # Falls back to template if no API key
        temperature=0.3
    )
    
    # Test explanation
    test_transaction = {
        'amount': 8500.0,
        'type': 'TRANSFER',
        'hour': 2,
        'is_night': True,
        'balance_error_orig': 1200.0
    }
    
    result = explainer.explain_prediction(
        transaction=test_transaction,
        fraud_probability=92.5,
        prediction='FRAUD'
    )
    
    print(f'\n📝 Explanation:')
    print(result['explanation'])