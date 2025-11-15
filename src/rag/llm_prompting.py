"""
LLM-based fraud explanation generation using RAG
Supports OpenAI → Perplexity → Local Phi fallback chain
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


# ============================================================================
# Production Fraud Explainer with OpenAI → Perplexity → Phi Fallback
# ============================================================================

class ProductionFraudExplainer:
    """Production-ready fraud explainer with OpenAI → Perplexity → Phi fallback."""
    
    def __init__(
        self,
        retriever=None,
        use_llm: bool = False,
        openai_key: Optional[str] = None,
        perplexity_key: Optional[str] = None,
        local_model: Optional[str] = None,
        model_kwargs: Optional[Dict] = None,
        generation_kwargs: Optional[Dict] = None
    ):
        self.retriever = retriever
        self.llm = None
        self.llm_error = None
        self.requested_llm = use_llm
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        self.perplexity_key = perplexity_key or os.getenv('PERPLEXITY_API_KEY')
        self.local_model = local_model
        self.model_kwargs = model_kwargs or {}
        self.generation_kwargs = generation_kwargs or {'max_new_tokens': 220, 'temperature': 0.3}
        self.llm_backend = None
    
        if not use_llm:
            self.use_llm = False
            return
    
        # Try OpenAI first
        if self.openai_key:
            try:
                from langchain_openai import OpenAI
                self.llm = OpenAI(temperature=0.3, api_key=self.openai_key)
                self.llm_backend = 'openai'
                self.use_llm = True
                print('✅ OpenAI LLM initialized')
                return
            except Exception as exc:
                self.llm_error = str(exc)
                print(f'⚠️ OpenAI initialization failed: {exc}')
    
        # Try Perplexity second
        if self.perplexity_key:
            try:
                from langchain_community.chat_models import ChatPerplexity
                self.llm = ChatPerplexity(
                    api_key=self.perplexity_key,
                    temperature=0.3,
                    model="sonar-reasoning"
                )
                self.llm_backend = 'perplexity'
                self.use_llm = True
                print('✅ Perplexity LLM initialized')
                return
            except Exception as exc:
                self.llm_error = str(exc)
                print(f'⚠️ Perplexity initialization failed: {exc}')
    
        # Try local model last
        if self.local_model:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                print(f'🚀 Loading local model: {self.local_model}')
                tokenizer = AutoTokenizer.from_pretrained(self.local_model, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(self.local_model, **self.model_kwargs)
                self.llm = pipeline('text-generation', model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)
                self.llm_backend = 'huggingface-pipeline'
                self.use_llm = True
                print('✅ Local LLM initialized')
                return
            except Exception as exc:
                self.llm_error = str(exc)
                print(f'⚠️ Local LLM initialization failed: {exc}')
    
        self.use_llm = False
        print("⚠️ No valid LLM backend available, using templates only.")

    def explain_prediction(
        self,
        transaction: Dict,
        prediction: int,
        fraud_probability: float,
        confidence: Optional[str] = None,
        top_k: int = 3
    ) -> str:
        """Generate explanation with LLM fallback chain."""
        
        # Determine confidence
        if confidence is None:
            if fraud_probability >= 0.8 or fraud_probability <= 0.2:
                confidence = "HIGH"
            elif fraud_probability >= 0.6 or fraud_probability <= 0.4:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        
        tx_type = transaction.get('type', 'UNKNOWN')
        amount = transaction.get('amount', 0.0)
        
        # Try LLM generation with fallback
        if self.use_llm and self.llm:
            try:
                # Build prompt
                prompt = f"Explain why this {tx_type} transaction of ${amount:.2f} is {'fraud' if prediction == 1 else 'legitimate'} with {fraud_probability*100:.1f}% probability. Be concise (2-3 sentences)."
                
                if self.llm_backend == 'huggingface-pipeline':
                    result = self.llm(prompt, **self.generation_kwargs)[0]['generated_text']
                    return result[len(prompt):].strip() or result.strip()
                
                if hasattr(self.llm, 'invoke'):
                    response = self.llm.invoke(prompt)
                    if hasattr(response, 'content'):
                        return response.content
                    return str(response)
            
            except Exception as exc:
                error_message = str(exc)
                print(f'⚠️ LLM generation failed: {error_message}')
                
                # OpenAI quota exceeded? Try Perplexity
                if self.llm_backend == "openai" and ("quota" in error_message.lower() or "429" in error_message):
                    print("⚠️ OpenAI quota exceeded. Trying Perplexity...")
                    if self.perplexity_key:
                        try:
                            from langchain_community.chat_models import ChatPerplexity
                            self.llm = ChatPerplexity(
                                api_key=self.perplexity_key,
                                temperature=0.3,
                                model="sonar-reasoning"
                            )
                            self.llm_backend = 'perplexity'
                            self.use_llm = True
                            print('✅ Switched to Perplexity')
                            return self.explain_prediction(transaction, prediction, fraud_probability, confidence, top_k)
                        except Exception as per_exc:
                            print(f"⚠️ Perplexity failed: {per_exc}")
                
                # Try local model
                if self.local_model and self.llm_backend in ["openai", "perplexity", None]:
                    print("⚠️ Trying local Phi model...")
                    try:
                        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                        tokenizer = AutoTokenizer.from_pretrained(self.local_model, trust_remote_code=True)
                        model = AutoModelForCausalLM.from_pretrained(self.local_model, **self.model_kwargs)
                        self.llm = pipeline('text-generation', model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)
                        self.llm_backend = 'huggingface-pipeline'
                        self.use_llm = True
                        print('✅ Switched to local Phi')
                        return self.explain_prediction(transaction, prediction, fraud_probability, confidence, top_k)
                    except Exception as local_exc:
                        print(f"⚠️ Local model failed: {local_exc}")
        
        # Template fallback
        if prediction == 1:
            signals = []
            if abs(transaction.get('balance_error_orig', 0)) > 0:
                signals.append("origin balance mismatch")
            if abs(transaction.get('balance_error_dest', 0)) > 0:
                signals.append("destination balance mismatch")
            if transaction.get('isFlaggedFraud', 0) == 1:
                signals.append("system flag")
            if not signals:
                signals.append("pattern anomalies")
            
            return (f"This {tx_type} transaction of ${amount:,.2f} is flagged as "
                   f"FRAUD with {confidence} confidence ({fraud_probability*100:.1f}% probability). "
                   f"Key indicators: {', '.join(signals)}.")
        else:
            return (f"This {tx_type} transaction of ${amount:,.2f} appears LEGITIMATE "
                   f"with {confidence} confidence ({(1-fraud_probability)*100:.1f}% probability). "
                   f"Transaction patterns align with normal behavior.")


# ============================================================================
# Advanced RAG-based Fraud Explainer (Original Implementation)
# ============================================================================

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


# ============================================================================
# Batch Processing
# ============================================================================

def generate_batch_explanations(
    explainer,
    predictions: List[Dict],
    save_path: Optional[Path] = None
) -> List[Dict]:
    """
    Generate explanations for batch of predictions.
    
    Args:
        explainer: FraudExplainer or ProductionFraudExplainer instance
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


# Export classes for easy import
__all__ = ['FraudExplainer', 'ProductionFraudExplainer', 'generate_batch_explanations']


# ============================================================================
# Test/Demo Code
# ============================================================================

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
        fraud_probability=0.925,  # Note: now 0-1 range
        prediction=1
    )
    
    print(f'\n📝 Explanation:')
    print(result)
