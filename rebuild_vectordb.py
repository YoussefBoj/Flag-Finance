"""
Quick Fix: Rebuild Vector Database for RAG
Fixes the tuple indexing error
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag.retriever import build_fraud_case_database

def main():
    """Rebuild the vector database with proper format."""
    
    print("="*70)
    print("FLAG-Finance: Vector Database Rebuild")
    print("="*70)
    print()
    print("This script will rebuild the RAG vector database with the correct format.")
    print("This fixes the 'tuple indices must be integers or slices' error.")
    print()
    
    # Get data path
    data_path = Path(__file__).parent / 'data'
    
    if not data_path.exists():
        print(f"❌ Error: Data directory not found at {data_path}")
        print()
        print("Please ensure you have the data directory with processed data.")
        return 1
    
    # Check for processed data
    paysim_file = data_path / 'processed' / 'paysim_sample_enhanced.csv'
    if not paysim_file.exists():
        paysim_file = data_path / 'processed' / 'paysim_data_enhanced.csv'
    
    if not paysim_file.exists():
        print(f"❌ Error: No processed PaySim data found")
        print()
        print("Expected location:")
        print(f"  {paysim_file}")
        print()
        print("Please run data processing first:")
        print("  python src/data/ingest.py")
        return 1
    
    # Build database
    try:
        print("Building vector database...")
        print()
        
        retriever = build_fraud_case_database(
            data_path=data_path,
            dataset='paysim',
            n_cases=500,
            save=True
        )
        
        print()
        print("="*70)
        print("✅ Vector Database Rebuilt Successfully!")
        print("="*70)
        print()
        print("Location:", data_path / 'vector_db' / 'fraud_cases_faiss')
        print()
        print("You can now restart your API:")
        print("  uvicorn src.api.app:app --reload")
        print()
        
        return 0
        
    except FileNotFoundError as e:
        print()
        print(f"❌ Error: {e}")
        print()
        print("Please ensure you have processed the data first:")
        print("  python src/data/ingest.py")
        return 1
        
    except Exception as e:
        print()
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())