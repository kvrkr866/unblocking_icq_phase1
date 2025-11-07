"""
##########################################################
#
# RAG Test Configuration
# Configuration file for RAG test suite
#
##########################################################
"""

# Test Configuration
TEST_CONFIG = {
    # Metrics thresholds
    "thresholds": {
        "precision_at_5": 0.6,  # Minimum precision@5
        "recall_at_5": 0.5,     # Minimum recall@5
        "mrr": 0.7,              # Minimum MRR
        "ndcg_at_5": 0.6,        # Minimum NDCG@5
        "hit_rate_at_5": 0.7,    # Minimum hit rate@5
    },
    
    # Performance thresholds
    "performance": {
        "retrieval_latency_max": 1.0,      # seconds
        "synthesis_latency_max": 5.0,      # seconds
        "total_latency_max": 10.0,         # seconds
    },
    
    # Test settings
    "test_settings": {
        "num_test_cases": 10,              # Number of test cases to run
        "k_values": [1, 3, 5, 10],         # K values for metrics
        "enable_ragas": True,              # Enable RAGAS evaluation
        "enable_langsmith": False,         # Enable LangSmith evaluation
    },
    
    # RAG settings
    "rag_settings": {
        "k": 10,                           # Default number of docs to retrieve
        "fetch_k": 20,                     # Fetch k for MMR
        "chunk_size": 1000,                # Text chunk size
        "chunk_overlap": 200,              # Chunk overlap
    },
}

# Export configuration
__all__ = ["TEST_CONFIG"]

