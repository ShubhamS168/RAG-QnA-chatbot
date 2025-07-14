import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    """Configuration settings for the RAG chatbot"""
    
    # OpenAI API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 500
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: str = "vector_store"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Dataset Configuration
    DATASET_URL: str = "https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction"
    LOCAL_DATA_PATH: str = "data/loan_data.csv"
    
    # RAG Configuration
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Streamlit Configuration
    PAGE_TITLE: str = "RAG Q&A Chatbot - Loan Approval Assistant"
    PAGE_ICON: str = "🤖"
    
    def __post_init__(self):
        """Set OpenAI API key as environment variable"""
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY

# Global configuration instance
config = Config()
