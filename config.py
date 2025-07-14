import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class to manage environment variables and settings.
    """
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables. "
                         "Please set it in a .env file or directly in your environment.")

    # Retriever settings
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2" # A good balance of performance and size
    TOP_K_RETRIEVAL = 3 # Number of top contexts to retrieve

    # Text chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Supported file types
    SUPPORTED_FILE_TYPES = ["pdf", "txt", "docx", "csv"]

# Example usage:
# from config import Config
# api_key = Config.GEMINI_API_KEY
