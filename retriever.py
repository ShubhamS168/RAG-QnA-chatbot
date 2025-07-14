import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

from config import Config

class Retriever:
    """
    Handles embedding generation, FAISS index creation, and context retrieval.
    """
    def __init__(self, model_name: str = Config.SENTENCE_TRANSFORMER_MODEL):
        """
        Initializes the SentenceTransformer model.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model '{model_name}': {e}")

    def build_faiss_index(self, texts: List[str]) -> Tuple[faiss.Index, List[str]]:
        """
        Generates embeddings for a list of texts and builds a FAISS index.

        Args:
            texts (List[str]): A list of text chunks.

        Returns:
            Tuple[faiss.Index, List[str]]: A tuple containing the FAISS index and the original texts.
        """
        if not texts:
            return None, []
        try:
            print(f"Generating embeddings for {len(texts)} text chunks...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            # Ensure embeddings are float32 as required by FAISS
            embeddings = np.array(embeddings).astype('float32')

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension) # L2 distance for similarity search
            index.add(embeddings)
            print("FAISS index built successfully.")
            return index, texts
        except Exception as e:
            raise RuntimeError(f"Error building FAISS index: {e}")

    def retrieve_context(self, query: str, index: faiss.Index, texts: List[str], k: int = Config.TOP_K_RETRIEVAL) -> List[str]:
        """
        Retrieves the top-k most relevant contexts for a given query from the FAISS index.

        Args:
            query (str): The natural language query.
            index (faiss.Index): The FAISS index.
            texts (List[str]): The original list of text chunks corresponding to the index.
            k (int): The number of top contexts to retrieve.

        Returns:
            List[str]: A list of the top-k relevant text chunks.
        """
        if index is None or not texts:
            return []
        try:
            query_embedding = self.model.encode([query]).astype('float32')
            # D, I are distances and indices respectively
            distances, indices = index.search(query_embedding, k)

            retrieved_contexts = []
            for i in indices[0]:
                if i != -1: # Ensure index is valid
                    retrieved_contexts.append(texts[i])
            return retrieved_contexts
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

# Example usage (for testing purposes, not part of the main app flow):
if __name__ == "__main__":
    retriever = Retriever()
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is a rapidly developing field.",
        "Machine learning is a subset of AI.",
        "The cat sat on the mat.",
        "Deep learning is a specific type of machine learning."
    ]

    faiss_index, original_texts = retriever.build_faiss_index(sample_texts)

    if faiss_index:
        query = "What is AI?"
        contexts = retriever.retrieve_context(query, faiss_index, original_texts, k=2)
        print(f"\nQuery: '{query}'")
        print("Retrieved Contexts:")
        for i, context in enumerate(contexts):
            print(f"  {i+1}. {context}")

        query = "Tell me about the fox."
        contexts = retriever.retrieve_context(query, faiss_index, original_texts, k=1)
        print(f"\nQuery: '{query}'")
        print("Retrieved Contexts:")
        for i, context in enumerate(contexts):
            print(f"  {i+1}. {context}")
