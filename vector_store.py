import faiss
import numpy as np
from typing import List, Tuple, Optional
import pickle
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import streamlit as st

class VectorStoreManager:
    """Manages vector storage and retrieval operations"""
    
    def __init__(self, config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents"""
        try:
            with st.spinner("Creating vector embeddings..."):
                # Create vector store
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                
                # Save vector store
                self._save_vector_store()
                
            st.success(f"Vector store created with {len(documents)} documents")
            return self.vector_store
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self) -> Optional[FAISS]:
        """Load existing vector store"""
        try:
            if os.path.exists(self.config.VECTOR_STORE_PATH):
                self.vector_store = FAISS.load_local(
                    self.config.VECTOR_STORE_PATH,
                    self.embeddings
                )
                return self.vector_store
            return None
        except Exception as e:
            st.warning(f"Could not load vector store: {str(e)}")
            return None
    
    def _save_vector_store(self):
        """Save vector store to disk"""
        if self.vector_store:
            os.makedirs(self.config.VECTOR_STORE_PATH, exist_ok=True)
            self.vector_store.save_local(self.config.VECTOR_STORE_PATH)
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Perform similarity search"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        k = k or self.config.TOP_K_RETRIEVAL
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            return results
            
        except Exception as e:
            st.error(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Perform similarity search with relevance scores"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        k = k or self.config.TOP_K_RETRIEVAL
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by similarity threshold
            filtered_results = [
                (doc, score) for doc, score in results
                if score >= self.config.SIMILARITY_THRESHOLD
            ]
            
            return filtered_results
            
        except Exception as e:
            st.error(f"Error during similarity search: {str(e)}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            self.vector_store.add_documents(documents)
            self._save_vector_store()
            st.success(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            raise
    
    def get_store_stats(self) -> dict:
        """Get vector store statistics"""
        if not self.vector_store:
            return {"status": "Not initialized"}
        
        try:
            # Get index stats
            index = self.vector_store.index
            stats = {
                "status": "Ready",
                "total_vectors": index.ntotal,
                "dimension": index.d,
                "is_trained": index.is_trained
            }
            return stats
            
        except Exception as e:
            return {"status": f"Error: {str(e)}"}
