from typing import List, Dict, Any, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import streamlit as st
import time

class RAGPipeline:
    """Main RAG pipeline for question answering"""
    
    def __init__(self, config, vector_store_manager):
        self.config = config
        self.vector_store_manager = vector_store_manager
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=config.CHAT_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Custom prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a helpful AI assistant specializing in loan approval analysis. 
            Use the following context to answer questions about loan applications, approval criteria, 
            and dataset insights. If the information is not available in the context, clearly state that.

            Context: {context}

            Question: {question}

            Provide a detailed, informative answer based on the context. Include relevant statistics 
            and insights when applicable.

            Answer:""",
            input_variables=["context", "question"]
        )
    
    def setup_qa_chain(self):
        """Set up the question-answering chain"""
        if not self.vector_store_manager.vector_store:
            raise ValueError("Vector store not initialized")
        
        # Create retriever
        retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={"k": self.config.TOP_K_RETRIEVAL}
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Process a question and return answer with sources"""
        try:
            # Setup QA chain if not already done
            if not hasattr(self, 'qa_chain'):
                self.setup_qa_chain()
            
            # Show thinking indicator
            with st.spinner("Thinking..."):
                # Add slight delay for better UX
                time.sleep(0.5)
                
                # Process question
                result = self.qa_chain({"query": question})
                
                # Extract answer and sources
                answer = result["result"]
                source_documents = result["source_documents"]
                
                # Format response
                response = {
                    "answer": answer,
                    "sources": self._format_sources(source_documents),
                    "retrieved_documents": len(source_documents),
                    "confidence": self._calculate_confidence(source_documents)
                }
                
                return response
                
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            return {
                "answer": "I'm sorry, I encountered an error while processing your question. Please try again.",
                "sources": [],
                "retrieved_documents": 0,
                "confidence": 0.0
            }
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for display"""
        sources = []
        for i, doc in enumerate(documents):
            source = {
                "id": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "type": doc.metadata.get("type", "unknown")
            }
            sources.append(source)
        return sources
    
    def _calculate_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score based on retrieved documents"""
        if not documents:
            return 0.0
        
        # Simple confidence calculation based on number of sources
        base_confidence = min(len(documents) / self.config.TOP_K_RETRIEVAL, 1.0)
        
        # Boost confidence for diverse source types
        source_types = set(doc.metadata.get("type", "unknown") for doc in documents)
        diversity_bonus = len(source_types) / 5  # Assuming max 5 types
        
        confidence = min(base_confidence + diversity_bonus, 1.0)
        return round(confidence, 2)
    
    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Get sample questions for the UI"""
        questions = [
            {
                "category": "Dataset Overview",
                "questions": [
                    "What is the overall loan approval rate?",
                    "How many loan applications are in the dataset?",
                    "What are the key features in the dataset?"
                ]
            },
            {
                "category": "Loan Analysis",
                "questions": [
                    "What factors most influence loan approval?",
                    "How does credit history affect approval rates?",
                    "What is the average loan amount?"
                ]
            },
            {
                "category": "Demographics",
                "questions": [
                    "How does gender affect loan approval?",
                    "Do married applicants have better approval rates?",
                    "What is the impact of education level?"
                ]
            },
            {
                "category": "Financial Insights",
                "questions": [
                    "What is the typical income range for approved loans?",
                    "How important is coapplicant income?",
                    "What property areas have the highest approval rates?"
                ]
            }
        ]
        return questions
    
    def update_model_settings(self, temperature: float, max_tokens: int):
        """Update model settings"""
        self.llm.temperature = temperature
        self.llm.max_tokens = max_tokens
        
        # Reinitialize QA chain with new settings
        if hasattr(self, 'qa_chain'):
            self.setup_qa_chain()
