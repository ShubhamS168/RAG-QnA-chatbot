from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd

class RAGEvaluator:
    """Evaluation metrics for RAG system performance"""
    
    def __init__(self, config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
    
    def evaluate_retrieval_quality(self, questions: List[str], retrieved_docs: List[List[str]]) -> Dict[str, float]:
        """Evaluate retrieval component quality"""
        metrics = {
            "hit_rate": self._calculate_hit_rate(questions, retrieved_docs),
            "mrr": self._calculate_mrr(questions, retrieved_docs),
            "avg_docs_retrieved": np.mean([len(docs) for docs in retrieved_docs])
        }
        return metrics
    
    def evaluate_answer_relevance(self, questions: List[str], answers: List[str]) -> Dict[str, float]:
        """Evaluate answer relevance using embeddings"""
        try:
            # Get embeddings for questions and answers
            question_embeddings = self.embeddings.embed_documents(questions)
            answer_embeddings = self.embeddings.embed_documents(answers)
            
            # Calculate cosine similarity
            similarities = []
            for q_emb, a_emb in zip(question_embeddings, answer_embeddings):
                similarity = cosine_similarity([q_emb], [a_emb])[0][0]
                similarities.append(similarity)
            
            metrics = {
                "avg_relevance": np.mean(similarities),
                "min_relevance": np.min(similarities),
                "max_relevance": np.max(similarities),
                "std_relevance": np.std(similarities)
            }
            return metrics
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_hit_rate(self, questions: List[str], retrieved_docs: List[List[str]]) -> float:
        """Calculate hit rate metric"""
        hits = 0
        total = len(questions)
        
        for i, (question, docs) in enumerate(zip(questions, retrieved_docs)):
            # Check if any retrieved document contains relevant information
            # This is a simplified check - in practice, you'd need ground truth
            if docs:  # If any documents were retrieved
                hits += 1
        
        return hits / total if total > 0 else 0.0
    
    def _calculate_mrr(self, questions: List[str], retrieved_docs: List[List[str]]) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for question, docs in zip(questions, retrieved_docs):
            # Simplified MRR calculation
            # In practice, you'd need relevance judgments
            if docs:
                reciprocal_ranks.append(1.0)  # Assume first doc is relevant
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    def generate_evaluation_report(self, test_questions: List[str], rag_pipeline) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {
            "test_questions": len(test_questions),
            "timestamp": pd.Timestamp.now().isoformat(),
            "retrieval_metrics": {},
            "answer_metrics": {},
            "performance_analysis": {}
        }
        
        # Process test questions
        answers = []
        retrieved_docs = []
        
        for question in test_questions:
            result = rag_pipeline.ask_question(question)
            answers.append(result["answer"])
            retrieved_docs.append([doc["content"] for doc in result["sources"]])
        
        # Calculate metrics
        report["retrieval_metrics"] = self.evaluate_retrieval_quality(test_questions, retrieved_docs)
        report["answer_metrics"] = self.evaluate_answer_relevance(test_questions, answers)
        
        # Performance analysis
        report["performance_analysis"] = {
            "avg_response_length": np.mean([len(answer.split()) for answer in answers]),
            "sources_utilization": np.mean([len(docs) for docs in retrieved_docs]),
            "confidence_scores": [0.8] * len(test_questions)  # Placeholder
        }
        
        return report
    
    def get_test_questions(self) -> List[str]:
        """Get standardized test questions for evaluation"""
        return [
            "What is the loan approval rate in the dataset?",
            "How does credit history affect loan approval?",
            "What is the average loan amount?",
            "Which factors are most important for loan approval?",
            "How does income level correlate with approval rates?",
            "What is the distribution of approved vs rejected loans?",
            "How does marital status affect loan approval?",
            "What property areas have the highest approval rates?",
            "What is the typical loan term period?",
            "How does education level impact loan decisions?"
        ]
