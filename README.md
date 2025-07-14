# RAG Q&A Chatbot - Loan Approval Assistant

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot for analyzing loan approval data. The system combines document retrieval with generative AI to provide intelligent responses about loan approval criteria, statistics, and insights.

## Features
- 🤖 **OpenAI GPT Integration**: Powered by GPT-3.5-turbo for intelligent responses
- 🔍 **FAISS Vector Search**: Efficient similarity search for document retrieval
- 📊 **Interactive Dashboard**: Streamlit-based user interface
- 📈 **Data Visualization**: Charts and metrics for data insights
- 🎯 **Evaluation Metrics**: Comprehensive RAG system evaluation
- 🚀 **Easy Deployment**: One-click Streamlit deployment

## Installation

1. Clone the repository:
git clone <repository-url>
cd rag-chatbot

text

2. Install dependencies:
pip install -r requirements.txt

text

3. Set up OpenAI API key in `config.py`

4. Run the application:
streamlit run main.py

text

## Usage

1. **Start the Application**: Run `streamlit run main.py`
2. **Initialize System**: Wait for vector store creation
3. **Ask Questions**: Use the chat interface to ask about loan data
4. **View Sources**: Check retrieved documents for each answer
5. **Adjust Settings**: Modify model parameters in the sidebar

## Project Structure
- `main.py`: Main Streamlit application
- `config.py`: Configuration settings
- `data_loader.py`: Data loading and preprocessing
- `vector_store.py`: Vector store operations
- `rag_pipeline.py`: RAG implementation
- `evaluation.py`: Evaluation metrics
- `utils.py`: Utility functions

## Evaluation
The system includes comprehensive evaluation metrics:
- **Hit Rate**: Percentage of successful retrievals
- **MRR**: Mean Reciprocal Rank for ranking quality
- **Answer Relevance**: Semantic similarity between questions and answers
- **Confidence Scoring**: Automated confidence assessment

## Deployment
Deploy to Streamlit Cloud:
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with automatic requirements detection

## Contributing
This project was developed as a data science intern project demonstrating RAG system implementation with modern tools and best practices.