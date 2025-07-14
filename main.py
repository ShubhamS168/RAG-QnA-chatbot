import streamlit as st
import pandas as pd
from config import config
from data_loader import DataLoader
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline
from evaluation import RAGEvaluator
# main.py  – use an explicit relative import
from .utils import format_currency, display_metrics, create_download_link
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

@st.cache_resource
def initialize_system():
    """Initialize the RAG system components"""
    data_loader = DataLoader(config)
    vector_store_manager = VectorStoreManager(config)
    rag_pipeline = RAGPipeline(config, vector_store_manager)
    evaluator = RAGEvaluator(config)
    
    return data_loader, vector_store_manager, rag_pipeline, evaluator

def setup_vector_store(data_loader, vector_store_manager):
    """Setup or load vector store"""
    if not st.session_state.vector_store_ready:
        # Try to load existing vector store
        if vector_store_manager.load_vector_store():
            st.success("✅ Vector store loaded successfully!")
            st.session_state.vector_store_ready = True
        else:
            # Create new vector store
            with st.spinner("Setting up knowledge base..."):
                df = data_loader.load_loan_data()
                documents = data_loader.create_documents_from_data(df)
                split_documents = data_loader.split_documents(documents)
                
                vector_store_manager.create_vector_store(split_documents)
                st.session_state.vector_store_ready = True
                st.success("✅ Knowledge base created successfully!")

def main():
    """Main application"""
    
    # Header
    st.title("🤖 RAG Q&A Chatbot - Loan Approval Assistant")
    st.markdown("*Powered by OpenAI GPT and FAISS Vector Search*")
    
    # Initialize components
    data_loader, vector_store_manager, rag_pipeline, evaluator = initialize_system()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        model_name = st.selectbox(
            "Select Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config.TEMPERATURE,
            step=0.1
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=1000,
            value=config.MAX_TOKENS,
            step=50
        )
        
        # Update settings
        if st.button("Update Settings"):
            config.CHAT_MODEL = model_name
            config.TEMPERATURE = temperature
            config.MAX_TOKENS = max_tokens
            st.success("Settings updated!")
        
        st.divider()
        
        # System status
        st.subheader("System Status")
        if st.session_state.vector_store_ready:
            st.success("✅ Vector Store: Ready")
            
            # Show vector store stats
            stats = vector_store_manager.get_store_stats()
            st.json(stats)
        else:
            st.warning("⚠️ Vector Store: Not Ready")
        
        st.divider()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Setup vector store
        setup_vector_store(data_loader, vector_store_manager)
        
        # Chat interface
        if st.session_state.vector_store_ready:
            # Initialize RAG pipeline
            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = rag_pipeline
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Display sources for assistant messages
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("📖 Sources"):
                            for source in message["sources"]:
                                st.markdown(f"**Source {source['id']}** ({source['type']})")
                                st.markdown(f"``````")
            
            # Chat input
            if prompt := st.chat_input("Ask about loan approval data..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = st.session_state.rag_pipeline.ask_question(prompt)
                        
                        # Display answer
                        st.markdown(result["answer"])
                        
                        # Display metadata
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Sources", result["retrieved_documents"])
                        with col_b:
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                        with col_c:
                            st.metric("Response Time", "< 1s")
                        
                        # Add to messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "confidence": result["confidence"]
                        })
                        
                        # Display sources
                        if result["sources"]:
                            with st.expander("📖 Sources"):
                                for source in result["sources"]:
                                    st.markdown(f"**Source {source['id']}** ({source['type']})")
                                    st.markdown(f"``````")
        
        else:
            st.warning("Please wait while the system initializes...")
    
    with col2:
        st.header("📊 Knowledge Base")
        
        # Dataset overview
        df = data_loader.load_loan_data()
        stats = data_loader.get_dataset_stats(df)
        
        # Metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Records", stats["total_records"])
            st.metric("Approval Rate", f"{stats['approval_rate']:.1%}")
        with col_b:
            st.metric("Features", stats["features"])
            st.metric("Avg Loan Amount", f"${stats['average_loan_amount']:,.0f}")
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = rag_pipeline.get_sample_questions()
        
        for category in sample_questions:
            with st.expander(f"📋 {category['category']}"):
                for question in category['questions']:
                    if st.button(question, key=f"btn_{question[:20]}"):
                        # Add question to chat
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
        
        # Dataset preview
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Visualizations
        st.subheader("📈 Quick Insights")
        
        # Loan status distribution
        status_counts = df['Loan_Status'].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=['Approved', 'Rejected'],
            title="Loan Status Distribution"
        )
        st.plotly_chart(fig_status, use_container_width=True)
        
        # Income distribution
        fig_income = px.histogram(
            df, 
            x='ApplicantIncome', 
            title="Applicant Income Distribution",
            nbins=20
        )
        st.plotly_chart(fig_income, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("---")
    st.markdown("""
    **RAG Q&A Chatbot** - Built with OpenAI GPT, FAISS, and Streamlit  
    *Data Science Intern Project - 2024*
    """)

if __name__ == "__main__":
    main()
