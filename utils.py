import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def format_currency(amount: float) -> str:
    """Format currency with proper formatting"""
    return f"${amount:,.2f}"

def display_metrics(metrics: Dict[str, Any], title: str = "Metrics"):
    """Display metrics in a formatted way"""
    st.subheader(title)
    
    # Create columns based on number of metrics
    cols = st.columns(len(metrics))
    
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(value, float):
                if 0 <= value <= 1:
                    st.metric(key.replace('_', ' ').title(), f"{value:.2%}")
                else:
                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
            else:
                st.metric(key.replace('_', ' ').title(), str(value))

def create_download_link(data: Any, filename: str, text: str = "Download") -> str:
    """Create a download link for data"""
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
    elif isinstance(data, dict):
        json_str = json.dumps(data, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
    else:
        b64 = base64.b64encode(str(data).encode()).decode()
    
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def show_evaluation_results(evaluation_report: Dict[str, Any]):
    """Display evaluation results in a formatted way"""
    st.header("📊 Evaluation Results")
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Questions", evaluation_report["test_questions"])
    with col2:
        st.metric("Avg Response Length", f"{evaluation_report['performance_analysis']['avg_response_length']:.1f} words")
    with col3:
        st.metric("Sources Per Response", f"{evaluation_report['performance_analysis']['sources_utilization']:.1f}")
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Retrieval Metrics")
        display_metrics(evaluation_report["retrieval_metrics"])
    
    with col2:
        st.subheader("💬 Answer Metrics")
        display_metrics(evaluation_report["answer_metrics"])
    
    # Download report
    st.download_button(
        label="📥 Download Evaluation Report",
        data=json.dumps(evaluation_report, indent=2),
        file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def create_visualization(data: pd.DataFrame, chart_type: str, **kwargs) -> go.Figure:
    """Create various types of visualizations"""
    
    if chart_type == "bar":
        fig = px.bar(data, **kwargs)
    elif chart_type == "line":
        fig = px.line(data, **kwargs)
    elif chart_type == "scatter":
        fig = px.scatter(data, **kwargs)
    elif chart_type == "pie":
        fig = px.pie(data, **kwargs)
    elif chart_type == "histogram":
        fig = px.histogram(data, **kwargs)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    return fig

def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    return api_key.startswith("sk-") and len(api_key) > 20

def log_interaction(question: str, answer: str, sources: List[Dict], confidence: float):
    """Log user interactions for analysis"""
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "sources_count": len(sources),
        "confidence": confidence
    }
    
    # In production, you'd save to a database
    # For now, we'll just store in session state
    if 'interaction_log' not in st.session_state:
        st.session_state.interaction_log = []
    
    st.session_state.interaction_log.append(interaction)

def get_interaction_analytics():
    """Get analytics from interaction log"""
    if 'interaction_log' not in st.session_state:
        return None
    
    log = st.session_state.interaction_log
    
    analytics = {
        "total_interactions": len(log),
        "avg_confidence": np.mean([i["confidence"] for i in log]),
        "avg_sources": np.mean([i["sources_count"] for i in log]),
        "common_questions": {},  # Would implement with more data
        "interaction_timeline": [i["timestamp"] for i in log]
    }
    
    return analytics

def export_conversation_history() -> str:
    """Export conversation history as JSON"""
    if 'messages' not in st.session_state:
        return "{}"
    
    history = {
        "export_timestamp": datetime.now().isoformat(),
        "messages": st.session_state.messages
    }
    
    return json.dumps(history, indent=2)
