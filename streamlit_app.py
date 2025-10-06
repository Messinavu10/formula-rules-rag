#!/usr/bin/env python3
"""
FIA Formula 1 Regulations Chatbot UI
A modern Streamlit interface for the FIA agentic RAG system
"""

import streamlit as st
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv()

from rag.agent import FIAAgent
from rag.rag_pipeline import FIARAGPipeline

# Page configuration
st.set_page_config(
    page_title="FIA Formula 1 Regulations Assistant",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for F1 theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF0000, #FFFFFF, #000000);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #FF0000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #f8f9fa;
        border-left-color: #FF0000;
        color: #2c3e50;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border-left-color: #0066cc;
        color: #2c3e50;
        line-height: 1.6;
        border: 1px solid #e9ecef;
    }
    
    .tool-indicator {
        background-color: #FF0000;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
        font-weight: 500;
    }
    
    .response-content {
        margin-top: 1rem;
        padding: 0.5rem 0;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_agent():
    """Initialize the FIA agent with error handling."""
    try:
        # Check environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "fia-rules")
        
        if not openai_api_key:
            st.error("❌ OPENAI_API_KEY not found. Please set it in your environment.")
            return None
            
        if not pinecone_api_key:
            st.error("❌ PINECONE_API_KEY not found. Please set it in your environment.")
            return None
        
        # Initialize RAG pipeline
        with st.spinner("🔧 Initializing RAG pipeline..."):
            rag_pipeline = FIARAGPipeline(
                index_name=index_name,
                openai_api_key=openai_api_key,
                pinecone_api_key=pinecone_api_key,
                model_name="gpt-4o-mini"
            )
        
        # Initialize agent
        with st.spinner("🤖 Initializing FIA Agent..."):
            agent = FIAAgent(
                rag_pipeline=rag_pipeline,
                model_name="gpt-4o-mini",
                enable_tracing=bool(langsmith_api_key),
                langsmith_api_key=langsmith_api_key
            )
        
        return agent
        
    except Exception as e:
        st.error(f"❌ Error initializing agent: {str(e)}")
        return None

def format_response_text(text):
    """Format response text for better readability."""
    # Replace numbered lists with proper markdown
    import re
    
    # Handle numbered lists (1. 2. 3. etc.)
    text = re.sub(r'(\d+)\.\s+', r'\n**\1.** ', text)
    
    # Handle bullet points
    text = re.sub(r'•\s+', r'\n• ', text)
    
    # Handle double line breaks for paragraphs
    text = re.sub(r'\n\n+', r'\n\n', text)
    
    # Clean up extra whitespace
    text = text.strip()
    
    return text

def display_chat_message(message, is_user=False, tool_used=None, metadata=None):
    """Display a chat message with styling."""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        tool_indicator = f'<span class="tool-indicator">{tool_used}</span>' if tool_used else ''
        
        # Format the message for better readability
        formatted_message = format_response_text(message)
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>FIA Assistant:</strong> {tool_indicator}
            <div class="response-content">
        </div>
        """, unsafe_allow_html=True)
        
        # Display the formatted message with proper markdown in a container
        with st.container():
            st.markdown(formatted_message)
        
        # Close the response content div
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display metadata if available and show_metrics is enabled
        if metadata and st.session_state.get('show_metrics', False):
            with st.expander("📊 Response Details", expanded=False):
                if 'reasoning_steps_count' in metadata:
                    st.metric("Reasoning Steps", metadata['reasoning_steps_count'])
                if 'model' in metadata:
                    st.text(f"Model: {metadata['model']}")
                if 'timestamp' in metadata:
                    st.text(f"Timestamp: {metadata['timestamp']}")
                if 'error' in metadata:
                    st.error(f"Error: {metadata['error']}")

def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏎️ FIA Formula 1 Regulations Assistant</h1>
        <p>Your intelligent guide to FIA technical, sporting, financial, and operational regulations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Available Tools")
        
        # Initialize session state
        if 'agent' not in st.session_state:
            st.session_state.agent = initialize_agent()
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'show_metrics' not in st.session_state:
            st.session_state.show_metrics = False
        
        if st.session_state.agent:
            tools = st.session_state.agent.get_available_tools()
            st.markdown("**Specialized Functions:**")
            for tool in tools:
                st.markdown(f"• **{tool['name']}**: {tool['description']}")
            
            # Show tracing status
            langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
            if langsmith_api_key:
                st.success("🔍 LangSmith tracing enabled")
            else:
                st.info("💡 Set LANGSMITH_API_KEY to enable tracing")
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        st.session_state.show_metrics = st.checkbox("Show Response Metrics", value=False)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Export chat button
        if st.button("📥 Export Chat"):
            if st.session_state.chat_history:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"fia_chat_export_{timestamp}.json"
                
                with open(filename, 'w') as f:
                    json.dump(st.session_state.chat_history, f, indent=2)
                
                st.success(f"✅ Chat exported to {filename}")
            else:
                st.warning("No chat history to export")
    
    # Main chat interface
    if not st.session_state.agent:
        st.error("❌ Agent not initialized. Please check your environment variables.")
        st.stop()
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(
            message['content'], 
            is_user=message['is_user'],
            tool_used=message.get('tool_used'),
            metadata=message.get('metadata') if st.session_state.show_metrics else None
        )
    
    # Chat input
    st.markdown("---")
    
    # Example questions
    with st.expander("💡 Example Questions", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Technical Regulations:**")
            st.markdown("• What are the safety requirements for F1 cars?")
            st.markdown("• What is the power unit specification?")
            st.markdown("• Compare Article 5 between 2024 and 2025")
            
        with col2:
            st.markdown("**Sporting Regulations:**")
            st.markdown("• What are the penalties for track limits violations?")
            st.markdown("• How does qualifying work?")
            st.markdown("• What are the race start procedures?")
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask about FIA Formula 1 regulations:",
            placeholder="e.g., What are the safety requirements for Formula 1 cars?",
            key="user_input"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            send_button = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)
        
        with col2:
            regenerate_button = st.form_submit_button("🔄 Regenerate", use_container_width=True)
    
    # Handle regenerate button
    if regenerate_button:
        if st.session_state.chat_history:
            # Remove last assistant message and regenerate
            while (st.session_state.chat_history and 
                   not st.session_state.chat_history[-1]['is_user']):
                st.session_state.chat_history.pop()
            st.rerun()
    
    # Process user input
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'content': user_input,
            'is_user': True,
            'timestamp': datetime.now().isoformat()
        })
        
        # Display user message
        display_chat_message(user_input, is_user=True)
        
        # Get agent response
        with st.spinner("🤖 FIA Assistant is thinking..."):
            try:
                response = st.session_state.agent.query(user_input)
                
                # Extract information from response
                answer = response.get('answer', 'No answer generated')
                tools_used = response.get('tools_used', [])
                tool_used = tools_used[0] if tools_used else None
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'content': answer,
                    'is_user': False,
                    'tool_used': tool_used,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': response.get('metadata', {})
                })
                
                # Display assistant response
                display_chat_message(
                    answer, 
                    is_user=False,
                    tool_used=tool_used,
                    metadata=response.get('metadata', {})
                )
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                
                st.session_state.chat_history.append({
                    'content': error_msg,
                    'is_user': False,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Form automatically clears input on submit
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        🏎️ FIA Formula 1 Regulations Assistant | Powered by RAG + Agentic AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
