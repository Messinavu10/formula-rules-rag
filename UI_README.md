# 🏎️ FIA Formula 1 Regulations Chatbot UI

A modern, interactive web interface for your FIA agentic RAG system.

## 🚀 Quick Start

### Option 1: Using the launcher script
```bash
python run_ui.py
```

### Option 2: Direct Streamlit command
```bash
streamlit run streamlit_app.py
```

## 🎯 Features

### 💬 **Interactive Chat Interface**
- Clean, modern chat UI with F1-themed styling
- Real-time conversation with the FIA agent
- Chat history persistence during session
- Export chat conversations to JSON

### 🛠️ **Smart Tool Selection**
- Automatic tool selection based on query intent
- Visual indicators showing which tool was used
- Support for all specialized functions:
  - `regulation_search`: Find specific regulations
  - `regulation_comparison`: Compare regulations across years
  - `penalty_lookup`: Find penalty information
  - `regulation_summary`: Summarize complex regulations
  - `general_rag`: General Q&A capabilities
  - `out_of_scope`: Handle non-F1 queries

### 📊 **Advanced Features**
- Response metrics display (optional)
- Chat regeneration capability
- Example questions to get started
- Tool information in sidebar
- Settings panel for customization

### 🎨 **Modern Design**
- F1-themed color scheme (red, white, black)
- Responsive layout
- Clean typography
- Intuitive navigation

## 🔧 **Configuration**

Make sure you have the following environment variables set:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"
export PINECONE_INDEX_NAME="fia-rules"  # optional, defaults to "fia-rules"
```

## 📱 **Usage**

1. **Start the UI**: Run `python run_ui.py`
2. **Open Browser**: Navigate to `http://localhost:8501`
3. **Ask Questions**: Type your F1 regulation questions
4. **Explore Tools**: Check the sidebar for available functions
5. **Export Chats**: Save conversations for later reference

## 💡 **Example Questions**

### Technical Regulations
- "What are the safety requirements for F1 cars?"
- "What is the power unit specification?"
- "Compare Article 5 between 2024 and 2025"

### Sporting Regulations
- "What are the penalties for track limits violations?"
- "How does qualifying work?"
- "What are the race start procedures?"

### Financial & Operational
- "What are the cost cap regulations?"
- "How are operational procedures defined?"

## 🛠️ **Troubleshooting**

### Common Issues:

1. **"Agent not initialized"**
   - Check your environment variables
   - Ensure Pinecone index exists
   - Verify API keys are correct

2. **"Streamlit not found"**
   - Install with: `pip install streamlit`
   - Or use the launcher script which auto-installs

3. **Port already in use**
   - Change port: `streamlit run streamlit_app.py --server.port 8502`

## 📊 **Integration with Evaluation**

The UI works seamlessly with your evaluation system:
- Use `python scripts/evaluate_agent.py` for comprehensive RAGAS metrics
- Use `python scripts/quick_evaluation.py` for quick performance tests
- Export chat data for analysis

## 🎨 **Customization**

The UI is built with Streamlit and includes:
- Custom CSS for F1 theming
- Responsive design
- Modular components
- Easy to extend with new features

## 🔗 **Related Files**

- `streamlit_app.py`: Main UI application
- `run_ui.py`: Launcher script
- `src/rag/agent.py`: FIA agent implementation
- `src/rag/rag_pipeline.py`: RAG pipeline
- `scripts/evaluate_agent.py`: Evaluation system

---

**🏎️ Enjoy your FIA Formula 1 Regulations Assistant!**
