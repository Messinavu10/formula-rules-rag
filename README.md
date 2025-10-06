# 🏎️ FIA Formula 1 Regulations RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system for FIA Formula 1 regulations, powered by LangChain, LangGraph, and OpenAI. This system provides intelligent querying, comparison, and analysis of FIA regulations across multiple years and types.

## 🌟 Features

### 🤖 **Intelligent Agent System**
- **Multi-tool Reasoning**: LangGraph-powered agent with specialized tools
- **State Management**: Persistent conversation state and reasoning tracking
- **Tool Selection**: Automatic tool selection based on query intent
- **Multi-step Analysis**: Complex queries broken down into reasoning steps

### 🔍 **Advanced RAG Pipeline**
- **Semantic Search**: Vector-based retrieval using Pinecone
- **Context-Aware**: Intelligent context selection and ranking
- **Source Attribution**: Complete source tracking and citations
- **Multi-year Support**: Regulations from 2024, 2025, and 2026

### 🛠️ **Specialized Tools**
- **Regulation Search**: Find specific rules and articles
- **Regulation Comparison**: Compare rules across different years
- **Penalty Lookup**: Search for penalties and sanctions
- **Regulation Summary**: Comprehensive topic summaries
- **General RAG**: Handle any regulation-related questions

### 📊 **Comprehensive Evaluation**
- **RAGAS Metrics**: Context relevance, precision, recall, faithfulness
- **Automated Testing**: CI/CD pipeline with quality checks
- **Performance Monitoring**: Detailed evaluation reports

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Sources   │───▶│  Document Store  │───▶│  Vector Store   │
│  (FIA Regs)     │    │   (Processed)    │    │   (Pinecone)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│   FIA Agent      │───▶│  RAG Pipeline   │
│ (User Interface)│    │  (LangGraph)     │    │  (Retrieval)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
formula-rules-rag/
├── 📁 data/                          # FIA regulation PDFs
│   ├── 2024/                        # 2024 regulations
│   ├── 2025/                        # 2025 regulations
│   └── 2026/                        # 2026 regulations
├── 📁 src/                          # Source code
│   ├── rag/                         # RAG system components
│   │   ├── agent.py                 # LangGraph agent
│   │   ├── rag_pipeline.py          # RAG pipeline
│   │   ├── retriever.py             # Vector retrieval
│   │   ├── tools.py                 # Specialized tools
│   │   └── query_interface.py        # Query interface
│   └── evaluation/                  # Evaluation framework
│       ├── evaluator.py             # RAGAS evaluator
│       └── dataset.py               # Test datasets
├── 📁 scripts/                      # Utility scripts
│   ├── ingest_pdfs_langchain.py     # PDF processing
│   ├── vectorize_data.py            # Vector store setup
│   ├── agent_demo.py                # Agent demonstration
│   └── evaluate_agent.py            # Evaluation runner
├── 📁 tests/                        # Test suite
├── streamlit_app.py                 # Streamlit UI
├── run_ui.py                        # UI launcher
└── requirements.txt                 # Dependencies
```

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/formula-rules-rag.git
cd formula-rules-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Environment Setup**

Create a `.env` file with your API keys:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Pinecone API
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=fia-rules-test

# LangSmith (optional, for tracing)
LANGSMITH_API_KEY=your_langsmith_api_key
```

### 3. **Data Processing**

```bash
# Process PDFs and create vector store
python scripts/ingest_pdfs_langchain.py
python scripts/vectorize_data.py
```

### 4. **Launch the Application**

```bash
# Option 1: Use the launcher script
python run_ui.py

# Option 2: Direct Streamlit launch
streamlit run streamlit_app.py
```

## 🛠️ Usage

### **Web Interface**
- Open your browser to `http://localhost:8501`
- Ask questions about FIA regulations
- Get detailed answers with source citations
- Compare regulations across years

### **Example Queries**
- "What are the track limits penalties for 2024?"
- "Compare the technical regulations between 2024 and 2025"
- "What are the penalties for MGU-K violations?"
- "Summarize the financial regulations for 2026"

### **Programmatic Usage**

```python
from src.rag.agent import FIAAgent
from src.rag.rag_pipeline import FIARAGPipeline

# Initialize the agent
agent = FIAAgent(
    index_name="fia-rules-test",
    openai_api_key="your_key",
    pinecone_api_key="your_key"
)

# Ask a question
response = agent.invoke("What are the track limits rules for 2024?")
print(response)
```

## 🔧 Development

### **Running Tests**

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_agent_core.py
pytest tests/test_tool_execution.py
pytest tests/test_performance.py
```

### **Code Quality**

```bash
# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Lint code
flake8 src/ scripts/ tests/

# Security check
bandit -r src/ scripts/
safety check
```

### **Evaluation**

```bash
# Run comprehensive evaluation
python scripts/evaluate_agent.py

# Quick evaluation
python scripts/quick_evaluation.py
```

## 📊 Evaluation Metrics

The system uses RAGAS (RAG Assessment) framework for comprehensive evaluation:

- **Context Relevance**: How relevant are retrieved documents
- **Context Precision**: Precision of retrieved context
- **Context Recall**: Recall of relevant information
- **Answer Relevancy**: Relevance of generated answers
- **Faithfulness**: Factual accuracy of responses

## 🏎️ FIA Regulation Coverage

### **Years Covered**
- **2024**: Sporting, Technical regulations
- **2025**: Sporting, Technical regulations  
- **2026**: Sporting, Technical, Financial, Operational regulations

### **Regulation Types**
- **Sporting Regulations**: Race procedures, penalties, safety
- **Technical Regulations**: Car specifications, safety systems
- **Financial Regulations**: Budget caps, spending limits
- **Operational Regulations**: Team operations, logistics

## 🔄 CI/CD Pipeline

The project includes a comprehensive CI/CD pipeline with:

- **Code Quality**: Black, isort, flake8, bandit, safety
- **Testing**: Unit tests, integration tests, performance tests
- **Evaluation**: Automated RAGAS evaluation
- **Security**: Vulnerability scanning with Trivy
- **Documentation**: Docstring validation
- **Deployment**: Automated releases and artifact creation

---