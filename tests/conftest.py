"""
Pytest configuration and fixtures for FIA Agent tests.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path - more robust approach
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Also add the project root to path
sys.path.insert(0, str(project_root))

# Now import the modules
from rag.agent import AgentState, FIAAgent
from rag.rag_pipeline import FIARAGPipeline


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline for testing."""
    mock_pipeline = Mock(spec=FIARAGPipeline)
    mock_pipeline.query.return_value = {
        "answer": "Test regulation answer",
        "sources": ["test.pdf"],
        "citations": ["Test citation"],
        "retrieved_documents": [
            {"content": "Test regulation content", "metadata": {"source": "test.pdf"}}
        ],
        "metadata": {"retrieved_docs": 1},
    }
    return mock_pipeline


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_llm.invoke.return_value = mock_response
    return mock_llm


@pytest.fixture
def sample_agent_state():
    """Sample agent state for testing."""
    return AgentState(
        messages=[],
        current_question="What are the safety requirements?",
        reasoning_steps=[],
        tools_used=[],
        selected_tools=[],
        final_answer=None,
        sources=[],
        session_id="test_session",
        tool_result=None,
        multi_tool_results={},
    )


@pytest.fixture
def mock_tools():
    """Mock tools for testing."""
    tools = []
    for tool_name in ["regulation_search", "penalty_lookup", "regulation_summary"]:
        mock_tool = Mock()
        mock_tool.name = tool_name
        mock_tool.description = f"Mock {tool_name} tool"
        mock_tool._run.return_value = f"Mock result from {tool_name}"
        tools.append(mock_tool)
    return tools


@pytest.fixture
def fia_agent(mock_rag_pipeline, mock_llm, mock_tools):
    """FIA Agent instance for testing."""
    with patch("rag.agent.ChatOpenAI", return_value=mock_llm):
        agent = FIAAgent(
            rag_pipeline=mock_rag_pipeline,
            model_name="gpt-4o-mini",
            enable_tracing=False,
        )
        # Mock the tools
        agent.tools = mock_tools
        return agent


@pytest.fixture
def test_questions():
    """Test questions for different scenarios."""
    return {
        "single_tool": "What are the safety requirements for F1 cars?",
        "multi_tool": "What are the safety requirements and penalties for violations?",
        "comparison": "Compare Article 5 between 2024 and 2025",
        "penalty": "What are the penalties for track limits violations?",
        "out_of_scope": "What is the weather today?",
    }
