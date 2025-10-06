"""
Tests for agent state management.
"""
import pytest
from rag.agent import AgentState

class TestAgentState:
    """Test cases for agent state management."""
    
    def test_agent_state_initialization(self):
        """Test agent state initialization."""
        state = AgentState(
            messages=[],
            current_question="Test question",
            reasoning_steps=[],
            tools_used=[],
            selected_tools=[],
            final_answer=None,
            sources=[],
            session_id="test_session",
            tool_result=None,
            multi_tool_results={}
        )
        
        assert state["current_question"] == "Test question"
        assert state["session_id"] == "test_session"
        assert isinstance(state["reasoning_steps"], list)
        assert isinstance(state["tools_used"], list)
        assert isinstance(state["selected_tools"], list)
        assert isinstance(state["multi_tool_results"], dict)
    
    def test_agent_state_reasoning_steps(self):
        """Test reasoning steps management."""
        state = AgentState(
            messages=[],
            current_question="Test question",
            reasoning_steps=[],
            tools_used=[],
            selected_tools=[],
            final_answer=None,
            sources=[],
            session_id="test_session",
            tool_result=None,
            multi_tool_results={}
        )
        
        # Add reasoning steps
        state["reasoning_steps"].append("Step 1: Intent classification")
        state["reasoning_steps"].append("Step 2: Tool selection")
        
        assert len(state["reasoning_steps"]) == 2
        assert "Intent classification" in state["reasoning_steps"][0]
        assert "Tool selection" in state["reasoning_steps"][1]
    
    def test_agent_state_tools_used(self):
        """Test tools used tracking."""
        state = AgentState(
            messages=[],
            current_question="Test question",
            reasoning_steps=[],
            tools_used=[],
            selected_tools=[],
            final_answer=None,
            sources=[],
            session_id="test_session",
            tool_result=None,
            multi_tool_results={}
        )
        
        # Add tools used
        state["tools_used"].append("regulation_search")
        state["tools_used"].append("penalty_lookup")
        
        assert len(state["tools_used"]) == 2
        assert "regulation_search" in state["tools_used"]
        assert "penalty_lookup" in state["tools_used"]
    
    def test_agent_state_multi_tool_results(self):
        """Test multi-tool results storage."""
        state = AgentState(
            messages=[],
            current_question="Test question",
            reasoning_steps=[],
            tools_used=[],
            selected_tools=[],
            final_answer=None,
            sources=[],
            session_id="test_session",
            tool_result=None,
            multi_tool_results={}
        )
        
        # Add multi-tool results
        state["multi_tool_results"]["regulation_search"] = "Safety requirements result"
        state["multi_tool_results"]["penalty_lookup"] = "Penalty information result"
        
        assert len(state["multi_tool_results"]) == 2
        assert state["multi_tool_results"]["regulation_search"] == "Safety requirements result"
        assert state["multi_tool_results"]["penalty_lookup"] == "Penalty information result"
    
    def test_agent_state_final_answer(self):
        """Test final answer storage."""
        state = AgentState(
            messages=[],
            current_question="Test question",
            reasoning_steps=[],
            tools_used=[],
            selected_tools=[],
            final_answer=None,
            sources=[],
            session_id="test_session",
            tool_result=None,
            multi_tool_results={}
        )
        
        # Set final answer
        state["final_answer"] = "This is the final answer"
        
        assert state["final_answer"] == "This is the final answer"
    
    def test_agent_state_sources(self):
        """Test sources tracking."""
        state = AgentState(
            messages=[],
            current_question="Test question",
            reasoning_steps=[],
            tools_used=[],
            selected_tools=[],
            final_answer=None,
            sources=[],
            session_id="test_session",
            tool_result=None,
            multi_tool_results={}
        )
        
        # Add sources
        state["sources"].append("Article 12: Safety Requirements")
        state["sources"].append("Article 14: Safety Equipment")
        
        assert len(state["sources"]) == 2
        assert "Article 12" in state["sources"][0]
        assert "Article 14" in state["sources"][1]