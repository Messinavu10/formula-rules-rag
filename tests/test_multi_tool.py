"""
Tests for multi-tool orchestration functionality.
"""

from unittest.mock import Mock, patch


class TestMultiToolOrchestration:
    """Test cases for multi-tool orchestration."""

    def test_multi_tool_intent_detection(self, fia_agent):
        """Test detection of multi-tool scenarios."""
        multi_tool_questions = [
            "What are the safety requirements and penalties?",
            "Compare regulations and summarize the changes",
            "Find engine specs and penalty information",
        ]

        for question in multi_tool_questions:
            with patch.object(fia_agent.llm, "invoke") as mock_invoke:
                mock_response = Mock()
                mock_response.content = "MULTI_TOOL"
                mock_invoke.return_value = mock_response

                intent = fia_agent._classify_intent(question)
                assert intent == "MULTI_TOOL"

    def test_multi_tool_selection(self, fia_agent):
        """Test selection of multiple tools."""
        question = "What are the safety requirements and penalties?"

        with patch.object(fia_agent.llm, "invoke") as mock_invoke:
            mock_response = Mock()
            mock_response.content = '["regulation_search", "penalty_lookup"]'
            mock_invoke.return_value = mock_response

            tools = fia_agent._select_multi_tools(question)
            assert len(tools) == 2
            assert "regulation_search" in tools
            assert "penalty_lookup" in tools

    def test_reason_node_multi_tool(self, fia_agent, sample_agent_state):
        """Test reasoning node with multi-tool selection."""
        with (
            patch.object(fia_agent, "_classify_intent", return_value="MULTI_TOOL"),
            patch.object(
                fia_agent,
                "_select_multi_tools",
                return_value=["regulation_search", "penalty_lookup"],
            ),
        ):

            result_state = fia_agent._reason_node(sample_agent_state)

            assert "selected_tools" in result_state
            assert result_state["selected_tools"] == [
                "regulation_search",
                "penalty_lookup",
            ]
            assert "MULTI_TOOL" in str(result_state["reasoning_steps"])

    def test_reason_node_single_tool(self, fia_agent, sample_agent_state):
        """Test reasoning node with single tool selection."""
        with (
            patch.object(fia_agent, "_classify_intent", return_value="SEARCH"),
            patch.object(fia_agent, "_select_tool", return_value="regulation_search"),
        ):

            result_state = fia_agent._reason_node(sample_agent_state)

            assert "selected_tools" in result_state
            assert result_state["selected_tools"] == ["regulation_search"]
            assert "SEARCH" in str(result_state["reasoning_steps"])

    def test_act_node_multi_tool_execution(self, fia_agent, sample_agent_state):
        """Test action node with multiple tools."""
        sample_agent_state["selected_tools"] = ["regulation_search", "penalty_lookup"]

        # Mock the tool execution by patching the tools themselves
        with (
            patch.object(
                fia_agent.tools[0], "_run", return_value="Safety requirements result"
            ),
            patch.object(
                fia_agent.tools[1], "_run", return_value="Penalty information result"
            ),
        ):

            result_state = fia_agent._act_node(sample_agent_state)

            assert len(result_state["multi_tool_results"]) == 2
            assert "regulation_search" in result_state["multi_tool_results"]
            assert "penalty_lookup" in result_state["multi_tool_results"]
            assert "regulation_search" in result_state["tools_used"]
            assert "penalty_lookup" in result_state["tools_used"]

    def test_act_node_single_tool_execution(self, fia_agent, sample_agent_state):
        """Test action node with single tool."""
        sample_agent_state["selected_tools"] = ["regulation_search"]

        # Mock the tool execution by patching the tool itself
        with patch.object(
            fia_agent.tools[0], "_run", return_value="Safety requirements result"
        ):
            result_state = fia_agent._act_node(sample_agent_state)

            assert "regulation_search" in result_state["multi_tool_results"]
            assert (
                result_state["multi_tool_results"]["regulation_search"]
                == "Safety requirements result"
            )
            assert "regulation_search" in result_state["tools_used"]

    def test_act_node_no_tools_selected(self, fia_agent, sample_agent_state):
        """Test action node with no tools selected."""
        sample_agent_state["selected_tools"] = []

        result_state = fia_agent._act_node(sample_agent_state)

        assert "Error: No tools selected" in result_state["reasoning_steps"]

    def test_act_node_tool_execution_error(self, fia_agent, sample_agent_state):
        """Test error handling in tool execution."""
        sample_agent_state["selected_tools"] = ["regulation_search"]

        # Mock the tool execution to raise an error
        with patch.object(
            fia_agent.tools[0], "_run", side_effect=Exception("Tool error")
        ):
            result_state = fia_agent._act_node(sample_agent_state)

            assert "regulation_search" in result_state["multi_tool_results"]
            assert (
                "Error executing"
                in result_state["multi_tool_results"]["regulation_search"]
            )

    def test_result_combination_multi_tool(self, fia_agent):
        """Test result combination for multiple tools."""
        results = {
            "regulation_search": "Safety requirements: Fire extinguishers, safety harnesses...",
            "penalty_lookup": "Penalties: 5-second penalty, drive-through penalty...",
        }

        with patch.object(fia_agent.llm, "invoke") as mock_invoke:
            mock_response = Mock()
            mock_response.content = "Combined comprehensive answer with both safety requirements and penalties..."
            mock_invoke.return_value = mock_response

            combined = fia_agent._combine_multi_tool_results(results, "Test question")
            assert "safety requirements" in combined.lower()
            assert "penalties" in combined.lower()

    def test_multi_tool_workflow_integration(self, fia_agent):
        """Test complete multi-tool workflow."""
        question = "What are the safety requirements and penalties?"

        with (
            patch.object(fia_agent, "_classify_intent", return_value="MULTI_TOOL"),
            patch.object(
                fia_agent,
                "_select_multi_tools",
                return_value=["regulation_search", "penalty_lookup"],
            ),
            patch.object(fia_agent, "_act_node") as mock_act,
            patch.object(fia_agent, "_reflect_node") as mock_reflect,
        ):

            mock_act.return_value = {
                "tool_result": "Combined safety and penalty information",
                "tools_used": ["regulation_search", "penalty_lookup"],
                "multi_tool_results": {
                    "regulation_search": "Safety requirements result",
                    "penalty_lookup": "Penalty information result",
                },
            }
            mock_reflect.return_value = {
                "final_answer": "Comprehensive answer combining safety requirements and penalties"
            }

            response = fia_agent.query(question)

            assert "answer" in response
            assert len(response["tools_used"]) == 2
            assert "regulation_search" in response["tools_used"]
            assert "penalty_lookup" in response["tools_used"]
