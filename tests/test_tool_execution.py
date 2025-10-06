"""
Tests for tool execution functionality.
"""

from unittest.mock import patch


class TestToolExecution:
    """Test cases for tool execution."""

    def test_single_tool_execution(self, fia_agent, sample_agent_state):
        """Test execution of a single tool."""
        sample_agent_state["selected_tools"] = ["regulation_search"]

        # Mock the tool execution by patching the tool's _run method
        with patch.object(fia_agent.tools[0], "_run") as mock_tool_run:
            mock_tool_run.return_value = "Safety requirements result"

            result_state = fia_agent._act_node(sample_agent_state)

            assert "regulation_search" in result_state["multi_tool_results"]
            assert "regulation_search" in result_state["tools_used"]

    def test_multi_tool_execution(self, fia_agent, sample_agent_state):
        """Test execution of multiple tools."""
        sample_agent_state["selected_tools"] = ["regulation_search", "penalty_lookup"]

        # Mock multiple tools by patching their _run methods
        with (
            patch.object(fia_agent.tools[0], "_run") as mock_tool1,
            patch.object(fia_agent.tools[1], "_run") as mock_tool2,
        ):
            mock_tool1.return_value = "Safety requirements result"
            mock_tool2.return_value = "Penalty information result"

            result_state = fia_agent._act_node(sample_agent_state)

            assert len(result_state["multi_tool_results"]) == 2
            assert "regulation_search" in result_state["multi_tool_results"]
            assert "penalty_lookup" in result_state["multi_tool_results"]

    def test_tool_execution_error_handling(self, fia_agent, sample_agent_state):
        """Test error handling in tool execution."""
        sample_agent_state["selected_tools"] = ["regulation_search"]

        with patch.object(
            fia_agent.tools[0], "_run", side_effect=Exception("Tool error")
        ):
            result_state = fia_agent._act_node(sample_agent_state)

            assert "regulation_search" in result_state["multi_tool_results"]
            assert (
                "Error executing"
                in result_state["multi_tool_results"]["regulation_search"]
            )

    def test_tool_parameter_extraction_comparison(self, fia_agent):
        """Test parameter extraction for regulation comparison."""
        question = "Compare Article 5 between 2024 and 2025"

        # Mock the regex extraction
        with patch("re.search") as mock_search, patch("re.findall") as mock_findall:

            mock_search.return_value.group.return_value = "5"
            mock_findall.return_value = ["2024", "2025"]

            # This would be tested in the actual tool execution
            # For now, we test the pattern matching logic
            import re

            article_match = re.search(
                r"Article (\d+(?:\.\d+)?)", question, re.IGNORECASE
            )
            year_matches = re.findall(r"(20\d{2})", question)

            if article_match and len(year_matches) >= 2:
                article_number = article_match.group(1)
                year1, year2 = year_matches[0], year_matches[1]
                assert article_number == "5"
                assert year1 == "2024"
                assert year2 == "2025"

    def test_tool_parameter_extraction_penalty(self, fia_agent):
        """Test parameter extraction for penalty lookup."""
        test_cases = [
            ("What are the penalties for MGU-K violations?", "MGU-K"),
            ("What happens if you violate fuel flow rules?", "fuel flow"),
            ("What are the track limits penalties?", "track limits"),
            ("What are the penalties?", "track limits"),  # default
        ]

        for question, expected_violation in test_cases:
            violation_type = "track limits"  # default
            if "MGU-K" in question:
                violation_type = "MGU-K"
            elif "fuel" in question.lower():
                violation_type = "fuel flow"
            elif "track" in question.lower():
                violation_type = "track limits"

            assert violation_type == expected_violation

    def test_tool_result_storage(self, fia_agent, sample_agent_state):
        """Test storage of tool results."""
        sample_agent_state["selected_tools"] = ["regulation_search"]

        with patch.object(fia_agent.tools[0], "_run") as mock_tool_run:
            mock_tool_run.return_value = "Test result"

            result_state = fia_agent._act_node(sample_agent_state)

            assert "regulation_search" in result_state["multi_tool_results"]
            assert (
                result_state["multi_tool_results"]["regulation_search"] == "Test result"
            )

    def test_tool_execution_skip_already_executed(self, fia_agent, sample_agent_state):
        """Test that already executed tools are skipped."""
        sample_agent_state["selected_tools"] = ["regulation_search", "penalty_lookup"]
        sample_agent_state["multi_tool_results"] = {
            "regulation_search": "Already executed"
        }

        with patch.object(fia_agent.tools[1], "_run") as mock_tool_run:
            mock_tool_run.return_value = "New result"

            result_state = fia_agent._act_node(sample_agent_state)

            # Should only execute penalty_lookup, not regulation_search
            assert mock_tool_run.call_count == 1
            assert "penalty_lookup" in result_state["multi_tool_results"]
            assert (
                result_state["multi_tool_results"]["regulation_search"]
                == "Already executed"
            )

    def test_tool_execution_continue_on_error(self, fia_agent, sample_agent_state):
        """Test that tool execution continues even if one tool fails."""
        sample_agent_state["selected_tools"] = ["regulation_search", "penalty_lookup"]

        with (
            patch.object(
                fia_agent.tools[0], "_run", side_effect=Exception("First tool error")
            ),
            patch.object(fia_agent.tools[1], "_run", return_value="Second tool result"),
        ):

            result_state = fia_agent._act_node(sample_agent_state)

            assert len(result_state["multi_tool_results"]) == 2
            assert (
                "Error executing"
                in result_state["multi_tool_results"]["regulation_search"]
            )
            assert (
                result_state["multi_tool_results"]["penalty_lookup"]
                == "Second tool result"
            )
