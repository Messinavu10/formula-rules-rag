"""
Tests for intent classification functionality.
"""
import pytest
from unittest.mock import Mock, patch

class TestIntentClassification:
    """Test cases for intent classification."""
    
    def test_single_tool_intents(self, fia_agent):
        """Test classification of single-tool intents."""
        test_cases = [
            ("What are the safety requirements?", "SEARCH"),
            ("What are the penalties for track limits?", "PENALTY"),
            ("Compare Article 5 between 2024 and 2025", "COMPARISON"),
            ("Summarize the technical regulations", "SUMMARY"),
            ("Explain the FIA rules", "GENERAL")
        ]
        
        for question, expected_intent in test_cases:
            with patch.object(fia_agent.llm, 'invoke') as mock_invoke:
                mock_response = Mock()
                mock_response.content = expected_intent
                mock_invoke.return_value = mock_response
                
                intent = fia_agent._classify_intent(question)
                assert intent == expected_intent
    
    def test_multi_tool_intents(self, fia_agent):
        """Test classification of multi-tool intents."""
        multi_tool_questions = [
            "What are the safety requirements and penalties?",
            "Compare regulations and summarize the changes",
            "Find engine specs and penalty information",
            "What are the requirements and what happens if you violate them?"
        ]
        
        for question in multi_tool_questions:
            with patch.object(fia_agent.llm, 'invoke') as mock_invoke:
                mock_response = Mock()
                mock_response.content = "MULTI_TOOL"
                mock_invoke.return_value = mock_response
                
                intent = fia_agent._classify_intent(question)
                assert intent == "MULTI_TOOL"
    
    def test_out_of_scope_detection(self, fia_agent):
        """Test detection of out-of-scope questions."""
        out_of_scope_questions = [
            "What is the weather today?",
            "How do I cook pasta?",
            "What are the stock prices?",
            "Tell me about history"
        ]
        
        for question in out_of_scope_questions:
            with patch.object(fia_agent.llm, 'invoke') as mock_invoke:
                mock_response = Mock()
                mock_response.content = "OUT_OF_SCOPE"
                mock_invoke.return_value = mock_response
                
                intent = fia_agent._classify_intent(question)
                assert intent == "OUT_OF_SCOPE"
    
    def test_intent_classification_error_handling(self, fia_agent):
        """Test error handling in intent classification."""
        with patch.object(fia_agent.llm, 'invoke', side_effect=Exception("LLM Error")):
            intent = fia_agent._classify_intent("Test question")
            assert intent == "GENERAL"  # Default fallback
    
    def test_invalid_intent_fallback(self, fia_agent):
        """Test fallback for invalid intent responses."""
        with patch.object(fia_agent.llm, 'invoke') as mock_invoke:
            mock_response = Mock()
            mock_response.content = "INVALID_INTENT"
            mock_invoke.return_value = mock_response
            
            intent = fia_agent._classify_intent("Test question")
            assert intent == "GENERAL"  # Default fallback
    
    def test_intent_classification_prompt_structure(self, fia_agent):
        """Test that the intent classification prompt is properly structured."""
        question = "What are the safety requirements?"
        
        with patch.object(fia_agent.llm, 'invoke') as mock_invoke:
            mock_response = Mock()
            mock_response.content = "SEARCH"
            mock_invoke.return_value = mock_response
            
            fia_agent._classify_intent(question)
            
            # Verify the prompt was called with proper structure
            call_args = mock_invoke.call_args
            messages = call_args[0][0]
            
            assert len(messages) == 2  # SystemMessage and HumanMessage
            assert "Classify this FIA regulation question" in messages[0].content
            assert question in messages[1].content
    
    def test_intent_classification_valid_intents(self, fia_agent):
        """Test that all valid intents are recognized."""
        valid_intents = ["COMPARISON", "PENALTY", "SEARCH", "SUMMARY", "GENERAL", "MULTI_TOOL", "OUT_OF_SCOPE"]
        
        for intent in valid_intents:
            with patch.object(fia_agent.llm, 'invoke') as mock_invoke:
                mock_response = Mock()
                mock_response.content = intent
                mock_invoke.return_value = mock_response
                
                result = fia_agent._classify_intent("Test question")
                assert result == intent