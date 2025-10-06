"""
Interactive Query Interface for FIA Regulations

This module provides a user-friendly interface for querying FIA regulations
with advanced features like filtering, conversation history, and formatted output.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .rag_pipeline import FIARAGPipeline

logger = logging.getLogger(__name__)


class FIAQueryInterface:
    """
    Interactive query interface for FIA regulations with advanced features.
    """

    def __init__(self, rag_pipeline: FIARAGPipeline):
        """
        Initialize the query interface.

        Args:
            rag_pipeline: Initialized RAG pipeline
        """
        self.rag_pipeline = rag_pipeline
        self.conversation_history = []

        logger.info("✅ Query interface initialized")

    def ask_question(
        self,  # calls the rag pipeline
        question: str,
        year_filter: Optional[str] = None,
        regulation_type_filter: Optional[str] = None,
        use_compression: bool = False,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Ask a question with optional filtering and formatting.

        Args:
            question: The question to ask
            year_filter: Filter by specific year
            regulation_type_filter: Filter by regulation type
            use_compression: Whether to use compressed retrieval
            include_sources: Whether to include source information

        Returns:
            Formatted response with answer, sources, and metadata
        """
        # Get answer from RAG pipeline
        result = self.rag_pipeline.query(
            question=question,
            year_filter=year_filter,
            regulation_type_filter=regulation_type_filter,
            use_compression=use_compression,
        )

        # Add to conversation history
        self.conversation_history.append(
            {
                "question": question,
                "answer": result["answer"],
                "timestamp": datetime.now().isoformat(),
                "filters": {
                    "year": year_filter,
                    "regulation_type": regulation_type_filter,
                },
            }
        )

        # Format response
        formatted_response = self._format_response(result, include_sources)

        return formatted_response

    def ask_followup(
        self, question: str, use_compression: bool = False
    ) -> Dict[str, Any]:
        """
        Ask a follow-up question using conversation history.

        Args:
            question: Follow-up question
            use_compression: Whether to use compressed retrieval

        Returns:
            Formatted response with context from previous questions
        """
        result = self.rag_pipeline.query_with_followup(
            question=question,
            conversation_history=self.conversation_history[-3:],  # Last 3 exchanges
            k=5,
        )

        # Add to conversation history
        self.conversation_history.append(
            {
                "question": question,
                "answer": result["answer"],
                "timestamp": datetime.now().isoformat(),
                "followup": True,
            }
        )

        formatted_response = self._format_response(result, include_sources=True)
        return formatted_response

    def _format_response(
        self, result: Dict[str, Any], include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Format the response for display.

        Args:
            result: Raw result from RAG pipeline
            include_sources: Whether to include source information

        Returns:
            Formatted response dictionary
        """
        formatted = {
            "answer": result["answer"],
            "metadata": result["metadata"],
            "timestamp": datetime.now().isoformat(),
        }

        if include_sources and result.get("sources"):
            formatted["sources"] = result["sources"]
            formatted["citations"] = result.get("citations", [])
            formatted["source_count"] = len(result["sources"])

        return formatted

    def display_response(self, response: Dict[str, Any], show_sources: bool = True):
        """
        Display a formatted response to the user.

        Args:
            response: Formatted response dictionary
            show_sources: Whether to show source information
        """
        print("\n" + "=" * 80)
        print("🏎️  FIA Formula 1 Regulations Query Response")
        print("=" * 80)

        # Display answer
        print(f"\n💬 Answer:")
        print("-" * 40)
        print(response["answer"])

        # Display sources if requested
        if show_sources and response.get("sources"):
            print(f"\n📚 Sources ({response.get('source_count', 0)} documents):")
            print("-" * 40)
            for i, source in enumerate(response["sources"], 1):
                print(f"{i}. {source}")

        # Display citations if available
        if response.get("citations"):
            print(f"\n📖 Citations:")
            print("-" * 40)
            for i, citation in enumerate(response["citations"], 1):
                print(f"{i}. {citation}")

        # Display metadata
        metadata = response.get("metadata", {})
        if metadata:
            print(f"\n📊 Query Info:")
            print("-" * 40)
            print(f"Documents retrieved: {metadata.get('retrieved_docs', 0)}")
            print(f"Model: {metadata.get('model', 'Unknown')}")
            print(f"Index: {metadata.get('index', 'Unknown')}")

        print("\n" + "=" * 80)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history

    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def export_conversation(self, filepath: str):
        """
        Export conversation history to a JSON file.

        Args:
            filepath: Path to save the conversation
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Conversation exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting conversation: {str(e)}")

    def get_available_filters(self) -> Dict[str, List[str]]:
        """Get available filter options."""
        return self.rag_pipeline.get_available_filters()

    def interactive_mode(self):
        """
        Start interactive query mode with command-line interface.
        """
        print("🏎️  FIA Formula 1 Regulations Interactive Query System")
        print("=" * 60)
        print("Type 'help' for commands, 'quit' to exit")

        while True:
            try:
                print(f"\n💬 Question #{len(self.conversation_history) + 1}:")
                user_input = input("> ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == "help":
                    self._show_help()
                    continue

                if user_input.lower() == "history":
                    self._show_history()
                    continue

                if user_input.lower() == "clear":
                    self.clear_conversation()
                    print("✅ Conversation history cleared")
                    continue

                if not user_input:
                    continue

                # Process the question
                response = self.ask_question(user_input)
                self.display_response(response)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    def _show_help(self):
        """Show help information."""
        print("\n📖 Available Commands:")
        print("-" * 30)
        print("help     - Show this help message")
        print("history  - Show conversation history")
        print("clear    - Clear conversation history")
        print("quit     - Exit the program")
        print("\n💡 Tips:")
        print("- Ask specific questions about F1 regulations")
        print("- Use follow-up questions for clarification")
        print("- The system remembers conversation context")

    def _show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet")
            return

        print(
            f"\n📝 Conversation History ({len(self.conversation_history)} exchanges):"
        )
        print("-" * 50)

        for i, exchange in enumerate(self.conversation_history, 1):
            print(f"\n{i}. Q: {exchange['question']}")
            print(f"   A: {exchange['answer'][:100]}...")
            print(f"   Time: {exchange['timestamp']}")
