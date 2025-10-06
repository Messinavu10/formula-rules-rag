#!/usr/bin/env python3
"""
FIA Regulations RAG Demo Script

This script demonstrates the production-ready RAG pipeline for FIA Formula 1
regulations with advanced features like filtering, citations, and conversation history.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.query_interface import FIAQueryInterface
from rag.rag_pipeline import FIARAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the RAG demo."""
    # Validate configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "fia-rules")

    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return

    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        return

    print("🏎️  FIA Formula 1 Regulations RAG System")
    print("=" * 50)

    try:
        # Initialize RAG pipeline
        print("🔧 Initializing RAG pipeline...")
        rag_pipeline = FIARAGPipeline(
            index_name=index_name,
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            model_name="gpt-4",
        )

        # Initialize query interface
        query_interface = FIAQueryInterface(rag_pipeline)

        print("✅ RAG system initialized successfully!")

        # Demo queries
        demo_queries = [
            {
                "question": "What are the safety requirements for Formula 1 cars?",
                "description": "General safety requirements",
            },
            {
                "question": "What is the maximum engine power allowed in 2024?",
                "description": "2024 specific technical regulations",
                "year_filter": "2024",
                "regulation_type_filter": "technical",
            },
            {
                "question": "What are the rules for pit stops during a race?",
                "description": "Sporting regulations for pit stops",
            },
        ]

        print(f"\n🎯 Running {len(demo_queries)} demo queries...")
        print("=" * 50)

        for i, demo in enumerate(demo_queries, 1):
            print(f"\n📋 Demo Query #{i}: {demo['description']}")
            print("-" * 40)

            # Ask the question
            response = query_interface.ask_question(
                question=demo["question"],
                year_filter=demo.get("year_filter"),
                regulation_type_filter=demo.get("regulation_type_filter"),
            )

            # Display the response
            query_interface.display_response(response)

        # Interactive mode
        print(f"\n🎮 Starting interactive mode...")
        print("You can now ask your own questions about F1 regulations!")
        query_interface.interactive_mode()

    except Exception as e:
        logger.error(f"Error in RAG demo: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
