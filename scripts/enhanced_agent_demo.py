#!/usr/bin/env python3
"""
Enhanced FIA Regulations Agent Demo Script

This script demonstrates the agentic RAG system with intent classification,
hierarchical fallback, and comprehensive tool selection.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.agent import FIAAgent
from rag.rag_pipeline import FIARAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the enhanced agent demo."""
    # Validate configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "fia-rules")
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return

    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        return

    print("🏎️  Enhanced FIA Formula 1 Regulations Agent System")
    print("=" * 70)
    print("🤖 Intent Classification + Hierarchical Fallback")
    print("📊 LangSmith Tracing Enabled")
    print("🛠️  6 Specialized Tools Available")
    print("=" * 70)

    try:
        # Initialize RAG pipeline
        print("🔧 Initializing RAG pipeline...")
        rag_pipeline = FIARAGPipeline(
            index_name=index_name,
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            model_name="gpt-4o-mini",
        )

        # Initialize enhanced agent
        print("🤖 Initializing Enhanced FIA Agent...")
        agent = FIAAgent(
            rag_pipeline=rag_pipeline,
            model_name="gpt-4o-mini",
            enable_tracing=bool(langsmith_api_key),
            langsmith_api_key=langsmith_api_key,
        )

        print("✅ Enhanced agent system initialized successfully!")

        # Show available tools
        tools = agent.get_available_tools()
        print(f"\n🛠️  Available Tools ({len(tools)}):")
        for tool in tools:
            print(f"  • {tool['name']}: {tool['description']}")

        # Demo queries that showcase intent classification
        demo_queries = [
            {
                "question": "Compare Article 5 between 2024 and 2025",
                "expected_intent": "COMPARISON",
                "expected_tool": "regulation_comparison",
                "description": "Intent: COMPARISON → Tool: regulation_comparison",
            },
            {
                "question": "What are the penalties for track limits violations?",
                "expected_intent": "PENALTY",
                "expected_tool": "penalty_lookup",
                "description": "Intent: PENALTY → Tool: penalty_lookup",
            },
            {
                "question": "Find Article 12 about safety requirements",
                "expected_intent": "SEARCH",
                "expected_tool": "regulation_search",
                "description": "Intent: SEARCH → Tool: regulation_search",
            },
            {
                "question": "Summarize all safety requirements for Formula 1 cars",
                "expected_intent": "SUMMARY",
                "expected_tool": "regulation_summary",
                "description": "Intent: SUMMARY → Tool: regulation_summary",
            },
            {
                "question": "What are the engine power limits?",
                "expected_intent": "GENERAL",
                "expected_tool": "general_rag",
                "description": "Intent: GENERAL → Tool: general_rag",
            },
            {
                "question": "What is the weather today?",
                "expected_intent": "OUT_OF_SCOPE",
                "expected_tool": "out_of_scope_handler",
                "description": "Intent: OUT_OF_SCOPE → Tool: out_of_scope_handler",
            },
        ]

        print(f"\n🎯 Running {len(demo_queries)} enhanced agent demo queries...")
        print("=" * 70)

        for i, demo in enumerate(demo_queries, 1):
            print(f"\n📋 Enhanced Demo #{i}: {demo['description']}")
            print("-" * 60)
            print(f"Question: {demo['question']}")
            print(f"Expected Intent: {demo['expected_intent']}")
            print(f"Expected Tool: {demo['expected_tool']}")

            # Query the enhanced agent
            print(f"\n🤖 Agent reasoning with intent classification...")
            response = agent.query(demo["question"])

            # Display results
            print(f"\n💬 Final Answer:")
            print("-" * 30)
            print(response["answer"])

            # Show reasoning steps (including intent classification)
            if response.get("reasoning_steps"):
                print(f"\n🧠 Reasoning Steps ({len(response['reasoning_steps'])}):")
                for j, step in enumerate(response["reasoning_steps"], 1):
                    print(f"  {j}. {step}")

            # Show tools used
            if response.get("tools_used"):
                print(f"\n🛠️  Tools Used: {', '.join(response['tools_used'])}")

            # Show sources (if available)
            if response.get("sources"):
                print(f"\n📚 Sources ({len(response['sources'])}):")
                for j, source in enumerate(response["sources"], 1):
                    print(f"  {j}. {source}")

            # Show session info
            if response.get("session_id"):
                print(f"\n📊 Session ID: {response['session_id']}")
                print("🔍 Check LangSmith for detailed intent classification traces!")

            print("\n" + "=" * 70)

        print("\n✅ Enhanced agent demo completed successfully!")
        print("🎉 Your agentic RAG system now has intent classification!")
        print("📊 Check LangSmith dashboard for detailed intent classification traces")
        print("🛠️  System now handles 6 different intent types with proper fallbacks")

    except Exception as e:
        logger.error(f"Error in enhanced agent demo: {str(e)}")
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
