#!/usr/bin/env python3
"""
FIA Regulations Agent Demo Script

This script demonstrates the agentic RAG system with multi-tool reasoning,
LangGraph state management, and LangSmith tracing.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.rag_pipeline import FIARAGPipeline
from rag.agent import FIAAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the agent demo."""
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
    
    print("🏎️  FIA Formula 1 Regulations Agent System")
    print("=" * 60)
    print("🤖 Multi-Tool Reasoning with LangGraph")
    print("📊 LangSmith Tracing Enabled")
    print("=" * 60)
    
    try:
        # Initialize RAG pipeline
        print("🔧 Initializing RAG pipeline...")
        rag_pipeline = FIARAGPipeline(
            index_name=index_name,
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            model_name="gpt-4-mini"
        )
        
        # Initialize agent
        print("🤖 Initializing FIA Agent...")
        agent = FIAAgent(
            rag_pipeline=rag_pipeline,
            model_name="gpt-4-mini",
            enable_tracing=bool(langsmith_api_key),
            langsmith_api_key=langsmith_api_key
        )
        
        print("✅ Agent system initialized successfully!")
        
        # Show available tools
        tools = agent.get_available_tools()
        print(f"\n🛠️  Available Tools ({len(tools)}):")
        for tool in tools:
            print(f"  • {tool['name']}: {tool['description']}")
        
        # Demo queries that showcase multi-step reasoning
        demo_queries = [
            {
                "question": "Find Article 5 for 2026 and compare it with the 2025 version.",
                "description": "Multi-step: Search + Compare + Analyze"
            },
            {
                "question": "List penalties for violating MGU-K limits and summarize them.",
                "description": "Multi-step: Lookup + Summarize + Analyze"
            },
            {
                "question": "What are the safety requirements for Formula 1 cars and how have they changed from 2024 to 2025?",
                "description": "Multi-step: Search + Compare + Analyze changes"
            }
        ]
        
        print(f"\n🎯 Running {len(demo_queries)} agent demo queries...")
        print("=" * 60)
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n📋 Agent Demo #{i}: {demo['description']}")
            print("-" * 50)
            print(f"Question: {demo['question']}")
            
            # Query the agent
            print(f"\n🤖 Agent reasoning...")
            response = agent.query(demo['question'])
            
            # Display results
            print(f"\n💬 Final Answer:")
            print("-" * 30)
            print(response['answer'])
            
            # Show reasoning steps
            if response.get('reasoning_steps'):
                print(f"\n🧠 Reasoning Steps ({len(response['reasoning_steps'])}):")
                for j, step in enumerate(response['reasoning_steps'], 1):
                    print(f"  {j}. {step}")
            
            # Show tools used
            if response.get('tools_used'):
                print(f"\n🛠️  Tools Used: {', '.join(response['tools_used'])}")
            
            # Show sources
            if response.get('sources'):
                print(f"\n📚 Sources ({len(response['sources'])}):")
                for j, source in enumerate(response['sources'], 1):
                    print(f"  {j}. {source}")
            
            # Show session info
            if response.get('session_id'):
                print(f"\n📊 Session ID: {response['session_id']}")
                print("🔍 Check LangSmith for detailed tracing!")
            
            print("\n" + "="*60)
        
        print("\n✅ Agent demo completed successfully!")
        print("🎉 Your agentic RAG system is working!")
        print("📊 Check LangSmith dashboard for detailed traces")
        
    except Exception as e:
        logger.error(f"Error in agent demo: {str(e)}")
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
