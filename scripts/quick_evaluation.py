#!/usr/bin/env python3
"""
Quick RAGAS Evaluation for FIA Agent

This script provides a quick evaluation of the FIA agent system
using a subset of the evaluation dataset for faster testing.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.dataset import FIAEvaluationDataset
from evaluation.evaluator import FIAAgentEvaluator
from rag.agent import FIAAgent
from rag.rag_pipeline import FIARAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_evaluation():
    """Run quick evaluation with limited dataset."""

    print("⚡ Quick FIA Agent Evaluation")
    print("=" * 40)
    print("🚀 Fast evaluation with sample data")
    print("=" * 40)

    # Configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "fia-rules")

    if not openai_api_key or not pinecone_api_key:
        print("❌ Please set OPENAI_API_KEY and PINECONE_API_KEY")
        return

    try:
        # Initialize components
        print("🔧 Initializing components...")
        rag_pipeline = FIARAGPipeline(
            index_name=index_name,
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
        )

        agent = FIAAgent(rag_pipeline=rag_pipeline, enable_tracing=False)
        evaluator = FIAAgentEvaluator(agent)

        # Test with regulation search tool
        print("🔍 Testing regulation search tool...")
        dataset_creator = FIAEvaluationDataset()
        search_dataset = dataset_creator.create_regulation_search_dataset()

        # Limit to first 2 samples for quick test
        limited_dataset = search_dataset.select(range(min(2, len(search_dataset))))

        result = evaluator.evaluate_tool(
            "regulation_search", limited_dataset, "search_focused"
        )

        print(f"\n📊 Quick Results:")
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            print("📈 Metrics:")
            for metric in df.columns:
                if metric not in ["question", "answer", "ground_truth"]:
                    score = df[metric].mean()
                    print(f"  • {metric}: {score:.3f}")
        else:
            print(f"Result: {result}")

        print(f"\n✅ Quick evaluation completed!")
        print(f"💡 Run full evaluation with: python scripts/evaluate_agent.py")

    except Exception as e:
        logger.error(f"Error in quick evaluation: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    quick_evaluation()
