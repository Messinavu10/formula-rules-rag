#!/usr/bin/env python3
"""
FIA Agent RAGAS Evaluation Script

This script provides comprehensive evaluation of the FIA regulations agent
using RAGAS metrics across all tools and intent categories.
"""

import os
import sys
import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Load environment variables
load_dotenv()

from rag.agent import FIAAgent
from rag.rag_pipeline import FIARAGPipeline
from evaluation.evaluator import FIAAgentEvaluator
from evaluation.dataset import FIAEvaluationDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    
    print("🏎️  FIA Agent RAGAS Evaluation System")
    print("=" * 60)
    print("📊 Comprehensive evaluation using RAGAS metrics")
    print("🛠️  Testing all agent tools and intent categories")
    print("=" * 60)
    
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
    
    try:
        # Initialize RAG pipeline
        print("🔧 Initializing RAG pipeline...")
        rag_pipeline = FIARAGPipeline(
            index_name=index_name,
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            model_name="gpt-4o-mini"
        )
        
        # Initialize agent
        print("🤖 Initializing FIA Agent...")
        agent = FIAAgent(
            rag_pipeline=rag_pipeline,
            model_name="gpt-4o-mini",
            enable_tracing=False
        )
        
        # Create evaluator
        print("📊 Initializing RAGAS evaluator...")
        evaluator = FIAAgentEvaluator(agent)
        
        # Show available tools
        tools = agent.get_available_tools()
        print(f"\n🛠️  Available Tools ({len(tools)}):")
        for tool in tools:
            print(f"  • {tool['name']}: {tool['description']}")
        
        # Show evaluation datasets
        dataset_creator = FIAEvaluationDataset()
        datasets = dataset_creator.create_comprehensive_dataset()
        dataset_info = dataset_creator.get_dataset_info()
        
        print(f"\n📋 Evaluation Datasets:")
        for name, info in dataset_info.items():
            print(f"  • {name}: {info['size']} samples")
            print(f"    Sample: {info['sample_question']}")
        
        # Run comprehensive evaluation
        print(f"\n🚀 Starting comprehensive evaluation...")
        print("=" * 60)
        
        results = evaluator.evaluate_all_tools()
        
        # Display results
        print(f"\n📊 EVALUATION RESULTS")
        print("=" * 60)
        
        for tool_name, result in results.items():
            if tool_name == "comprehensive_report":
                continue
                
            print(f"\n🔧 {tool_name.upper()}")
            print("-" * 40)
            
            if isinstance(result, dict) and "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                # Display metrics
                if hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    print("📈 Metrics:")
                    for metric in df.columns:
                        if metric not in ['question', 'answer', 'ground_truth']:
                            try:
                                # Check if the column contains numeric data
                                if pd.api.types.is_numeric_dtype(df[metric]):
                                    score = df[metric].mean()
                                    if pd.notna(score) and isinstance(score, (int, float)):
                                        print(f"  • {metric}: {score:.3f}")
                                    else:
                                        print(f"  • {metric}: {score}")
                                else:
                                    # For non-numeric columns, show sample values
                                    sample_values = df[metric].dropna().head(3).tolist()
                                    print(f"  • {metric}: {len(df[metric])} entries (sample: {sample_values[:1]})")
                            except Exception as e:
                                print(f"  • {metric}: Error processing - {str(e)}")
        
        # Display comprehensive report
        if "comprehensive_report" in results:
            report = results["comprehensive_report"]
            print(f"\n📋 COMPREHENSIVE REPORT")
            print("=" * 60)
            
            if "summary" in report:
                summary = report["summary"]
                print(f"📊 Overall Performance:")
                
                # Handle overall_score formatting safely
                overall_score = summary.get('overall_score', 'N/A')
                if isinstance(overall_score, (int, float)):
                    print(f"  • Overall Score: {overall_score:.3f}")
                else:
                    print(f"  • Overall Score: {overall_score}")
                
                print(f"  • Tools Evaluated: {summary.get('total_tools_evaluated', 'N/A')}")
                print(f"  • Total Metrics: {summary.get('total_metrics', 'N/A')}")
            
            if "recommendations" in report:
                recommendations = report["recommendations"]
                if recommendations:
                    print(f"\n💡 Recommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"  {i}. {rec}")
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fia_agent_evaluation_{timestamp}.json"
        
        print(f"\n💾 Exporting results to {filename}...")
        export_path = evaluator.export_results(filename)
        print(f"✅ Results exported to: {export_path}")
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 Comprehensive RAGAS evaluation of FIA agent system")
        print(f"🛠️  All tools evaluated with industry-standard metrics")
        print(f"📈 Performance insights and recommendations generated")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def evaluate_single_tool(tool_name: str):
    """Evaluate a single tool."""
    
    print(f"🔧 Evaluating {tool_name} tool...")
    
    # Initialize components (same as main)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "fia-rules")
    
    rag_pipeline = FIARAGPipeline(
        index_name=index_name,
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key
    )
    
    agent = FIAAgent(rag_pipeline=rag_pipeline, enable_tracing=False)
    evaluator = FIAAgentEvaluator(agent)
    
    # Get dataset for specific tool
    dataset_creator = FIAEvaluationDataset()
    datasets = dataset_creator.create_comprehensive_dataset()
    
    if tool_name in datasets:
        dataset = datasets[tool_name]
        result = evaluator.evaluate_tool(tool_name, dataset)
        
        print(f"\n📊 {tool_name.upper()} Results:")
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            for metric in df.columns:
                if metric not in ['question', 'answer', 'ground_truth']:
                    try:
                        # Check if the column contains numeric data
                        if pd.api.types.is_numeric_dtype(df[metric]):
                            score = df[metric].mean()
                            if pd.notna(score) and isinstance(score, (int, float)):
                                print(f"  • {metric}: {score:.3f}")
                            else:
                                print(f"  • {metric}: {score}")
                        else:
                            # For non-numeric columns, show sample values
                            sample_values = df[metric].dropna().head(3).tolist()
                            print(f"  • {metric}: {len(df[metric])} entries (sample: {sample_values[:1]})")
                    except Exception as e:
                        print(f"  • {metric}: Error processing - {str(e)}")
        else:
            print(f"Result: {result}")
    else:
        print(f"❌ Tool {tool_name} not found in datasets")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FIA Agent with RAGAS")
    parser.add_argument("--tool", help="Evaluate specific tool only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.tool:
        evaluate_single_tool(args.tool)
    else:
        main()
