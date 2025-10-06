"""
RAGAS Evaluation Framework for FIA Agent

This module provides comprehensive evaluation capabilities using RAGAS
to assess the performance of the FIA regulations agent across all tools.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextRelevance,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from evaluation.dataset import FIAEvaluationDataset
from rag.agent import FIAAgent

logger = logging.getLogger(__name__)


class FIAAgentEvaluator:
    """
    Comprehensive evaluator for FIA regulations agent using RAGAS metrics.
    """

    def __init__(self, agent: FIAAgent):
        """
        Initialize the evaluator with an FIA agent.

        Args:
            agent: Initialized FIA agent to evaluate
        """
        self.agent = agent
        self.dataset_creator = FIAEvaluationDataset()
        self.evaluation_results = {}

        # Define metrics for different evaluation scenarios
        self.metrics = {
            "comprehensive": [
                faithfulness,  # Measures how well the model's answer matches the ground truth
                answer_relevancy,  # Measures how well the model's answer is relevant to the question
                context_precision,  # how well the agent's answer is supported by the retrieved context
                context_recall,  # Measures how well the agent retrieved the relevant context
                ContextRelevance,  # how relevant the retrieved context are
            ],
            "search_focused": [faithfulness, answer_relevancy, context_precision],
            "comparison_focused": [faithfulness, answer_relevancy, context_recall],
            "summary_focused": [faithfulness, answer_relevancy, ContextRelevance],
        }

    def evaluate_tool(
        self, tool_name: str, dataset: Dataset, metrics_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Evaluate a specific tool using RAGAS metrics.

        Args:
            tool_name: Name of the tool to evaluate
            dataset: Evaluation dataset
            metrics_type: Type of metrics to use

        Returns:
            Evaluation results dictionary
        """
        try:
            logger.info(f"Evaluating {tool_name} with {metrics_type} metrics...")

            # Get agent responses for the dataset
            responses = self._get_agent_responses(dataset, tool_name)

            # Create evaluation dataset with responses
            eval_dataset = self._create_evaluation_dataset(dataset, responses)

            # Run RAGAS evaluation
            metrics = self.metrics.get(metrics_type, self.metrics["comprehensive"])
            result = evaluate(
                dataset=eval_dataset,
                metrics=metrics,
                llm=self.agent.llm,
                embeddings=self.agent.rag_pipeline.retriever.embeddings,
            )

            # Store results
            self.evaluation_results[tool_name] = {
                "metrics": result,
                "dataset_size": len(dataset),
                "metrics_type": metrics_type,
            }

            logger.info(f"✅ {tool_name} evaluation completed")
            return result

        except Exception as e:
            logger.error(f"Error evaluating {tool_name}: {str(e)}")
            return {"error": str(e)}

    def evaluate_all_tools(self) -> Dict[str, Any]:
        """
        Evaluate all agent tools comprehensively.

        Returns:
            Comprehensive evaluation results
        """
        logger.info("🚀 Starting comprehensive FIA agent evaluation...")

        # Get all datasets
        datasets = self.dataset_creator.create_comprehensive_dataset()

        results = {}

        # Evaluate each tool
        tool_metrics = {
            "regulation_search": "search_focused",
            "regulation_comparison": "comparison_focused",
            "penalty_lookup": "search_focused",
            "regulation_summary": "summary_focused",
            "general_rag": "comprehensive",
            "out_of_scope": "search_focused",
        }

        for tool_name, dataset in datasets.items():
            metrics_type = tool_metrics.get(tool_name, "comprehensive")
            result = self.evaluate_tool(tool_name, dataset, metrics_type)
            results[tool_name] = result

        # Generate comprehensive report
        comprehensive_report = self._generate_comprehensive_report(results)
        results["comprehensive_report"] = comprehensive_report

        logger.info("✅ Comprehensive evaluation completed")
        return results

    def _get_agent_responses(
        self, dataset: Dataset, tool_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get agent responses for a dataset.

        Args:
            dataset: Evaluation dataset
            tool_name: Name of the tool being evaluated

        Returns:
            List of agent responses
        """
        responses = []

        for item in dataset:
            try:
                # Query the agent
                response = self.agent.query(item["question"])

                # Extract relevant information
                agent_response = {
                    "answer": response.get("answer", ""),
                    "sources": response.get("sources", []),
                    "reasoning_steps": response.get("reasoning_steps", []),
                    "tools_used": response.get("tools_used", []),
                    "session_id": response.get("session_id", ""),
                }

                responses.append(agent_response)

            except Exception as e:
                logger.error(f"Error getting response for question: {str(e)}")
                responses.append(
                    {
                        "answer": f"Error: {str(e)}",
                        "sources": [],
                        "reasoning_steps": [],
                        "tools_used": [],
                        "session_id": "",
                    }
                )

        return responses

    def _create_evaluation_dataset(
        self, original_dataset: Dataset, responses: List[Dict[str, Any]]
    ) -> Dataset:
        """
        Create evaluation dataset with agent responses.

        Args:
            original_dataset: Original evaluation dataset
            responses: Agent responses

        Returns:
            Dataset ready for RAGAS evaluation
        """
        eval_data = []

        for i, item in enumerate(original_dataset):
            response = responses[i] if i < len(responses) else {}

            eval_item = {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "contexts": item["contexts"],
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
            }

            eval_data.append(eval_item)

        return Dataset.from_list(eval_data)

    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            results: Evaluation results for all tools

        Returns:
            Comprehensive report
        """
        report = {"summary": {}, "tool_performance": {}, "recommendations": []}

        # Calculate overall performance
        total_metrics = 0
        total_score = 0

        for tool_name, result in results.items():
            if isinstance(result, dict) and "error" not in result:
                # Extract metric scores
                tool_scores = {}
                if hasattr(result, "to_pandas"):
                    df = result.to_pandas()
                    for metric in df.columns:
                        if metric not in ["question", "answer", "ground_truth"]:
                            score = df[metric].mean()
                            tool_scores[metric] = score
                            total_score += score
                            total_metrics += 1

                report["tool_performance"][tool_name] = tool_scores

        # Calculate overall performance
        if total_metrics > 0:
            overall_score = total_score / total_metrics
            report["summary"]["overall_score"] = overall_score
            report["summary"]["total_tools_evaluated"] = len(
                [r for r in results.values() if "error" not in r]
            )
            report["summary"]["total_metrics"] = total_metrics

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(
            report["tool_performance"]
        )

        return report

    def _generate_recommendations(self, tool_performance: Dict[str, Any]) -> List[str]:
        """
        Generate improvement recommendations based on evaluation results.

        Args:
            tool_performance: Performance metrics for each tool

        Returns:
            List of recommendations
        """
        recommendations = []

        for tool_name, metrics in tool_performance.items():
            if not metrics:
                continue

            # Check for low scores
            for metric, score in metrics.items():
                if score < 0.7:  # Threshold for improvement
                    recommendations.append(
                        f"Improve {metric} for {tool_name} (current: {score:.2f})"
                    )

            # Check for high scores
            high_scores = [score for score in metrics.values() if score > 0.9]
            if len(high_scores) == len(metrics):
                recommendations.append(
                    f"{tool_name} is performing excellently across all metrics"
                )

        return recommendations

    def export_results(self, filename: str = "fia_agent_evaluation.json") -> str:
        """
        Export evaluation results to file.

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        import json
        from pathlib import Path

        # Convert results to serializable format
        export_data = {}
        for tool_name, result in self.evaluation_results.items():
            try:
                # Check if result is a dictionary (our custom format)
                if isinstance(result, dict):
                    if "metrics" in result and hasattr(result["metrics"], "to_pandas"):
                        # RAGAS EvaluationResult object
                        df = result["metrics"].to_pandas()
                        export_data[tool_name] = {
                            "metrics": df.to_dict("records"),
                            "summary": {
                                "total_samples": len(df),
                                "metrics_computed": (
                                    list(df.columns) if not df.empty else []
                                ),
                                "dataset_size": result.get("dataset_size", 0),
                                "metrics_type": result.get("metrics_type", "unknown"),
                            },
                        }
                    else:
                        # Regular dictionary
                        export_data[tool_name] = result
                elif hasattr(result, "to_pandas"):
                    # Direct RAGAS EvaluationResult object
                    df = result.to_pandas()
                    export_data[tool_name] = {
                        "metrics": df.to_dict("records"),
                        "summary": {
                            "total_samples": len(df),
                            "metrics_computed": (
                                list(df.columns) if not df.empty else []
                            ),
                        },
                    }
                elif hasattr(result, "__dict__"):
                    # Try to convert object attributes to dict
                    export_data[tool_name] = {
                        "type": str(type(result).__name__),
                        "data": str(result),
                    }
                else:
                    # Fallback to string representation
                    export_data[tool_name] = {
                        "type": str(type(result).__name__),
                        "data": str(result),
                    }
            except Exception as e:
                logger.warning(f"Could not serialize result for {tool_name}: {e}")
                export_data[tool_name] = {
                    "type": str(type(result).__name__),
                    "error": str(e),
                    "data": str(result),
                }

        # Add metadata
        export_data["_metadata"] = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "tools_evaluated": list(self.evaluation_results.keys()),
            "total_tools": len(self.evaluation_results),
        }

        # Save to file
        output_path = Path(filename)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Results exported to {output_path}")
        return str(output_path)


def create_fia_agent_evaluator(agent: FIAAgent) -> FIAAgentEvaluator:
    """
    Create and return FIA agent evaluator.

    Args:
        agent: Initialized FIA agent

    Returns:
        FIA agent evaluator
    """
    return FIAAgentEvaluator(agent)
