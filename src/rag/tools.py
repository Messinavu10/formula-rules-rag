"""
Specialized Tools for FIA Formula 1 Regulations Agent

This module defines tools that the agent can use to perform specific tasks
like regulation comparison, penalty lookup, and multi-step analysis.
"""

import logging
from typing import List, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from .rag_pipeline import FIARAGPipeline

logger = logging.getLogger(__name__)


class RegulationSearchInput(BaseModel):
    """Input for regulation search tool."""

    query: str = Field(description="The search query for regulations")
    year_filter: Optional[str] = Field(
        None, description="Filter by specific year (2024, 2025, 2026)"
    )
    regulation_type: Optional[str] = Field(
        None,
        description="Filter by regulation type (sporting, technical, financial, operational)",
    )


class RegulationComparisonInput(BaseModel):
    """Input for regulation comparison tool."""

    article_number: str = Field(
        description="The article number to compare (e.g., '5', '12.3')"
    )
    year1: str = Field(description="First year to compare (e.g., '2024')")
    year2: str = Field(description="Second year to compare (e.g., '2025')")
    regulation_type: Optional[str] = Field(
        None, description="Type of regulation to compare"
    )


class PenaltyLookupInput(BaseModel):
    """Input for penalty lookup tool."""

    violation_type: str = Field(
        description="Type of violation (e.g., 'track limits', 'MGU-K', 'fuel flow')"
    )
    year: Optional[str] = Field(None, description="Year to look up penalties for")


class RegulationSearchTool(BaseTool):
    """Tool for searching FIA regulations."""

    name: str = "regulation_search"
    description: str = (
        "Search FIA regulations for specific information. Use this to find articles, sections, or specific rules."
    )
    args_schema: type = RegulationSearchInput
    rag_pipeline: FIARAGPipeline

    def __init__(self, rag_pipeline: FIARAGPipeline):
        super().__init__(rag_pipeline=rag_pipeline)

    def _run(
        self,
        query: str,
        year_filter: Optional[str] = None,
        regulation_type: Optional[str] = None,
    ) -> str:
        """Search regulations and return formatted results."""
        try:
            result = self.rag_pipeline.query(
                question=query,
                year_filter=year_filter,
                regulation_type_filter=regulation_type,
                k=3,
            )

            if not result.get("answer"):
                return "No relevant regulations found for your query."

            # Format the response
            response = f"**Search Results:**\n{result['answer']}\n\n"

            if result.get("sources"):
                response += "**Sources:**\n"
                for i, source in enumerate(result["sources"], 1):
                    response += f"{i}. {source}\n"

            return response

        except Exception as e:
            logger.error(f"Error in regulation search: {str(e)}")
            return f"Error searching regulations: {str(e)}"


class RegulationComparisonTool(BaseTool):
    """Tool for comparing regulations between different years."""

    name: str = "regulation_comparison"
    description: str = (
        "Compare specific articles or sections between different years of FIA regulations."
    )
    args_schema: type = RegulationComparisonInput
    rag_pipeline: FIARAGPipeline

    def __init__(self, rag_pipeline: FIARAGPipeline):
        super().__init__(rag_pipeline=rag_pipeline)

    def _run(
        self,
        article_number: str,
        year1: str,
        year2: str,
        regulation_type: Optional[str] = None,
    ) -> str:
        """Compare regulations between two years."""
        try:
            # Search for the article in year1
            query1 = f"Article {article_number}"
            if regulation_type:
                query1 += f" in {regulation_type} regulations"

            result1 = self.rag_pipeline.query(
                question=query1,
                year_filter=year1,
                regulation_type_filter=regulation_type,
                k=2,
            )

            # Search for the article in year2
            result2 = self.rag_pipeline.query(
                question=query1,
                year_filter=year2,
                regulation_type_filter=regulation_type,
                k=2,
            )

            if not result1.get("answer") or not result2.get("answer"):
                return f"Could not find Article {article_number} in one or both years."

            # Format comparison
            comparison = f"**Article {article_number} Comparison:**\n\n"
            comparison += f"**{year1} Version:**\n{result1['answer']}\n\n"
            comparison += f"**{year2} Version:**\n{result2['answer']}\n\n"

            # Add sources
            comparison += "**Sources:**\n"
            if result1.get("sources"):
                comparison += f"{year1}: {result1['sources'][0]}\n"
            if result2.get("sources"):
                comparison += f"{year2}: {result2['sources'][0]}\n"

            return comparison

        except Exception as e:
            logger.error(f"Error in regulation comparison: {str(e)}")
            return f"Error comparing regulations: {str(e)}"


class PenaltyLookupTool(BaseTool):
    """Tool for looking up penalties for specific violations."""

    name: str = "penalty_lookup"
    description: str = (
        "Look up penalties and sanctions for specific violations in FIA regulations."
    )
    args_schema: type = PenaltyLookupInput
    rag_pipeline: FIARAGPipeline

    def __init__(self, rag_pipeline: FIARAGPipeline):
        super().__init__(rag_pipeline=rag_pipeline)

    def _run(self, violation_type: str, year: Optional[str] = None) -> str:
        """Look up penalties for a specific violation type."""
        try:
            query = f"penalties for {violation_type} violations"
            if year:
                query += f" in {year}"

            result = self.rag_pipeline.query(
                question=query, year_filter=year, regulation_type_filter="sporting", k=3
            )

            if not result.get("answer"):
                return f"No penalty information found for {violation_type} violations."

            # Format penalty information
            penalty_info = f"**Penalties for {violation_type.title()} Violations:**\n\n"
            penalty_info += result["answer"] + "\n\n"

            if result.get("sources"):
                penalty_info += "**Sources:**\n"
                for i, source in enumerate(result["sources"], 1):
                    penalty_info += f"{i}. {source}\n"

            return penalty_info

        except Exception as e:
            logger.error(f"Error in penalty lookup: {str(e)}")
            return f"Error looking up penalties: {str(e)}"


class RegulationSummaryTool(BaseTool):
    """Tool for summarizing multiple regulations on a topic."""

    name: str = "regulation_summary"
    description: str = (
        "Summarize and analyze multiple regulations on a specific topic across different years or types."
    )
    args_schema: type = RegulationSearchInput
    rag_pipeline: FIARAGPipeline

    def __init__(self, rag_pipeline: FIARAGPipeline):
        super().__init__(rag_pipeline=rag_pipeline)

    def _run(
        self,
        query: str,
        year_filter: Optional[str] = None,
        regulation_type: Optional[str] = None,
    ) -> str:
        """Create a comprehensive summary of regulations on a topic."""
        try:
            # Get comprehensive results
            result = self.rag_pipeline.query(
                question=query,
                year_filter=year_filter,
                regulation_type_filter=regulation_type,
                k=5,
            )

            if not result.get("answer"):
                return f"No regulations found for: {query}"

            # Format comprehensive summary
            summary = f"**Comprehensive Analysis: {query}**\n\n"
            summary += result["answer"] + "\n\n"

            if result.get("sources"):
                summary += f"**Analysis based on {len(result['sources'])} regulation documents:**\n"
                for i, source in enumerate(result["sources"], 1):
                    summary += f"{i}. {source}\n"

            return summary

        except Exception as e:
            logger.error(f"Error in regulation summary: {str(e)}")
            return f"Error creating summary: {str(e)}"


class GeneralRAGTool(BaseTool):
    """General RAG tool for any FIA regulation questions."""

    name: str = "general_rag"
    description: str = (
        "General search tool for any FIA regulation questions that don't fit specific tools."
    )
    args_schema: type = RegulationSearchInput
    rag_pipeline: FIARAGPipeline

    def __init__(self, rag_pipeline: FIARAGPipeline):
        super().__init__(rag_pipeline=rag_pipeline)

    def _run(
        self,
        query: str,
        year_filter: Optional[str] = None,
        regulation_type: Optional[str] = None,
    ) -> str:
        """Handle general regulation questions."""
        try:
            result = self.rag_pipeline.query(
                question=query,
                year_filter=year_filter,
                regulation_type_filter=regulation_type,
                k=3,
            )

            if not result.get("answer"):
                return "I couldn't find any relevant information in the FIA regulations for your question."

            # Format general response
            response = f"**Answer:**\n{result['answer']}\n\n"

            if result.get("sources"):
                response += "**Sources:**\n"
                for i, source in enumerate(result["sources"], 1):
                    response += f"{i}. {source}\n"

            return response

        except Exception as e:
            logger.error(f"Error in general RAG: {str(e)}")
            return f"Error processing your question: {str(e)}"


class OutOfScopeTool(BaseTool):
    """Tool for handling non-FIA regulation questions."""

    name: str = "out_of_scope_handler"
    description: str = "Handle questions that are not about FIA regulations."
    args_schema: type = RegulationSearchInput

    def __init__(self):
        super().__init__()

    def _run(
        self,
        query: str,
        year_filter: Optional[str] = None,
        regulation_type: Optional[str] = None,
    ) -> str:
        """Handle out-of-scope questions."""
        return f"""I'm a specialized FIA Formula 1 regulations assistant. 

Your question: "{query}"

I can only help with questions about FIA Formula 1 regulations, including:
- Technical regulations (engines, safety, aerodynamics, etc.)
- Sporting regulations (race rules, penalties, procedures, etc.) 
- Financial regulations (budget caps, spending rules, etc.)
- Operational regulations (logistics, procedures, etc.)

Please ask me about FIA regulations instead. For example:
- "What are the engine power limits?"
- "What are the safety requirements for Formula 1 cars?"
- "What are the penalties for track limits violations?"
- "Compare the 2024 and 2025 technical regulations"
"""


def create_fia_tools(rag_pipeline: FIARAGPipeline) -> List[BaseTool]:
    """
    Create all FIA regulation tools for the agent.

    Args:
        rag_pipeline: Initialized RAG pipeline

    Returns:
        List of tools for the agent
    """
    tools = [
        RegulationSearchTool(rag_pipeline),
        RegulationComparisonTool(rag_pipeline),
        PenaltyLookupTool(rag_pipeline),
        RegulationSummaryTool(rag_pipeline),
        GeneralRAGTool(rag_pipeline),
        OutOfScopeTool(),
    ]

    logger.info(f"Created {len(tools)} FIA regulation tools")
    return tools
