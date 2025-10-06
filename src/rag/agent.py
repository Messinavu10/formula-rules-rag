"""
FIA Regulations Agent with LangGraph

This module implements an agentic system using LangGraph for multi-step reasoning,
state management, and tool usage with LangSmith tracing.
"""

import logging
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langsmith import Client

from .rag_pipeline import FIARAGPipeline
from .tools import create_fia_tools

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the FIA regulations agent."""

    messages: Annotated[
        List[Dict[str, str]], "List of conversation messages"
    ]  # conversation history
    current_question: str  # current question being asked
    reasoning_steps: Annotated[
        List[str], "List of reasoning steps taken"
    ]  # agents thinking process
    tools_used: Annotated[
        List[str], "List of tools used in this session"
    ]  # tools used to answer the question
    selected_tools: Annotated[
        List[str], "List of tools selected for execution"
    ]  # tools selected for this question
    final_answer: Optional[str]  # #final answer
    sources: Annotated[List[str], "List of sources consulted"]  # sources consulted
    session_id: str  # session id
    tool_result: Optional[str]  # result from tool execution
    multi_tool_results: Annotated[
        Dict[str, str], "Results from multiple tools"
    ]  # results from multi-tool execution


class FIAAgent:
    """
    FIA Formula 1 Regulations Agent with multi-tool reasoning and state management.
    """

    def __init__(
        self,
        rag_pipeline: FIARAGPipeline,
        model_name: str = "gpt-4-mini",
        enable_tracing: bool = True,
        langsmith_api_key: Optional[str] = None,
    ):
        """
        Initialize the FIA agent.

        Args:
            rag_pipeline: Initialized RAG pipeline
            model_name: LLM model name
            enable_tracing: Whether to enable LangSmith tracing
            langsmith_api_key: LangSmith API key for tracing
        """
        self.rag_pipeline = rag_pipeline
        self.model_name = model_name

        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

        # Create tools
        self.tools = create_fia_tools(rag_pipeline)

        # Set up LangSmith tracing
        if enable_tracing and langsmith_api_key:
            self._setup_langsmith_tracing(langsmith_api_key)

        # Create the agent graph
        self.agent_graph = self._create_agent_graph()

        logger.info(f"✅ FIA Agent initialized with {len(self.tools)} tools")

    def _setup_langsmith_tracing(self, api_key: str):
        """Set up LangSmith tracing for monitoring agent behavior."""
        try:
            import os

            # Set environment variables for tracing
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGCHAIN_PROJECT"] = "fia-regulations-agent"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

            # Initialize the client
            client = Client(api_key=api_key)

            logger.info("✅ LangSmith tracing enabled")
        except Exception as e:
            logger.warning(f"Failed to set up LangSmith tracing: {str(e)}")

    def _create_agent_graph(self) -> StateGraph:  # workflow graph
        """Create the LangGraph agent with state management."""

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("reflect", self._reflect_node)

        # Add edges
        workflow.add_edge("reason", "act")
        workflow.add_edge("act", "reflect")

        # Add conditional edge from reflect
        workflow.add_conditional_edges(
            "reflect", self._should_continue, {"continue": "reason", "end": END}
        )

        # Set entry point
        workflow.set_entry_point("reason")

        return workflow.compile()

    def _classify_intent(self, question: str) -> str:
        """Classify the user's intent using LLM with multi-tool support."""
        try:
            classification_prompt = f"""Classify this FIA regulation question into one of these categories:

1. COMPARISON - Comparing regulations between years (e.g., "compare 2024 and 2025", "differences between years")
2. PENALTY - Looking up penalties or violations (e.g., "penalties for track limits", "violations", "sanctions")
3. SEARCH - Finding specific regulations (e.g., "find Article 5", "search for engine rules", "what are the requirements")
4. SUMMARY - Comprehensive analysis (e.g., "summarize safety requirements", "comprehensive analysis")
5. GENERAL - General regulation questions (e.g., "what are the rules", "explain regulations")
6. MULTI_TOOL - Questions requiring multiple tools (e.g., "safety requirements AND penalties", "compare AND summarize", "find regulations AND penalties")
7. OUT_OF_SCOPE - Not about FIA regulations (e.g., "weather", "cooking", "other topics")

Question: {question}

Return only the category name (COMPARISON, PENALTY, SEARCH, SUMMARY, GENERAL, MULTI_TOOL, or OUT_OF_SCOPE)."""

            messages = [
                SystemMessage(content=classification_prompt),
                HumanMessage(content=question),
            ]

            response = self.llm.invoke(messages)
            intent = response.content.strip().upper()

            # Validate intent
            valid_intents = [
                "COMPARISON",
                "PENALTY",
                "SEARCH",
                "SUMMARY",
                "GENERAL",
                "MULTI_TOOL",
                "OUT_OF_SCOPE",
            ]
            if intent not in valid_intents:
                intent = "GENERAL"  # Default fallback

            return intent

        except Exception as e:
            logger.error(f"Error in intent classification: {str(e)}")
            return "GENERAL"  # Default fallback

    def _select_tool(self, intent: str) -> str:
        """Select tool based on intent classification."""
        tool_mapping = {
            "COMPARISON": "regulation_comparison",
            "PENALTY": "penalty_lookup",
            "SEARCH": "regulation_search",
            "SUMMARY": "regulation_summary",
            "GENERAL": "general_rag",
            "OUT_OF_SCOPE": "out_of_scope_handler",
        }
        return tool_mapping.get(intent, "general_rag")

    def _select_multi_tools(self, question: str) -> List[str]:
        """Select multiple tools for complex questions requiring orchestration."""
        try:
            multi_tool_prompt = f"""Analyze this FIA regulation question and determine which tools are needed:

Available tools:
- regulation_search: Find specific regulations
- regulation_comparison: Compare regulations between years
- penalty_lookup: Look up penalties for violations
- regulation_summary: Create comprehensive summaries
- general_rag: General regulation questions

Question: {question}

Determine which tools are needed to fully answer this question. Consider:
1. Does it ask for specific regulations? → regulation_search
2. Does it ask for comparisons? → regulation_comparison
3. Does it ask for penalties? → penalty_lookup
4. Does it ask for summaries? → regulation_summary
5. Is it a general question? → general_rag

Return a JSON list of tool names, e.g., ["regulation_search", "penalty_lookup"]"""

            messages = [
                SystemMessage(content=multi_tool_prompt),
                HumanMessage(content=question),
            ]

            response = self.llm.invoke(messages)
            tools_text = response.content.strip()

            # Parse JSON response
            import json

            try:
                tools = json.loads(tools_text)
                if isinstance(tools, list):
                    return tools
                else:
                    return [tools]
            except json.JSONDecodeError:
                # Fallback: extract tool names from text
                import re

                tool_names = re.findall(
                    r'"(regulation_\w+|penalty_\w+|general_\w+)"', tools_text
                )
                return tool_names if tool_names else ["general_rag"]

        except Exception as e:
            logger.error(f"Error in multi-tool selection: {str(e)}")
            return ["general_rag"]  # Default fallback

    def _reason_node(self, state: AgentState) -> AgentState:
        """Enhanced reasoning node with multi-tool support."""
        try:
            current_question = state["current_question"]
            reasoning_steps = state.get("reasoning_steps", [])

            # Step 1: Classify intent
            intent = self._classify_intent(current_question)
            state["reasoning_steps"].append(f"Intent Classification: {intent}")

            # Step 2: Select tools based on intent
            if intent == "MULTI_TOOL":
                tools = self._select_multi_tools(current_question)
                state["reasoning_steps"].append(f"Selected Multi-Tools: {tools}")
                state["selected_tools"] = tools  # Store multiple tools
            else:
                tool = self._select_tool(intent)
                state["reasoning_steps"].append(f"Selected Tool: {tool}")
                state["selected_tools"] = [tool]  # Store as single tool list

            # Step 3: Create reasoning prompt
            tools_text = ", ".join(state["selected_tools"])
            reasoning_prompt = f"""You are an expert FIA Formula 1 regulations analyst. 

Current Question: {current_question}
Intent: {intent}
Selected Tools: {tools_text}

Previous Reasoning Steps:
{chr(10).join(reasoning_steps) if reasoning_steps else "None"}

Available Tools:
- regulation_search: Search for specific regulations
- regulation_comparison: Compare regulations between years
- penalty_lookup: Look up penalties for violations
- regulation_summary: Create comprehensive summaries
- general_rag: General regulation questions
- out_of_scope_handler: Non-regulation questions

Based on the intent classification, explain why you're using the selected tool(s) and what you expect to accomplish."""

            messages = [
                SystemMessage(content=reasoning_prompt),
                HumanMessage(content=current_question),
            ]

            response = self.llm.invoke(messages)
            reasoning = response.content

            # Update state
            state["reasoning_steps"].append(f"Reasoning: {reasoning}")
            state["reasoning_steps"].append(f"Next Action: {tools_text}")

            return state

        except Exception as e:
            logger.error(f"Error in reasoning node: {str(e)}")
            state["reasoning_steps"].append(f"Error in reasoning: {str(e)}")
            return state

    def _act_node(self, state: AgentState) -> AgentState:
        """Action node - execute selected tools with multi-tool orchestration support."""
        try:
            current_question = state.get("current_question", "")
            selected_tools = state.get("selected_tools", [])
            state.get("reasoning_steps", [])

            if not selected_tools:
                state["reasoning_steps"].append("Error: No tools selected")
                return state

            # Initialize multi-tool results
            if "multi_tool_results" not in state:
                state["multi_tool_results"] = {}

            # Execute each selected tool
            for tool_name in selected_tools:
                if tool_name in state["multi_tool_results"]:
                    # Skip if already executed
                    continue

                state["reasoning_steps"].append(f"Executing tool: {tool_name}")

                # Find and execute the tool
                tool_result = None
                for tool in self.tools:
                    if tool.name == tool_name:
                        try:
                            # Parse question and extract parameters based on tool type
                            if tool_name == "regulation_comparison":
                                # Extract article number and years from question
                                import re

                                article_match = re.search(
                                    r"Article (\d+(?:\.\d+)?)",
                                    current_question,
                                    re.IGNORECASE,
                                )
                                year_matches = re.findall(
                                    r"(20\d{2})", current_question
                                )

                                if article_match and len(year_matches) >= 2:
                                    article_number = article_match.group(1)
                                    year1, year2 = year_matches[0], year_matches[1]
                                    tool_result = tool._run(
                                        article_number=article_number,
                                        year1=year1,
                                        year2=year2,
                                    )
                                else:
                                    tool_result = f"Could not parse article number and years from: {current_question}"

                            elif tool_name == "penalty_lookup":
                                # Extract violation type from question
                                violation_type = "track limits"  # default
                                if "MGU-K" in current_question:
                                    violation_type = "MGU-K"
                                elif "fuel" in current_question.lower():
                                    violation_type = "fuel flow"
                                elif "track" in current_question.lower():
                                    violation_type = "track limits"

                                tool_result = tool._run(violation_type=violation_type)

                            else:
                                # For other tools, use the question directly
                                tool_result = tool._run(query=current_question)

                            # Store result
                            state["multi_tool_results"][tool_name] = tool_result
                            state["tools_used"].append(tool_name)
                            state["reasoning_steps"].append(
                                f"Tool {tool_name} executed successfully"
                            )
                            break

                        except Exception as e:
                            error_msg = f"Error executing {tool_name}: {str(e)}"
                            state["reasoning_steps"].append(error_msg)
                            state["multi_tool_results"][tool_name] = error_msg
                            continue

            # Combine results if multiple tools were used
            if len(selected_tools) > 1:
                combined_result = self._combine_multi_tool_results(
                    state["multi_tool_results"], current_question
                )
                state["tool_result"] = combined_result
                state["reasoning_steps"].append(
                    f"Combined results from {len(selected_tools)} tools"
                )
            else:
                # Single tool result
                tool_name = selected_tools[0]
                state["tool_result"] = state["multi_tool_results"].get(
                    tool_name, "No result"
                )

            return state

        except Exception as e:
            logger.error(f"Error in act node: {str(e)}")
            state["reasoning_steps"].append(f"Error in action: {str(e)}")
            return state

    def _combine_multi_tool_results(
        self, results: Dict[str, str], question: str
    ) -> str:
        """Combine results from multiple tools into a comprehensive answer."""
        try:
            if not results:
                return "No results from tools"

            # Create a prompt to combine results
            results_text = "\n\n".join(
                [f"**{tool_name}**:\n{result}" for tool_name, result in results.items()]
            )

            combination_prompt = f"""You are an expert FIA Formula 1 regulations analyst. Combine the following results from multiple tools into a comprehensive, well-structured answer.

Original Question: {question}

Tool Results:
{results_text}

Instructions:
1. Synthesize the information from all tools
2. Create a coherent, comprehensive answer
3. Organize the information logically
4. Highlight key points and relationships
5. Ensure the answer directly addresses the original question
6. Use clear headings and structure

Provide a well-organized, comprehensive answer that combines all the information effectively."""

            messages = [
                SystemMessage(content=combination_prompt),
                HumanMessage(content=question),
            ]

            response = self.llm.invoke(messages)
            combined_result = response.content

            return combined_result

        except Exception as e:
            logger.error(f"Error combining multi-tool results: {str(e)}")
            # Fallback: just concatenate results
            return "\n\n".join(
                [f"**{tool_name}**:\n{result}" for tool_name, result in results.items()]
            )

    def _reflect_node(self, state: AgentState) -> AgentState:
        """Reflection node - evaluate results and decide next steps."""
        try:
            # Get the tool result and generate final answer
            tool_result = state.get("tool_result", "")
            current_question = state.get("current_question", "")

            if tool_result:
                # Generate final answer based on tool result
                final_answer_prompt = f"""Based on the tool result, provide a comprehensive answer to the user's question.

Question: {current_question}

Tool Result: {tool_result}

Provide a clear, well-structured answer that directly addresses the user's question."""

                messages = [
                    SystemMessage(content=final_answer_prompt),
                    HumanMessage(content=current_question),
                ]

                response = self.llm.invoke(messages)
                final_answer = response.content

                state["final_answer"] = final_answer
                state["reasoning_steps"].append(
                    "Generated final answer based on tool result"
                )
            else:
                state["final_answer"] = (
                    "I was unable to process your question. Please try rephrasing it."
                )
                state["reasoning_steps"].append("No tool result available")

            return state

        except Exception as e:
            logger.error(f"Error in reflection node: {str(e)}")
            state["reasoning_steps"].append(f"Error in reflection: {str(e)}")
            state["final_answer"] = f"Error processing your question: {str(e)}"
            return state

    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue or end the agent loop based on results quality."""
        try:
            current_question = state.get("current_question", "")
            tool_result = state.get("tool_result", "")
            reasoning_steps = state.get("reasoning_steps", [])

            # Check if we have a valid result
            if not tool_result or tool_result.strip() == "":
                return "continue"  # Try again if no result

            # Check if this is an out-of-scope question
            intent = None
            for step in reasoning_steps:
                if step.startswith("Intent Classification:"):
                    intent = step.replace("Intent Classification:", "").strip()
                    break

            # Special handling for out-of-scope questions
            if intent == "OUT_OF_SCOPE":
                state["reasoning_steps"].append(
                    "Out-of-scope question handled correctly, ending process"
                )
                return "end"

            # Check result quality using LLM
            quality_prompt = f"""You are a quality assessor for an AI agent. Your job is to decide whether an answer is good enough or needs improvement.

Question: {current_question}

Answer: {tool_result}

Evaluate the answer quality:
- If the answer is incomplete, inaccurate, unclear, or lacks specificity, return: CONTINUE
- If the answer is complete, accurate, clear, and specific, return: END

IMPORTANT: You must respond with ONLY one word: either "CONTINUE" or "END"
Do not provide explanations, scores, or detailed analysis."""

            messages = [
                SystemMessage(content=quality_prompt),
                HumanMessage(
                    content=f"Question: {current_question}\nAnswer: {tool_result}"
                ),
            ]

            response = self.llm.invoke(messages)
            decision_text = response.content.strip().upper()

            # Parse decision from response (handle cases where LLM provides extra text)
            if "CONTINUE" in decision_text:
                decision = "CONTINUE"
            elif "END" in decision_text:
                decision = "END"
            else:
                # Default to END if we can't parse the response
                decision = "END"
                state["reasoning_steps"].append(
                    f"Could not parse quality assessment: {decision_text}"
                )

            # Add reasoning to state
            state["reasoning_steps"].append(f"Quality Assessment: {decision}")

            # Check for maximum iterations to prevent infinite loops
            iteration_count = len(
                [step for step in reasoning_steps if "Tool Result:" in step]
            )
            if iteration_count >= 3:  # Maximum 3 iterations
                state["reasoning_steps"].append(
                    "Maximum iterations reached, ending process"
                )
                return "end"

            if decision == "CONTINUE":
                state["reasoning_steps"].append(
                    "Answer quality insufficient, continuing with refinement"
                )
                return "continue"
            else:
                state["reasoning_steps"].append(
                    "Answer quality sufficient, ending process"
                )
                return "end"

        except Exception as e:
            logger.error(f"Error in should_continue: {str(e)}")
            state["reasoning_steps"].append(f"Error in quality assessment: {str(e)}")
            return "end"  # Default to end on error

    def query(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the agent with a question.

        Args:
            question: The question to ask
            session_id: Optional session ID for tracking

        Returns:
            Agent response with reasoning and sources
        """
        try:
            # Initialize state
            state = AgentState(
                messages=[],
                current_question=question,
                reasoning_steps=[],
                tools_used=[],
                selected_tools=[],
                final_answer=None,
                sources=[],
                session_id=session_id or f"session_{datetime.now().isoformat()}",
                tool_result=None,
                multi_tool_results={},
            )

            # Run the agent graph
            result = self.agent_graph.invoke(state)

            # Format response
            response = {
                "answer": result.get("final_answer", "No answer generated"),
                "reasoning_steps": result.get("reasoning_steps", []),
                "tools_used": result.get("tools_used", []),
                "sources": result.get("sources", []),
                "session_id": result.get("session_id"),
                "metadata": {
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "reasoning_steps_count": len(result.get("reasoning_steps", [])),
                },
            }

            logger.info(f"Agent query completed for: '{question[:50]}...'")
            return response

        except Exception as e:
            logger.error(f"Error in agent query: {str(e)}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "reasoning_steps": [],
                "tools_used": [],
                "sources": [],
                "session_id": session_id,
                "metadata": {"error": str(e)},
            }

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get information about available tools."""
        return [
            {"name": tool.name, "description": tool.description} for tool in self.tools
        ]
