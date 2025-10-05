"""
FIA Regulations Agent with LangGraph

This module implements an agentic system using LangGraph for multi-step reasoning,
state management, and tool usage with LangSmith tracing.
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import BaseTool
from langsmith import Client

from .rag_pipeline import FIARAGPipeline
from .tools import create_fia_tools

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the FIA regulations agent."""
    messages: Annotated[List[Dict[str, str]], "List of conversation messages"] # conversation history
    current_question: str # current question being asked
    reasoning_steps: Annotated[List[str], "List of reasoning steps taken"] #agents thinking process
    tools_used: Annotated[List[str], "List of tools used in this session"] #tools used to answer the question
    final_answer: Optional[str] # #final answer
    sources: Annotated[List[str], "List of sources consulted"] #sources consulted
    session_id: str # session id
    tool_result: Optional[str] # result from tool execution


class FIAAgent:
    """
    FIA Formula 1 Regulations Agent with multi-tool reasoning and state management.
    """
    
    def __init__(self, 
                 rag_pipeline: FIARAGPipeline,
                 model_name: str = "gpt-4-mini",
                 enable_tracing: bool = True,
                 langsmith_api_key: Optional[str] = None):
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
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1
        )
        
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
    
    def _create_agent_graph(self) -> StateGraph: #workflow graph
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
            "reflect",
            self._should_continue,
            {
                "continue": "reason",
                "end": END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("reason")
        
        return workflow.compile()
    
    def _classify_intent(self, question: str) -> str:
        """Classify the user's intent using LLM."""
        try:
            classification_prompt = f"""Classify this FIA regulation question into one of these categories:

1. COMPARISON - Comparing regulations between years (e.g., "compare 2024 and 2025", "differences between years")
2. PENALTY - Looking up penalties or violations (e.g., "penalties for track limits", "violations", "sanctions")
3. SEARCH - Finding specific regulations (e.g., "find Article 5", "search for engine rules", "what are the requirements")
4. SUMMARY - Comprehensive analysis (e.g., "summarize safety requirements", "comprehensive analysis")
5. GENERAL - General regulation questions (e.g., "what are the rules", "explain regulations")
6. OUT_OF_SCOPE - Not about FIA regulations (e.g., "weather", "cooking", "other topics")

Question: {question}

Return only the category name (COMPARISON, PENALTY, SEARCH, SUMMARY, GENERAL, or OUT_OF_SCOPE)."""

            messages = [
                SystemMessage(content=classification_prompt),
                HumanMessage(content=question)
            ]
            
            response = self.llm.invoke(messages)
            intent = response.content.strip().upper()
            
            # Validate intent
            valid_intents = ["COMPARISON", "PENALTY", "SEARCH", "SUMMARY", "GENERAL", "OUT_OF_SCOPE"]
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
            "OUT_OF_SCOPE": "out_of_scope_handler"
        }
        return tool_mapping.get(intent, "general_rag")
    
    def _reason_node(self, state: AgentState) -> AgentState:
        """Enhanced reasoning node with intent classification."""
        try:
            current_question = state["current_question"]
            reasoning_steps = state.get("reasoning_steps", [])
            
            # Step 1: Classify intent
            intent = self._classify_intent(current_question)
            state["reasoning_steps"].append(f"Intent Classification: {intent}")
            
            # Step 2: Select tool based on intent
            tool = self._select_tool(intent)
            state["reasoning_steps"].append(f"Selected Tool: {tool}")
            
            # Step 3: Create reasoning prompt
            reasoning_prompt = f"""You are an expert FIA Formula 1 regulations analyst. 

Current Question: {current_question}
Intent: {intent}
Selected Tool: {tool}

Previous Reasoning Steps:
{chr(10).join(reasoning_steps) if reasoning_steps else "None"}

Available Tools:
- regulation_search: Search for specific regulations
- regulation_comparison: Compare regulations between years
- penalty_lookup: Look up penalties for violations
- regulation_summary: Create comprehensive summaries
- general_rag: General regulation questions
- out_of_scope_handler: Non-regulation questions

Based on the intent classification, explain why you're using the selected tool and what you expect to accomplish."""

            messages = [
                SystemMessage(content=reasoning_prompt),
                HumanMessage(content=current_question)
            ]
            
            response = self.llm.invoke(messages)
            reasoning = response.content
            
            # Update state
            state["reasoning_steps"].append(f"Reasoning: {reasoning}")
            state["reasoning_steps"].append(f"Next Action: {tool}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in reasoning node: {str(e)}")
            state["reasoning_steps"].append(f"Error in reasoning: {str(e)}")
            return state
    
    def _act_node(self, state: AgentState) -> AgentState:
        """Action node - execute the selected tool with refinement capabilities."""
        try:
            current_question = state.get("current_question", "")
            reasoning_steps = state.get("reasoning_steps", [])
            previous_result = state.get("tool_result", "")
            
            # Get the last selected tool from reasoning steps
            selected_tool = None
            for step in reversed(reasoning_steps):
                if step.startswith("Selected Tool:"):
                    selected_tool = step.replace("Selected Tool:", "").strip()
                    break
            
            if not selected_tool:
                state["reasoning_steps"].append("Error: No tool selected")
                return state
            
            # Check if this is a refinement attempt
            iteration_count = len([step for step in reasoning_steps if "Tool Result:" in step])
            is_refinement = iteration_count > 0
            
            if is_refinement:
                # Decide whether to refine query or switch tools
                strategy_prompt = f"""The previous attempt didn't provide a complete answer. 

Original Question: {current_question}
Previous Result: {previous_result[:300]}...
Current Tool: {selected_tool}

Decide the best strategy:
1. REFINE_QUERY: Try the same tool with a better query
2. SWITCH_TOOL: Try a different tool that might work better

Available tools:
- regulation_search: General search
- regulation_summary: Comprehensive analysis  
- general_rag: General questions
- regulation_comparison: Compare regulations
- penalty_lookup: Look up penalties

Return only: REFINE_QUERY or SWITCH_TOOL:tool_name"""

                messages = [
                    SystemMessage(content=strategy_prompt),
                    HumanMessage(content=current_question)
                ]
                
                response = self.llm.invoke(messages)
                strategy = response.content.strip()
                state["reasoning_steps"].append(f"Refinement Strategy: {strategy}")
                
                if strategy.startswith("SWITCH_TOOL:"):
                    # Switch to a different tool
                    new_tool = strategy.replace("SWITCH_TOOL:", "").strip()
                    state["reasoning_steps"].append(f"Switching to tool: {new_tool}")
                    selected_tool = new_tool
                    query_to_use = current_question
                else:
                    # Refine the query for the same tool
                    refinement_prompt = f"""Create a more specific, refined query that will get better results. 
                    
Original Question: {current_question}
Previous Result: {previous_result[:300]}...

Focus on:
1. More specific keywords
2. Different search terms  
3. Alternative phrasings
4. Additional context

Return only the refined query:"""

                    messages = [
                        SystemMessage(content=refinement_prompt),
                        HumanMessage(content=current_question)
                    ]
                    
                    response = self.llm.invoke(messages)
                    refined_query = response.content.strip()
                    state["reasoning_steps"].append(f"Refinement Query: {refined_query}")
                    query_to_use = refined_query
            else:
                query_to_use = current_question
            
            # Find and execute the tool with proper parameters
            tool_result = None
            for tool in self.tools:
                if tool.name == selected_tool:
                    try:
                        # Parse question and extract parameters based on tool type
                        if selected_tool == "regulation_comparison":
                            # Extract article number and years from question
                            import re
                            article_match = re.search(r'Article (\d+(?:\.\d+)?)', query_to_use, re.IGNORECASE)
                            year_matches = re.findall(r'(20\d{2})', query_to_use)
                            
                            if article_match and len(year_matches) >= 2:
                                article_number = article_match.group(1)
                                year1, year2 = year_matches[0], year_matches[1]
                                tool_result = tool._run(article_number=article_number, year1=year1, year2=year2)
                            else:
                                tool_result = f"Could not parse article number and years from: {query_to_use}"
                        
                        elif selected_tool == "penalty_lookup":
                            # Extract violation type from question
                            violation_type = "track limits"  # default
                            if "MGU-K" in query_to_use:
                                violation_type = "MGU-K"
                            elif "fuel" in query_to_use.lower():
                                violation_type = "fuel flow"
                            elif "track" in query_to_use.lower():
                                violation_type = "track limits"
                            
                            tool_result = tool._run(violation_type=violation_type)
                        
                        else:
                            # For other tools, use the refined query
                            tool_result = tool._run(query=query_to_use)
                        
                        state["tools_used"].append(selected_tool)
                        state["reasoning_steps"].append(f"Tool {selected_tool} executed successfully")
                        break
                    except Exception as e:
                        state["reasoning_steps"].append(f"Error executing {selected_tool}: {str(e)}")
                        return state
            
            if tool_result:
                state["reasoning_steps"].append(f"Tool Result: {tool_result[:200]}...")
                # Store the tool result for reflection
                state["tool_result"] = tool_result
            
            return state
            
        except Exception as e:
            logger.error(f"Error in act node: {str(e)}")
            state["reasoning_steps"].append(f"Error in action: {str(e)}")
            return state
    
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
                    HumanMessage(content=current_question)
                ]
                
                response = self.llm.invoke(messages)
                final_answer = response.content
                
                state["final_answer"] = final_answer
                state["reasoning_steps"].append("Generated final answer based on tool result")
            else:
                state["final_answer"] = "I was unable to process your question. Please try rephrasing it."
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
                state["reasoning_steps"].append("Out-of-scope question handled correctly, ending process")
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
                HumanMessage(content=f"Question: {current_question}\nAnswer: {tool_result}")
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
                state["reasoning_steps"].append(f"Could not parse quality assessment: {decision_text}")
            
            # Add reasoning to state
            state["reasoning_steps"].append(f"Quality Assessment: {decision}")
            
            # Check for maximum iterations to prevent infinite loops
            iteration_count = len([step for step in reasoning_steps if "Tool Result:" in step])
            if iteration_count >= 3:  # Maximum 3 iterations
                state["reasoning_steps"].append("Maximum iterations reached, ending process")
                return "end"
            
            if decision == "CONTINUE":
                state["reasoning_steps"].append("Answer quality insufficient, continuing with refinement")
                return "continue"
            else:
                state["reasoning_steps"].append("Answer quality sufficient, ending process")
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
                final_answer=None,
                sources=[],
                session_id=session_id or f"session_{datetime.now().isoformat()}"
            )
            
            # Run the agent graph
            result = self.agent_graph.invoke(state)
            
            # Format response
            response = {
                'answer': result.get('final_answer', 'No answer generated'),
                'reasoning_steps': result.get('reasoning_steps', []),
                'tools_used': result.get('tools_used', []),
                'sources': result.get('sources', []),
                'session_id': result.get('session_id'),
                'metadata': {
                    'model': self.model_name,
                    'timestamp': datetime.now().isoformat(),
                    'reasoning_steps_count': len(result.get('reasoning_steps', []))
                }
            }
            
            logger.info(f"Agent query completed for: '{question[:50]}...'")
            return response
            
        except Exception as e:
            logger.error(f"Error in agent query: {str(e)}")
            return {
                'answer': f"Error processing your question: {str(e)}",
                'reasoning_steps': [],
                'tools_used': [],
                'sources': [],
                'session_id': session_id,
                'metadata': {'error': str(e)}
            }
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get information about available tools."""
        return [
            {
                'name': tool.name,
                'description': tool.description
            }
            for tool in self.tools
        ]
