"""
Production-Ready RAG Pipeline for FIA Regulations

This module implements a complete RAG pipeline that combines retrieval
with generation, including proper context, citations, and source references.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .retriever import FIAAdvancedRetriever

logger = logging.getLogger(__name__)


class FIARAGPipeline:
    """
    Production-ready RAG pipeline for FIA Formula 1 regulations.
    """

    def __init__(
        self,
        index_name: str,
        openai_api_key: str,
        pinecone_api_key: str,
        model_name: str = "gpt-4-mini",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            index_name: Pinecone index name
            openai_api_key: OpenAI API key
            pinecone_api_key: Pinecone API key
            model_name: LLM model name
        """
        self.index_name = index_name

        # Initialize retriever
        self.retriever = FIAAdvancedRetriever(
            index_name=index_name,
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key, model=model_name, temperature=0
        )

        # Create prompt template
        self.prompt_template = self._create_prompt_template()

        logger.info(f"✅ RAG pipeline initialized with model: {model_name}")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create a ReAct-style prompt for FIA regulations."""

        react_system_prompt = """You are an expert FIA Formula 1 regulations analyst. You must use a structured approach to analyze regulations and provide accurate answers.

REACT METHODOLOGY:
1. **THINK**: Analyze the question and identify what specific regulation information is needed
2. **SEARCH**: Look through the provided regulation documents for relevant information  
3. **EXTRACT**: Pull out the specific articles, sections, and requirements
4. **REASON**: Connect the information to answer the question comprehensively
5. **CITE**: Provide proper citations for every claim

RESPONSE FORMAT:
**THINK**: [Your reasoning about what information is needed]
**SEARCH**: [What you're looking for in the documents]
**EXTRACT**: [Key information found with specific references]
**REASON**: [How this information answers the question]
**ANSWER**: [Final comprehensive answer with citations]

Context from FIA Regulations:
{context}

Question: {question}

Please use the ReAct methodology to provide a detailed, well-reasoned answer:"""

        return ChatPromptTemplate.from_messages(
            [("system", react_system_prompt), ("human", "{question}")]
        )

    def query(
        self,
        question: str,
        k: int = 5,
        year_filter: Optional[str] = None,
        regulation_type_filter: Optional[str] = None,
        use_compression: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.

        Args:
            question: The question to ask
            k: Number of documents to retrieve
            year_filter: Filter by year
            regulation_type_filter: Filter by regulation type
            use_compression: Whether to use compressed retrieval

        Returns:
            Dictionary containing answer, sources, and metadata
        """
        try:
            # Retrieve relevant documents
            if use_compression:
                retrieved_docs = self.retriever.retrieve_compressed(question, k=k)
            else:
                retrieved_docs = self.retriever.retrieve_with_metadata(
                    question,
                    k=k,
                    year_filter=year_filter,
                    regulation_type_filter=regulation_type_filter,
                )

            if not retrieved_docs:
                return {
                    "answer": "I couldn't find any relevant information in the FIA regulations for your question.",
                    "sources": [],
                    "citations": [],
                    "metadata": {"retrieved_docs": 0},
                }

            # Prepare context for the LLM
            context_parts = []
            citations = []

            for doc in retrieved_docs:
                context_parts.append(f"Source: {doc['source_info']}\n{doc['text']}")
                citations.append(doc["citation"])

            context = "\n\n".join(context_parts)

            # Generate answer using LLM
            messages = [
                SystemMessage(
                    content=self.prompt_template.format_messages(
                        context=context, question=question
                    )[0].content
                ),
                HumanMessage(content=question),
            ]

            response = self.llm.invoke(messages)
            answer = response.content

            # Extract sources for display
            sources = [doc["source_info"] for doc in retrieved_docs]

            result = {
                "answer": answer,
                "sources": sources,
                "citations": citations,
                "retrieved_documents": retrieved_docs,
                "metadata": {
                    "retrieved_docs": len(retrieved_docs),
                    "model": self.llm.model_name,
                    "index": self.index_name,
                },
            }

            logger.info(f"Generated answer for query: '{question[:50]}...'")
            return result

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "citations": [],
                "metadata": {"error": str(e)},
            }

    def query_with_followup(
        self,
        question: str,
        conversation_history: List[Dict[str, str]] = None,
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Query with conversation history for follow-up questions.

        Args:
            question: Current question
            conversation_history: Previous Q&A pairs
            k: Number of documents to retrieve

        Returns:
            Dictionary containing answer and context
        """
        # Build context from conversation history
        context = ""
        if conversation_history:
            context = "Previous conversation:\n"
            for item in conversation_history:
                context += f"Q: {item.get('question', '')}\n"
                context += f"A: {item.get('answer', '')}\n\n"

        # Add current question context
        enhanced_question = f"{context}Current question: {question}"

        return self.query(enhanced_question, k=k)

    def get_available_filters(self) -> Dict[str, List[str]]:
        """
        Get available filter options for the regulations.

        Returns:
            Dictionary of available years and regulation types
        """
        return {
            "years": ["2024", "2025", "2026"],
            "regulation_types": ["sporting", "technical", "financial", "operational"],
        }
