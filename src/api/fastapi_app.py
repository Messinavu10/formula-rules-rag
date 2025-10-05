"""
FastAPI Application for FIA Regulations RAG System

This module provides a REST API for querying FIA Formula 1 regulations
with advanced filtering and conversation capabilities.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import RAG components
from rag.rag_pipeline import FIARAGPipeline
from rag.query_interface import FIAQueryInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FIA Formula 1 Regulations RAG API",
    description="REST API for querying FIA Formula 1 regulations with AI-powered search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG system
rag_pipeline = None
query_interface = None


# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask about F1 regulations")
    year_filter: Optional[str] = Field(None, description="Filter by specific year (2024, 2025, 2026)")
    regulation_type_filter: Optional[str] = Field(None, description="Filter by regulation type (sporting, technical, financial, operational)")
    use_compression: bool = Field(False, description="Whether to use compressed retrieval")
    include_sources: bool = Field(True, description="Whether to include source information")


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    citations: Optional[List[str]] = None
    metadata: Dict[str, Any]
    timestamp: str


class ConversationRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, str]]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    index_name: str
    model: str


# Dependency to get RAG system
def get_rag_system():
    """Get the initialized RAG system."""
    if rag_pipeline is None or query_interface is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return query_interface


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_pipeline, query_interface
    
    try:
        # Get configuration
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "fia-rules")
        
        if not openai_api_key or not pinecone_api_key:
            logger.error("Missing required environment variables")
            return
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = FIARAGPipeline(
            index_name=index_name,
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            model_name="gpt-4"
        )
        
        # Initialize query interface
        query_interface = FIAQueryInterface(rag_pipeline)
        
        logger.info("✅ RAG system initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        index_name=rag_pipeline.index_name,
        model=rag_pipeline.llm.model_name
    )


# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_regulations(
    request: QueryRequest,
    rag_system: FIAQueryInterface = Depends(get_rag_system)
):
    """
    Query FIA regulations with optional filtering.
    """
    try:
        response = rag_system.ask_question(
            question=request.question,
            year_filter=request.year_filter,
            regulation_type_filter=request.regulation_type_filter,
            use_compression=request.use_compression,
            include_sources=request.include_sources
        )
        
        return QueryResponse(**response)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Follow-up query endpoint
@app.post("/query/followup", response_model=QueryResponse)
async def query_followup(
    request: ConversationRequest,
    rag_system: FIAQueryInterface = Depends(get_rag_system)
):
    """
    Ask a follow-up question using conversation history.
    """
    try:
        response = rag_system.ask_followup(
            question=request.question,
            use_compression=False
        )
        
        return QueryResponse(**response)
        
    except Exception as e:
        logger.error(f"Error processing follow-up query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Get available filters
@app.get("/filters")
async def get_filters(rag_system: FIAQueryInterface = Depends(get_rag_system)):
    """Get available filter options."""
    return rag_system.get_available_filters()


# Get conversation history
@app.get("/conversation")
async def get_conversation(rag_system: FIAQueryInterface = Depends(get_rag_system)):
    """Get conversation history."""
    return rag_system.get_conversation_history()


# Clear conversation history
@app.delete("/conversation")
async def clear_conversation(rag_system: FIAQueryInterface = Depends(get_rag_system)):
    """Clear conversation history."""
    rag_system.clear_conversation()
    return {"message": "Conversation history cleared"}


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FIA Formula 1 Regulations RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "followup": "/query/followup",
            "filters": "/filters",
            "conversation": "/conversation"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
