"""
Advanced Retriever for FIA Regulations

This module provides a sophisticated retriever that combines semantic search
with metadata filtering for precise regulation retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class FIAAdvancedRetriever:
    """
    Advanced retriever for FIA regulations with semantic search and filtering.
    """
    
    def __init__(self, 
                 index_name: str,
                 openai_api_key: str,
                 pinecone_api_key: str,
                 model_name: str = "text-embedding-3-small"):
        """
        Initialize the advanced retriever.
        
        Args:
            index_name: Pinecone index name
            openai_api_key: OpenAI API key
            pinecone_api_key: Pinecone API key
            model_name: Embedding model name
        """
        self.index_name = index_name
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=model_name
        )
        
        # Initialize vector store
        self.vectorstore = PineconeVectorStore(
            embedding=self.embeddings,
            index_name=index_name
        )
        
        # Initialize LLM for compression
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4.0-mini",
            temperature=0
        )
        
        # Initialize compression retriever
        self.compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.vectorstore.as_retriever()
        )
        
        logger.info(f"✅ Advanced retriever initialized for index: {index_name}")
    
    def retrieve_with_metadata(self, # return raw document chunks with metadata for full context and filtering capabilities
                              query: str, 
                              k: int = 5,
                              year_filter: Optional[str] = None,
                              regulation_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents with metadata filtering and enhanced context.
        
        Args:
            query: Search query
            k: Number of results to return
            year_filter: Filter by specific year (e.g., "2024", "2025")
            regulation_type_filter: Filter by regulation type (e.g., "sporting", "technical")
            
        Returns:
            List of retrieved documents with enhanced metadata
        """
        try:
            # Build filter for Pinecone
            filter_dict = {}
            if year_filter:
                filter_dict["year"] = year_filter
            if regulation_type_filter:
                filter_dict["regulation_type"] = regulation_type_filter
            
            # Perform similarity search with optional filtering
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query, 
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Format results with enhanced metadata
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                metadata = doc.metadata
                
                # Create citation information
                citation = self._create_citation(metadata)
                
                result = {
                    'rank': i,
                    'score': round(score, 4),
                    'text': doc.page_content,
                    'metadata': metadata,
                    'citation': citation,
                    'source_info': self._format_source_info(metadata)
                }
                formatted_results.append(result)
            
            logger.info(f"Retrieved {len(formatted_results)} documents for query: '{query}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_compressed(self, query: str, k: int = 5) -> List[Dict[str, Any]]: # uses LL to extract only most relevant parts, reduces noise
        """
        Retrieve and compress documents using LLM-based compression.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of compressed documents
        """
        try:
            # Use compression retriever
            compressed_docs = self.compression_retriever.get_relevant_documents(query)
            
            # Format results
            formatted_results = []
            for i, doc in enumerate(compressed_docs, 1):
                metadata = doc.metadata
                citation = self._create_citation(metadata)
                
                result = {
                    'rank': i,
                    'text': doc.page_content,
                    'metadata': metadata,
                    'citation': citation,
                    'source_info': self._format_source_info(metadata),
                    'compressed': True
                }
                formatted_results.append(result)
            
            logger.info(f"Retrieved {len(formatted_results)} compressed documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in compressed retrieval: {str(e)}")
            return []
    
    def _create_citation(self, metadata: Dict[str, Any]) -> str: # create a formatted citation from metadata
        """Create a formatted citation from metadata."""
        year = metadata.get('year', 'Unknown')
        regulation_type = metadata.get('regulation_type', 'Unknown')
        section = metadata.get('section', 'Unknown')
        article = metadata.get('article', '')
        
        citation_parts = [f"FIA {year} {regulation_type.title()} Regulations"]
        if section != 'Unknown':
            citation_parts.append(f"Section {section}")
        if article:
            citation_parts.append(f"Article {article}")
        
        return ", ".join(citation_parts)
    
    def _format_source_info(self, metadata: Dict[str, Any]) -> str: # format source information for display
        """Format source information for display."""
        source_file = metadata.get('source_file', 'Unknown')
        year = metadata.get('year', 'Unknown')
        regulation_type = metadata.get('regulation_type', 'Unknown')
        
        return f"{year} {regulation_type.title()} Regulations ({source_file})"
