#!/usr/bin/env python3
"""
FIA Regulation Query Test Script

This script allows me to test queries against your vectorized FIA regulations
stored in Pinecone. It demonstrates RAG (Retrieval-Augmented Generation) 
capabilities with your Formula 1 rules data.

Features:
- Query your Pinecone vector store
- Display relevant regulation chunks
- Show metadata (source, year, regulation type)
- Interactive query interface
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fia-rules")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FIAQueryEngine:
    """Query engine for FIA regulations using Pinecone vector store."""
    
    def __init__(self):
        """Initialize the query engine."""
        # Initialize OpenAI embeddings (same as used for indexing)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        
        # Initialize Pinecone vector store
        self.vectorstore = PineconeVectorStore(
            embedding=self.embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        
        logger.info(f"✅ Query engine initialized with index: {PINECONE_INDEX_NAME}")
    
    def query(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant regulations.
        
        Args:
            question: The question to search for
            k: Number of results to return (default: 5)
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(
                question, 
                k=k
            )
            
            # Format results
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                result = {
                    'rank': i,
                    'score': round(score, 4),
                    'text': doc.page_content,
                    'metadata': doc.metadata
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return []
    
    def display_results(self, question: str, results: List[Dict[str, Any]]):
        """Display query results in a formatted way."""
        print(f"\n🔍 Query: '{question}'")
        print("=" * 80)
        
        if not results:
            print("❌ No results found.")
            return
        
        for result in results:
            print(f"\n📄 Result #{result['rank']} (Score: {result['score']})")
            print("-" * 60)
            
            # Display metadata
            metadata = result['metadata']
            print(f"📁 Source: {metadata.get('source_file', 'Unknown')}")
            print(f"📅 Year: {metadata.get('year', 'Unknown')}")
            print(f"📋 Type: {metadata.get('regulation_type', 'Unknown')}")
            print(f"📄 Section: {metadata.get('section', 'Unknown')}")
            print(f"🔢 Chunk: {metadata.get('chunk_index', 'Unknown')}")
            
            # Display text content (truncated for readability)
            text = result['text']
            if len(text) > 500:
                text = text[:500] + "..."
            print(f"\n📝 Content:\n{text}")
            print("-" * 60)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the vector store."""
        try:
            # This is a simple way to get some stats
            # Note: Pinecone doesn't have a direct "count" method in langchain_pinecone
            # We'll use a dummy query to get some basic info
            dummy_results = self.vectorstore.similarity_search("test", k=1)
            return {
                'index_name': PINECONE_INDEX_NAME,
                'status': 'connected',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'index_name': PINECONE_INDEX_NAME,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """Main function to run interactive queries."""
    # Validate configuration
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY environment variable not set")
        return
    
    print("🏎️  FIA Formula 1 Regulations Query Engine")
    print("=" * 50)
    
    # Initialize query engine
    try:
        engine = FIAQueryEngine()
    except Exception as e:
        print(f"❌ Error initializing query engine: {str(e)}")
        return
    
    # Get index stats
    stats = engine.get_index_stats()
    print(f"📊 Index: {stats['index_name']}")
    print(f"🔗 Status: {stats['status']}")
    
    # Sample queries for testing
    sample_queries = [
        "What are the safety requirements for Formula 1 cars?",
        "What is the maximum engine power allowed?",
        "What are the rules for pit stops?",
        "What are the penalties for exceeding track limits?",
        "What are the financial regulations for teams?"
    ]
    
    print(f"\n💡 Sample queries you can try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. {query}")
    
    print("\n" + "=" * 50)
    
    # Interactive query loop
    while True:
        try:
            print("\n🔍 Enter your question (or 'quit' to exit):")
            question = input("> ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Query the vector store
            print(f"\n🔎 Searching for: '{question}'...")
            results = engine.query(question, k=5)
            
            # Display results
            engine.display_results(question, results)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
