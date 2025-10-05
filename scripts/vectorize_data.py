#!/usr/bin/env python3
"""
FIA Regulation Vector Store Ingestion Script

This script uses the latest langchain_pinecone integration to automatically
handle document loading, embedding creation, and vector storage.

Features:
- Load processed chunks from JSON files
- Automatic embedding creation with OpenAI
- Automatic Pinecone storage with langchain_pinecone
- Built-in error handling and rate limiting
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
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


class FIAVectorStorePipeline:
    """Simple vectorization pipeline using the latest langchain_pinecone integration."""
    
    def __init__(self, processed_data_dir: str):
        """
        Initialize vector store pipeline.
        
        Args:
            processed_data_dir: Directory containing processed JSON files
        """
        self.processed_data_dir = Path(processed_data_dir)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
    
    def load_documents(self) -> List[Document]:
        """Load all documents from processed JSON files."""
        documents = []
        json_files = list(self.processed_data_dir.glob("*.json"))
        # Exclude summary files
        json_files = [f for f in json_files if f.name not in ["ingestion_summary.json", "vectorization_summary.json"]]
        
        logger.info(f"Loading {len(json_files)} JSON files...")
        
        for json_file in json_files:
            try:
                # Load JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert chunks to LangChain Documents
                for chunk in data['chunks']:
                    # Create Document with text and metadata
                    doc = Document(
                        page_content=chunk['text'],
                        metadata={
                            'chunk_id': chunk['chunk_id'],
                            'chunk_index': chunk['chunk_index'],
                            'char_count': chunk['char_count'],
                            'source_file': json_file.name,
                            **chunk['metadata']  # Include all original metadata
                        }
                    )
                    documents.append(doc)
                
                logger.info(f"Loaded {len(data['chunks'])} chunks from {json_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {str(e)}")
                continue
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def vectorize_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Vectorize documents using the latest langchain_pinecone integration."""
        logger.info("Creating vector store with langchain_pinecone...")
        
        try:
            # Use the latest langchain_pinecone integration
            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=PINECONE_INDEX_NAME
            )
            
            logger.info("✅ Vector store created successfully!")
            
            return {
                'status': 'success',
                'total_documents': len(documents),
                'index_name': PINECONE_INDEX_NAME,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_vectorization(self) -> Dict[str, Any]:
        """
        Run the complete vectorization pipeline.
        
        Returns:
            Processing summary
        """
        logger.info("Starting FIA Regulation vectorization pipeline...")
        
        # Load documents
        documents = self.load_documents()
        if not documents:
            logger.error("No documents found to process")
            return {'status': 'no_documents_found'}
        
        # Vectorize documents (langchain_pinecone handles everything!)
        result = self.vectorize_documents(documents)
        
        return result


def main():
    """Main function to run the vectorization pipeline."""
    # Validate configuration
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY environment variable not set")
        return
    
    # Configuration
    PROCESSED_DATA_DIR = "/Users/naveenkumar/Desktop/formula-rules-rag/processed_data"
    
    # Initialize and run pipeline
    pipeline = FIAVectorStorePipeline(processed_data_dir=PROCESSED_DATA_DIR)
    
    # Run vectorization
    result = pipeline.run_vectorization()
    
    # Save summary
    summary_path = Path(PROCESSED_DATA_DIR) / "vectorization_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")
    
    if result['status'] == 'success':
        print(f"\nSuccess! Vectorized {result['total_documents']} documents")
        print(f"Index: {result['index_name']}")
    else:
        print(f"\nError: {result['error']}")


if __name__ == "__main__":
    main()