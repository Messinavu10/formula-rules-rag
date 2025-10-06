#!/usr/bin/env python3
"""
FIA Regulation PDF Ingestion Script

This script processes FIA regulation PDFs using LangChain components,
extracts text, chunks them, and saves the results to JSON format for vector store ingestion.

Features:
- Robust PDF text extraction using PyMuPDF
- LangChain document processing and chunking
- Metadata extraction (year, regulation type, version)
- Progress tracking and error handling
- JSON output ready for vector store
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from langchain.schema import Document
# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FIALangChainIngestionPipeline:
    """Main pipeline for processing FIA regulation PDFs using LangChain."""

    def __init__(
        self, data_dir: str, output_dir: str, chunk_size: int = 1000, overlap: int = 200
    ):
        """
        Initialize ingestion pipeline.

        Args:
            data_dir: Directory containing PDF files
            output_dir: Directory to save processed JSON files
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)

        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as string
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()

            doc.close()
            return text.strip()

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def extract_metadata(self, pdf_path: str, relative_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF path and file.

        Args:
            pdf_path: Full path to PDF file
            relative_path: Relative path from data directory

        Returns:
            Metadata dictionary
        """
        # Parse path to extract year, regulation type, and version
        path_parts = Path(relative_path).parts

        year = path_parts[0] if len(path_parts) > 0 else "unknown"
        filename = Path(pdf_path).stem

        # Extract regulation type and version from filename
        regulation_type = "unknown"
        version = "unknown"

        if "sporting" in filename.lower():
            regulation_type = "sporting"
        elif "technical" in filename.lower():
            regulation_type = "technical"
        elif "financial" in filename.lower():
            regulation_type = "financial"
        elif "operational" in filename.lower():
            regulation_type = "operational"

        if "final" in filename.lower():
            version = "final"
        elif "v1" in filename.lower():
            version = "v1"

        # Get file stats
        file_stats = os.stat(pdf_path)

        return {
            "document_id": f"{year}_{regulation_type}_{version}",
            "year": year,
            "regulation_type": regulation_type,
            "version": version,
            "filename": filename,
            "file_path": relative_path,
            "file_size_bytes": file_stats.st_size,
            "processed_at": datetime.now().isoformat(),
            "source": "fia_regulations",
        }

    def find_pdf_files(self) -> List[Path]:
        """Find all PDF files in the data directory."""
        pdf_files = []
        for pdf_path in self.data_dir.rglob("*.pdf"):
            pdf_files.append(pdf_path)
        return sorted(pdf_files)

    def process_single_pdf(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single PDF file using LangChain.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing processed data or None if failed
        """
        try:
            # Get relative path for metadata
            relative_path = pdf_path.relative_to(self.data_dir)

            logger.info(f"Processing: {relative_path}")

            # Extract text using PyMuPDF
            text = self.extract_text_from_pdf(str(pdf_path))
            if not text:
                logger.warning(f"No text extracted from {relative_path}")
                return None

            # Extract metadata
            metadata = self.extract_metadata(str(pdf_path), str(relative_path))

            # Create LangChain Document
            doc = Document(page_content=text, metadata=metadata)

            # Chunk using LangChain
            chunks = self.text_splitter.split_documents([doc])

            if not chunks:
                logger.warning(f"No chunks created from {relative_path}")
                return None

            # Convert LangChain documents to our format
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append(
                    {
                        "chunk_id": f"{metadata['document_id']}_chunk_{i}",
                        "text": chunk.page_content,
                        "metadata": chunk.metadata.copy(),
                        "chunk_index": i,
                        "char_count": len(chunk.page_content),
                    }
                )

            # Prepare output data
            output_data = {
                "document_info": metadata,
                "total_chunks": len(processed_chunks),
                "chunks": processed_chunks,
                "processing_stats": {
                    "total_characters": len(text),
                    "average_chunk_size": sum(
                        chunk["char_count"] for chunk in processed_chunks
                    )
                    / len(processed_chunks),
                    "chunk_size_limit": self.text_splitter._chunk_size,
                    "overlap_size": self.text_splitter._chunk_overlap,
                },
            }

            logger.info(
                f"Successfully processed {relative_path}: {len(processed_chunks)} chunks"
            )
            return output_data

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return None

    def save_processed_data(self, data: Dict[str, Any], output_path: Path) -> bool:
        """
        Save processed data to JSON file.

        Args:
            data: Processed data dictionary
            output_path: Path to save JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving to {output_path}: {str(e)}")
            return False

    def run_ingestion(self) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline.

        Returns:
            Summary statistics
        """
        logger.info("Starting FIA Regulation PDF ingestion with LangChain...")

        # Find all PDF files
        pdf_files = self.find_pdf_files()
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        if not pdf_files:
            logger.warning("No PDF files found in data directory")
            return {"status": "no_files_found"}

        # Process files
        successful_files = 0
        failed_files = 0
        total_chunks = 0

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            # Process PDF
            processed_data = self.process_single_pdf(pdf_path)

            if processed_data:
                # Save to JSON
                output_filename = (
                    f"{processed_data['document_info']['document_id']}.json"
                )
                output_path = self.output_dir / output_filename

                if self.save_processed_data(processed_data, output_path):
                    successful_files += 1
                    total_chunks += processed_data["total_chunks"]
                    logger.info(f"Saved processed data to {output_path}")
                else:
                    failed_files += 1
            else:
                failed_files += 1

        # Generate summary
        summary = {
            "status": "completed",
            "total_files": len(pdf_files),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_chunks": total_chunks,
            "output_directory": str(self.output_dir),
            "processing_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Ingestion completed: {successful_files}/{len(pdf_files)} files processed successfully"
        )
        logger.info(f"Total chunks created: {total_chunks}")

        return summary


def main():
    """Main function to run the ingestion pipeline."""
    # Configuration
    DATA_DIR = "/Users/naveenkumar/Desktop/formula-rules-rag/data"
    OUTPUT_DIR = "/Users/naveenkumar/Desktop/formula-rules-rag/processed_data"
    CHUNK_SIZE = 1000
    OVERLAP = 200

    # Initialize and run pipeline
    pipeline = FIALangChainIngestionPipeline(
        data_dir=DATA_DIR, output_dir=OUTPUT_DIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP
    )

    # Run ingestion
    summary = pipeline.run_ingestion()

    # Save summary
    summary_path = Path(OUTPUT_DIR) / "ingestion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Summary saved to {summary_path}")
    print(f"\nIngestion Summary:")
    print(f"Files processed: {summary['successful_files']}/{summary['total_files']}")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Output directory: {summary['output_directory']}")


if __name__ == "__main__":
    main()
