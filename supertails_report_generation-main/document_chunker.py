#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized Document chunker module for processing large PDF files and creating text chunks.
This module handles PDF text extraction and chunking for vector database indexing with
improved performance for large files.
"""

import os
import json
import re
import time
from typing import List, Dict, Any
from pathlib import Path


class VeterinaryDocumentChunker:
    """
    Optimized document chunker for processing large veterinary PDF documents into searchable chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the document chunker.

        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file with optimized performance for large files.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text as string
        """
        try:
            file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
            print(f"Processing PDF: {os.path.basename(pdf_path)} ({file_size:.1f} MB)")

            start_time = time.time()

            # Try to use PyMuPDF first (best for large files)
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                print(f"Total pages: {total_pages}")

                text_parts = []
                for page_num in range(total_pages):
                    if page_num % 10 == 0:  # Progress update every 10 pages
                        print(f"Processing page {page_num + 1}/{total_pages}")

                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(page_text)

                doc.close()
                full_text = "\n".join(text_parts)

                processing_time = time.time() - start_time
                print(f"Text extraction completed in {processing_time:.1f} seconds")
                print(f"Extracted {len(full_text)} characters")

                return full_text

            except ImportError:
                print("PyMuPDF not available, trying alternatives...")

            # Fallback to pdfplumber (good for medium files)
            try:
                import pdfplumber

                with pdfplumber.open(pdf_path) as pdf:
                    text_parts = []
                    total_pages = len(pdf.pages)
                    print(f"Total pages: {total_pages}")

                    for page_num, page in enumerate(pdf.pages):
                        if page_num % 10 == 0:  # Progress update every 10 pages
                            print(f"Processing page {page_num + 1}/{total_pages}")

                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)

                    full_text = "\n".join(text_parts)

                    processing_time = time.time() - start_time
                    print(f"Text extraction completed in {processing_time:.1f} seconds")
                    print(f"Extracted {len(full_text)} characters")

                    return full_text

            except ImportError:
                print("pdfplumber not available, trying PyPDF2...")

            # Fallback to PyPDF2 (basic but works)
            try:
                import PyPDF2

                with open(pdf_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    print(f"Total pages: {total_pages}")

                    text_parts = []
                    for page_num, page in enumerate(pdf_reader.pages):
                        if page_num % 10 == 0:  # Progress update every 10 pages
                            print(f"Processing page {page_num + 1}/{total_pages}")

                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)

                    full_text = "\n".join(text_parts)

                    processing_time = time.time() - start_time
                    print(f"Text extraction completed in {processing_time:.1f} seconds")
                    print(f"Extracted {len(full_text)} characters")

                    return full_text

            except ImportError:
                pass

            raise ImportError(
                "No PDF processing library available. Install PyMuPDF (recommended), pdfplumber, or PyPDF2."
            )

        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text with optimized processing.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        print("Cleaning extracted text...")
        start_time = time.time()

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)]", "", text)

        # Normalize line breaks
        text = text.replace("\n", " ").replace("\r", " ")

        cleaned_text = text.strip()

        cleaning_time = time.time() - start_time
        print(f"Text cleaning completed in {cleaning_time:.1f} seconds")
        print(f"Cleaned text length: {len(cleaned_text)} characters")

        return cleaned_text

    def create_chunks(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Create text chunks from a document with optimized processing.

        Args:
            text: Text to chunk
            metadata: Additional metadata for the chunks

        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []

        print("Creating text chunks...")
        start_time = time.time()

        chunks = []
        start = 0
        chunk_id = 0
        total_chars = len(text)

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                search_text = text[search_start:end]

                # Find the last sentence ending
                sentence_endings = [".", "!", "?", "\n"]
                last_ending = -1
                for ending in sentence_endings:
                    pos = search_text.rfind(ending)
                    if pos > last_ending:
                        last_ending = pos

                if last_ending > 0:
                    end = search_start + last_ending + 1

            # Extract chunk text
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_data = {
                    "chunk_id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "metadata": metadata or {},
                    "start_char": start,
                    "end_char": end,
                    "length": len(chunk_text),
                }
                chunks.append(chunk_data)
                chunk_id += 1

            # Move start position for next chunk
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        chunking_time = time.time() - start_time
        print(f"Chunking completed in {chunking_time:.1f} seconds")
        print(f"Created {len(chunks)} chunks")

        return chunks

    def process_pdf_file(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a single PDF file into chunks with progress tracking.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of chunk dictionaries
        """
        print(f"\n{'='*60}")
        print(f"Processing PDF: {pdf_path}")
        print(f"{'='*60}")

        total_start_time = time.time()

        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"No text extracted from {pdf_path}")
            return []

        # Clean text
        text = self.clean_text(text)

        # Create metadata
        metadata = {
            "source_file": os.path.basename(pdf_path),
            "file_path": pdf_path,
            "file_size": os.path.getsize(pdf_path),
            "total_text_length": len(text),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

        # Create chunks
        chunks = self.create_chunks(text, metadata)

        total_time = time.time() - total_start_time
        print(f"\nTotal processing time: {total_time:.1f} seconds")
        print(f"Successfully processed {len(chunks)} chunks")
        print(f"{'='*60}\n")

        return chunks

    def process_directory(self, pdf_directory: str, output_file: str) -> None:
        """Process all PDF files in a directory and save chunks to JSON.

        Args:
            pdf_directory: Directory containing PDF files
            output_file: Path to save the JSON output
        """
        pdf_dir = Path(pdf_directory)

        if not pdf_dir.exists():
            print(f"Directory not found: {pdf_directory}")
            return

        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {pdf_directory}")
            return

        print(f"Found {len(pdf_files)} PDF files to process")

        all_chunks = []

        for pdf_file in pdf_files:
            try:
                chunks = self.process_pdf_file(str(pdf_file))
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue

        # Save chunks to JSON file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(all_chunks)} chunks to {output_file}")


def main():
    """Example usage of the document chunker."""
    import argparse

    parser = argparse.ArgumentParser(description="Process PDF files into text chunks")
    parser.add_argument("pdf_directory", help="Directory containing PDF files")
    parser.add_argument("output_file", help="Output JSON file for chunks")
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=50, help="Chunk overlap in characters"
    )

    args = parser.parse_args()

    chunker = VeterinaryDocumentChunker(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    chunker.process_directory(args.pdf_directory, args.output_file)


if __name__ == "__main__":
    main()
