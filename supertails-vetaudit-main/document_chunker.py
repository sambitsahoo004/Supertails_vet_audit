import os
import fitz  # PyMuPDF
import json
from typing import Dict, List, Any, Tuple
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class VeterinaryDocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize document chunker with configurable chunk size and overlap."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize the LangChain text splitter with token awareness for OpenAI's models
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",  # Embedding model we'll be using
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try these separators in order
        )

    def process_directory(self, directory_path: str, output_path: str) -> None:
        """Process all PDF files in a directory and save chunks to output file."""
        all_chunks = []

        for filename in os.listdir(directory_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                print(f"Processing {filename}...")

                # Extract text from PDF with page information
                pages_data = self.extract_text_from_pdf_with_pages(file_path)

                # Get document metadata
                metadata = self.extract_metadata(file_path)

                # Create chunks with metadata
                document_chunks = self.create_chunks_with_page_info(pages_data, metadata)

                all_chunks.extend(document_chunks)
                print(f"Created {len(document_chunks)} chunks from {filename}")

        # Save all chunks to output file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_chunks, f, indent=2)

        print(f"Saved {len(all_chunks)} chunks to {output_path}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF document (kept for backward compatibility)."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def extract_text_from_pdf_with_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page information."""
        doc = fitz.open(pdf_path)
        pages_data = []

        for page_num, page in enumerate(doc):
            pages_data.append({
                "text": page.get_text(),
                "page_number": page_num + 1  # Page numbers typically start at 1
            })

        return pages_data

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF file."""
        doc = fitz.open(file_path)
        metadata = doc.metadata

        # Add filename and path to metadata
        metadata["filename"] = os.path.basename(file_path)
        metadata["file_path"] = file_path

        return metadata

    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document text into overlapping chunks with metadata (kept for backward compatibility)."""
        # Clean text by removing excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Create a LangChain Document with the text and metadata
        langchain_doc = Document(page_content=text, metadata=metadata)

        # Split the document using the recursive splitter
        split_docs = self.splitter.split_documents([langchain_doc])

        # Convert the LangChain Document objects to our chunk format
        chunks = []
        for i, doc in enumerate(split_docs):
            chunk = {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "chunk_id": f"{metadata['filename']}_{i}"
            }
            chunks.append(chunk)

        return chunks

    def create_chunks_with_page_info(self, pages_data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """Split document text into chunks while preserving page numbers."""
        all_chunks = []
        chunk_counter = 0

        for page_data in pages_data:
            text = page_data["text"]
            page_number = page_data["page_number"]

            # Skip empty pages
            if not text.strip():
                continue

            # Clean text by removing excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Add page number to the metadata for this page's chunks
            page_metadata = metadata.copy()
            page_metadata["page_number"] = page_number

            # Create a LangChain Document with page metadata
            langchain_doc = Document(page_content=text, metadata=page_metadata)

            # Split this page's content
            split_docs = self.splitter.split_documents([langchain_doc])

            # Convert to our chunk format
            for doc in split_docs:
                chunk = {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "chunk_id": f"{metadata['filename']}_p{page_number}_{chunk_counter}"
                }
                all_chunks.append(chunk)
                chunk_counter += 1

        return all_chunks