# vector_database.py

import json
import os
import time
from typing import List, Dict, Any, Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models as rest
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Import the document chunker
from document_chunker import VeterinaryDocumentChunker

# Import configuration
import config


class VeterinaryKnowledgeBase:
    def __init__(self, collection_name: str = config.COLLECTION_NAME,
                 host: str = config.QDRANT_HOST,
                 port: int = config.QDRANT_PORT,
                 embedding_model: str = "text-embedding-ada-002"):
        """Initialize knowledge base with OpenAI embeddings model and vector database.

        Args:
            collection_name: Name of the vector collection
            host: Qdrant server host
            port: Qdrant server port
            embedding_model: OpenAI embedding model name
        """
        try:
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in environment variables")

            self.client = QdrantClient(host=host, port=port)
            self.collection_name = collection_name
            self.embedding_model = embedding_model
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

            # For text-embedding-ada-002, the dimension is 1536
            self.embedding_dim = 1536

            # Create collection if it doesn't exist
            self._create_collection_if_not_exists()
        except Exception as e:
            print(f"Error initializing VeterinaryKnowledgeBase: {str(e)}")
            raise

    def _create_collection_if_not_exists(self) -> None:
        """Create vector collection if it doesn't already exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                )
                print(f"Created new collection: {self.collection_name}")
            else:
                print(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI API with retry logic.

        Args:
            text: Text to encode

        Returns:
            Text embedding as a list of floats
        """
        # Truncate text if too long (OpenAI has token limits)
        max_tokens = 8000  # Conservative limit for text-embedding-ada-002
        if len(text) > max_tokens * 4:  # Rough approximation (4 chars per token)
            text = text[:max_tokens * 4]

        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def load_chunks_from_file(self, file_path: str) -> None:
        """Load document chunks from file and index them in the vector database.

        Args:
            file_path: Path to JSON file containing document chunks
        """
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            if not chunks:
                print(f"No chunks found in file: {file_path}")
                return

            self.index_chunks(chunks)
        except json.JSONDecodeError:
            print(f"Error parsing JSON from {file_path}")
            raise
        except Exception as e:
            print(f"Error loading chunks from file: {str(e)}")
            raise

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Index document chunks in the vector database.

        Args:
            chunks: List of document chunks to index
        """
        try:
            if not chunks:
                print("No chunks provided for indexing")
                return

            print(f"Indexing {len(chunks)} chunks...")

            # Process in batches to avoid memory issues and rate limits
            batch_size = 50  # Smaller batch size for API calls
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                # Extract text from chunks for embedding
                texts = [chunk.get("text", "") for chunk in batch]

                # Process each text individually to handle errors better
                points = []
                for j, (chunk, text) in enumerate(zip(batch, texts)):
                    try:
                        # Get embedding from OpenAI
                        embedding = self._get_embedding(text)

                        # Ensure chunk_id is a string for consistent handling
                        chunk_id = str(chunk.get("chunk_id", f"{i + j}"))

                        # Create point with unique ID
                        # Convert chunk_id to a unique integer ID for Qdrant
                        # Just using the position for now, but could use a hash function
                        point_id = i + j

                        points.append(
                            PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    "text": text,
                                    "metadata": chunk.get("metadata", {}),
                                    "chunk_id": chunk_id,
                                    # Store page number as a separate field for easy filtering
                                    "page_number": chunk.get("metadata", {}).get("page_number", 0)
                                }
                            )
                        )

                        # Brief pause to avoid hitting rate limits
                        time.sleep(0.1)

                    except Exception as e:
                        print(f"Error processing chunk {i + j}: {str(e)}")
                        continue

                if points:
                    # Upload batch to database
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )

                print(f"Indexed batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

                # Sleep between batches to manage API rate limits
                time.sleep(1)

        except Exception as e:
            print(f"Error indexing chunks: {str(e)}")
            raise

    def search(self, query: str, limit: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for veterinary knowledge related to a query.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional Qdrant filters to apply

        Returns:
            List of matching document chunks with relevance scores
        """
        try:
            if not query.strip():
                return []

            # Generate embedding for query using OpenAI
            query_vector = self._get_embedding(query)

            # Search in vector database
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filters
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "chunk_id": result.payload.get("chunk_id", ""),
                    "page_number": result.payload.get("metadata", {}).get("page_number", 0),
                    "relevance_score": result.score
                })

            return formatted_results
        except Exception as e:
            print(f"Error searching vector database: {str(e)}")
            return []

    def search_by_page(self, query: str, page_number: int = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for veterinary knowledge related to a query, filtered by page number.

        Args:
            query: Search query text
            page_number: Optional page number to filter results
            limit: Maximum number of results to return

        Returns:
            List of matching document chunks with relevance scores
        """
        try:
            if not query.strip():
                return []

            # Create filter if page number is specified
            filters = None
            if page_number is not None:
                filters = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="page_number",
                            match=rest.MatchValue(value=page_number)
                        )
                    ]
                )

            # Generate embedding for query using OpenAI
            query_vector = self._get_embedding(query)

            # Search in vector database
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filters
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "chunk_id": result.payload.get("chunk_id", ""),
                    "page_number": result.payload.get("metadata", {}).get("page_number", 0),
                    "relevance_score": result.score
                })

            return formatted_results
        except Exception as e:
            print(f"Error searching vector database: {str(e)}")
            return []

    def count_chunks(self) -> int:
        """Count the total number of chunks in the collection.

        Returns:
            Total number of document chunks in the collection
        """
        try:
            count = self.client.count(
                collection_name=self.collection_name,
                count_filter=None
            )
            return count.count
        except Exception as e:
            print(f"Error counting chunks: {str(e)}")
            return 0

    def count_pages(self) -> Dict[int, int]:
        """Count the number of chunks per page number.

        Returns:
            Dictionary mapping page numbers to chunk counts
        """
        try:
            # Use scrolling to get all points
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Get in batches of 1000
                with_payload=True,
                with_vectors=False
            )

            # Count chunks per page
            page_counts = {}
            points = results[0]  # First element is the points list

            while points:
                for point in points:
                    page_number = point.payload.get("metadata", {}).get("page_number", 0)
                    if page_number in page_counts:
                        page_counts[page_number] += 1
                    else:
                        page_counts[page_number] = 1

                # Get next batch if any
                last_id = points[-1].id
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                    offset=last_id
                )
                points = results[0]

            return page_counts

        except Exception as e:
            print(f"Error counting pages: {str(e)}")
            return {}

    def delete_collection(self) -> bool:
        """Delete the entire collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
            return False


def main():
    """Main function to process PDF files and index them in vector database."""
    print("=== Veterinary Knowledge Base Processor ===")

    # Define PDF directory and ensure it exists
    pdf_dir = os.path.join(config.DOCS_FOLDER, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)

    # Check if there are any PDFs to process
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}. Please add PDF files and run again.")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    # Process PDFs
    print("\n== Step 1: Processing PDFs ==")
    chunker = VeterinaryDocumentChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    chunker.process_directory(pdf_dir, config.CHUNKS_FILE)

    # Index chunks
    print("\n== Step 2: Indexing Chunks ==")
    kb = VeterinaryKnowledgeBase(
        collection_name=config.COLLECTION_NAME,
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        embedding_model="text-embedding-ada-002"
    )

    # Index chunks
    kb.load_chunks_from_file(config.CHUNKS_FILE)

    # Print final count
    count = kb.count_chunks()
    print(f"\nProcessing complete! Collection now contains {count} chunks.")
    print("\nYou can now query the knowledge base using the 'search' method.")

    # Example search
    print("\n== Example Search ==")
    search_query = "common symptoms of diabetes in dogs"
    print(f"Query: {search_query}")
    results = kb.search(search_query, limit=3)

    if results:
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results):
            page_info = f"(Page {result.get('page_number', 'unknown')})"
            print(f"\n[{i + 1}] Score: {result['relevance_score']:.4f} {page_info}")
            print("-" * 40)
            print(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])
    else:
        print("No results found.")

    # Example page-specific search
    if results and results[0].get('page_number'):
        first_result_page = results[0].get('page_number')
        print(f"\n== Page-Specific Search (Page {first_result_page}) ==")
        page_results = kb.search_by_page(search_query, page_number=first_result_page, limit=2)

        if page_results:
            print(f"\nFound {len(page_results)} results on page {first_result_page}:")
            for i, result in enumerate(page_results):
                print(f"\n[{i + 1}] Score: {result['relevance_score']:.4f}")
                print("-" * 40)
                print(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])
        else:
            print(f"No results found on page {first_result_page}.")


if __name__ == "__main__":
    main()