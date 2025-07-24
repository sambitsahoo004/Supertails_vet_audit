#!/usr/bin/env python3

"""
Knowledge base module for veterinary call scoring.
This module contains the CloudVeterinaryKnowledgeBase class to avoid circular imports.
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_random_exponential


class CloudVeterinaryKnowledgeBase:
    """
    Cloud-based VeterinaryKnowledgeBase that uses Qdrant Cloud
    """

    def __init__(
        self,
        collection_name: str,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        embedding_model: str = "text-embedding-ada-002",
        openai_api_key: str = None,
    ):
        """Initialize cloud-based knowledge base"""
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Get Qdrant cloud credentials from environment or parameters
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY")

        # Get OpenAI API key for embeddings
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        if not self.qdrant_url:
            raise ValueError(
                "QDRANT_URL environment variable or qdrant_url parameter is required"
            )
        if not self.qdrant_api_key:
            raise ValueError(
                "QDRANT_API_KEY environment variable or qdrant_api_key parameter is required"
            )

        # Initialize Qdrant client directly
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=30,
        )

        # Initialize OpenAI client if API key is available
        self.openai_client = None
        if self.openai_api_key:
            try:
                from openai import OpenAI

                self.openai_client = OpenAI(api_key=self.openai_api_key)
                # For text-embedding-ada-002, the dimension is 1536
                self.embedding_dim = 1536
                print(f"✓ OpenAI client initialized for embeddings")
            except ImportError:
                print("⚠️ OpenAI package not available, using fallback embeddings")
                self.embedding_dim = 128
        else:
            print("⚠️ OPENAI_API_KEY not provided, using fallback embeddings")
            self.embedding_dim = 128

        print(f"✓ Connected to Qdrant Cloud: {self.qdrant_url}")

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get text embedding using OpenAI API or fallback method.

        Args:
            text: Text to encode

        Returns:
            Text embedding as a list of floats
        """
        if self.openai_client:
            return self._get_openai_embedding(text)
        else:
            return self._get_fallback_embedding(text)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI API with retry logic."""
        try:
            # Truncate text if too long (OpenAI has token limits)
            max_tokens = 8000  # Conservative limit for text-embedding-ada-002
            if len(text) > max_tokens * 4:  # Rough approximation (4 chars per token)
                text = text[: max_tokens * 4]

            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting OpenAI embedding: {e}")
            # Fallback to hash-based embedding
            return self._get_fallback_embedding(text)

    def _get_fallback_embedding(self, text: str) -> List[float]:
        """
        Get text embedding using a simple hash-based approach for vector search.
        This is a fallback method when no embedding model is available.
        """
        # Create a simple hash-based embedding
        hash_obj = hashlib.md5(text.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()

        # Convert hash to 128-dimensional vector
        vector = []
        for i in range(0, len(hash_hex), 2):
            if i + 1 < len(hash_hex):
                # Convert hex pair to integer and normalize
                val = int(hash_hex[i : i + 2], 16) / 255.0
                vector.append(val)

        # Pad or truncate to 128 dimensions
        while len(vector) < 128:
            vector.append(0.0)
        vector = vector[:128]

        return vector

    def _create_collection_if_not_exists(self) -> None:
        """Create vector collection if it doesn't already exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim, distance=Distance.COSINE
                    ),
                )
                print(f"Created new collection: {self.collection_name}")
            else:
                print(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise

    def load_chunks_from_file(self, file_path: str) -> None:
        """Load document chunks from file and index them in the vector database.

        Args:
            file_path: Path to JSON file containing document chunks
        """
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return

            with open(file_path, "r", encoding="utf-8") as f:
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
                batch = chunks[i : i + batch_size]

                # Extract text from chunks for embedding
                texts = [chunk.get("text", "") for chunk in batch]

                # Get embeddings for all texts in batch
                embeddings = []
                for text in texts:
                    embedding = self._get_text_embedding(text)
                    embeddings.append(embedding)

                # Create points for Qdrant
                points = []
                for j, chunk in enumerate(batch):
                    point = PointStruct(
                        id=hash(chunk.get("text", "")) % (2**63),  # Use hash as ID
                        vector=embeddings[j],
                        payload={
                            "text": chunk.get("text", ""),
                            "page": chunk.get("page", 0),
                            "chunk_id": chunk.get("chunk_id", ""),
                            "source": chunk.get("source", ""),
                            "metadata": chunk.get("metadata", {}),
                        },
                    )
                    points.append(point)

                # Upload batch to Qdrant
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                    print(
                        f"Indexed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}"
                    )
                except Exception as e:
                    print(f"Error uploading batch to Qdrant: {e}")
                    # Continue with next batch

                # Small delay to avoid rate limits
                time.sleep(0.1)

            print(f"✓ Successfully indexed {len(chunks)} chunks")

        except Exception as e:
            print(f"Error indexing chunks: {str(e)}")
            raise

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict] = None,
    ):
        """
        Search for relevant chunks using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0.0 to 1.0)
            filters: Optional filters to apply to the search

        Returns:
            List of search results with scores and payloads
        """
        try:
            # Get embedding for query
            query_vector = self._get_text_embedding(query)

            # Prepare search parameters
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "score_threshold": score_threshold,
            }

            # Add filters if provided
            if filters:
                search_params["query_filter"] = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key, match=models.MatchValue(value=value)
                        )
                        for key, value in filters.items()
                    ]
                )

            # Perform search
            results = self.client.search(**search_params)

            # Format results to match expected structure
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "text": result.payload.get("text", ""),
                        "score": result.score,
                        "payload": result.payload,
                        "id": result.id,
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error searching knowledge base: {str(e)}")
            return []

    def search_by_page(self, query: str, page_number: int = None, limit: int = 3):
        """
        Search for chunks from a specific page.

        Args:
            query: Search query text
            page_number: Specific page number to search in
            limit: Maximum number of results to return

        Returns:
            List of search results from the specified page
        """
        try:
            # Get embedding for query
            query_vector = self._get_text_embedding(query)

            # Prepare search parameters
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit,
            }

            # Add page filter if specified
            if page_number is not None:
                search_params["query_filter"] = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="page", match=models.MatchValue(value=page_number)
                        )
                    ]
                )

            # Perform search
            results = self.client.search(**search_params)

            # Format results to match expected structure
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "text": result.payload.get("text", ""),
                        "score": result.score,
                        "payload": result.payload,
                        "id": result.id,
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error searching by page: {str(e)}")
            return []

    def count_chunks(self) -> int:
        """
        Get the total number of chunks in the collection.

        Returns:
            Total number of chunks
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            print(f"Error counting chunks: {str(e)}")
            return 0

    def count_pages(self) -> Dict[int, int]:
        """
        Get the count of chunks per page.

        Returns:
            Dictionary mapping page numbers to chunk counts
        """
        try:
            # Get all points to count by page
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your data size
            )[0]

            page_counts = {}
            for point in all_points:
                page = point.payload.get("page", 0)
                page_counts[page] = page_counts.get(page, 0) + 1

            return page_counts

        except Exception as e:
            print(f"Error counting pages: {str(e)}")
            return {}

    def delete_collection(self) -> bool:
        """
        Delete the entire collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✓ Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {}
