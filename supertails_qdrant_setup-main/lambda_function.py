import json
import os
import logging
import time
from typing import List, Dict, Any, Optional
from qdrant_simple import QdrantHTTPClient, Point
import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential
import base64
import io
import sys
import json

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Import configuration
try:
    import config
except ImportError:
    # Fallback configuration if config.py is not available
    class Config:
        COLLECTION_NAME = "veterinary_knowledge"
        DOCS_FOLDER = "./docs"
        CHUNKS_FILE = "./data/chunks.json"
        CHUNK_SIZE = 500
        CHUNK_OVERLAP = 50
        QDRANT_HOST = "localhost"
        QDRANT_PORT = 6333
        OPENAI_API_KEY = ""
        PARAMETERS_CONFIG = {}

    config = Config()

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    import fitz  # PyMuPDF for PDF processing

    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    logger.warning("PyMuPDF (fitz) not available - PDF processing will be limited")
    PDF_PROCESSING_AVAILABLE = False


def lambda_handler(event, context):
    """
    AWS Lambda handler function for Qdrant operations with OpenAI embeddings
    """
    try:
        # Log the incoming event for debugging
        logger.info(f"Received event: {json.dumps(event)}")

        # Get connection details from environment variables with config fallbacks
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        openai_api_key = os.environ.get(
            "OPENAI_API_KEY", getattr(config, "OPENAI_API_KEY", "")
        )

        # Validate environment variables
        if not qdrant_url or not qdrant_api_key:
            error_response = {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": "Missing QDRANT_URL or QDRANT_API_KEY environment variables",
                        "qdrant_url_set": bool(qdrant_url),
                        "qdrant_api_key_set": bool(qdrant_api_key),
                    }
                ),
            }
            logger.error(f"Environment validation failed: {error_response}")
            return error_response

        # Initialize Qdrant HTTP client
        logger.info("Initializing Qdrant client...")
        qdrant_client = QdrantHTTPClient(
            url=qdrant_url, api_key=qdrant_api_key, timeout=30
        )

        # Parse the operation from event
        operation = event.get("operation", "health_check")
        logger.info(f"Processing operation: {operation}")

        if operation == "health_check":
            return health_check(qdrant_client)
        elif operation == "list_collections":
            return list_collections(qdrant_client)
        elif operation == "create_collection":
            return create_collection(qdrant_client, event)
        elif operation == "semantic_search":
            return semantic_search(qdrant_client, event, openai_api_key)
        elif operation == "index_chunks":
            return index_chunks(qdrant_client, event, openai_api_key)
        elif operation == "search_by_page":
            return search_by_page(qdrant_client, event, openai_api_key)
        elif operation == "collection_info":
            return collection_info(qdrant_client, event)
        elif operation == "count_chunks":
            return count_chunks(qdrant_client, event)
        elif operation == "count_pages":
            return count_pages(qdrant_client, event)
        elif operation == "delete_collection":
            return delete_collection(qdrant_client, event)
        elif operation == "scroll_points":
            return scroll_points(qdrant_client, event)
        elif operation == "process_pdf_documents":
            return process_pdf_documents(qdrant_client, event, openai_api_key)
        elif operation == "index_pdf_documents":
            return index_pdf_documents(qdrant_client, event, openai_api_key)
        elif operation == "deployment_check":
            return deployment_check(qdrant_client, event)
        else:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": f"Unknown operation: {operation}",
                        "available_operations": [
                            "health_check",
                            "create_collection",
                            "semantic_search",
                            "index_chunks",
                            "search_by_page",
                            "list_collections",
                            "collection_info",
                            "count_chunks",
                            "count_pages",
                            "delete_collection",
                            "scroll_points",
                            "process_pdf_documents",
                            "index_pdf_documents",
                            "deployment_check",
                        ],
                    }
                ),
            }

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Invalid JSON in request: {str(e)}"}),
        }
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Internal server error: {str(e)}"}),
        }


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(
    text: str, openai_api_key: str, model: str = "text-embedding-ada-002"
) -> List[float]:
    """Get embedding for text using OpenAI API with retry logic"""
    try:
        # Truncate text if too long (matching vector_database.py logic)
        max_tokens = 8000  # Conservative limit for text-embedding-ada-002
        if len(text) > max_tokens * 4:  # Rough approximation (4 chars per token)
            text = text[: max_tokens * 4]

        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }

        data = {"model": model, "input": text}

        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://api.openai.com/v1/embeddings", headers=headers, json=data
            )
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]

    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise


def health_check(client):
    """Check Qdrant connection health"""
    try:
        logger.info("Performing health check...")
        cluster_info = client.get_cluster_info()
        logger.info("Health check successful")
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "healthy",
                    "cluster_info": cluster_info,
                    "message": "Qdrant connection is working properly",
                    "config_info": {
                        "collection_name": getattr(
                            config, "COLLECTION_NAME", "veterinary_knowledge"
                        ),
                        "chunk_size": getattr(config, "CHUNK_SIZE", 500),
                        "chunk_overlap": getattr(config, "CHUNK_OVERLAP", 50),
                    },
                }
            ),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "unhealthy", "error": str(e)}),
        }


def create_collection(client, event):
    """Create a new collection with default embedding dimensions (enhanced error handling)"""
    try:
        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )
        vector_size = event.get("vector_size", 1536)
        distance_metric = event.get("distance", "Cosine")

        logger.info(f"Creating collection: {collection_name}")

        # Validate collection name
        if not collection_name or not isinstance(collection_name, str):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid collection name provided"}),
            }

        # Check if collection already exists
        try:
            if client.collection_exists(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                return {
                    "statusCode": 200,
                    "body": json.dumps(
                        {
                            "message": f"Collection {collection_name} already exists",
                            "collection_name": collection_name,
                            "status": "exists",
                        }
                    ),
                }
        except Exception as e:
            logger.warning(f"Could not check if collection exists: {str(e)}")

        # Create collection with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = client.create_collection(
                    name=collection_name,
                    vector_size=vector_size,
                    distance=distance_metric,
                )

                logger.info(f"Collection {collection_name} created successfully")
                return {
                    "statusCode": 200,
                    "body": json.dumps(
                        {
                            "message": f"Collection {collection_name} created successfully",
                            "collection_name": collection_name,
                            "vector_size": vector_size,
                            "distance": distance_metric,
                            "result": result,
                            "attempt": attempt + 1,
                        }
                    ),
                }
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {str(e)}")
                time.sleep(2**attempt)  # Exponential backoff

    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": f"Failed to create collection: {str(e)}",
                    "collection_name": event.get("collection_name", "unknown"),
                }
            ),
        }


def semantic_search(client, event, openai_api_key):
    """Semantic search using text query converted to embedding"""
    try:
        if not openai_api_key:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "OpenAI API key not configured"}),
            }

        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )
        query_text = event.get("query_text", "")
        limit = event.get("limit", 5)
        score_threshold = event.get("score_threshold", 0.0)

        if not query_text.strip():
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No query text provided"}),
            }

        logger.info(f"Performing semantic search for: {query_text}")

        # Generate embedding for query
        query_vector = get_embedding(query_text, openai_api_key)

        # Perform search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Format results (matching vector_database.py format exactly)
        results = []
        for result in search_results:
            formatted_result = {
                "text": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {}),
                "chunk_id": result.payload.get("chunk_id", ""),
                "page_number": result.payload.get("metadata", {}).get("page_number", 0),
                "relevance_score": result.score,
            }
            results.append(formatted_result)

        logger.info(f"Found {len(results)} semantic search results")
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "query": query_text,
                    "results": results,
                    "count": len(results),
                    "collection_name": collection_name,
                }
            ),
        }
    except Exception as e:
        logger.error(f"Failed to perform semantic search: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": f"Failed to perform semantic search: {str(e)}"}
            ),
        }


def index_chunks(client, event, openai_api_key):
    """Index document chunks with embeddings (matching vector_database.py implementation)"""
    try:
        if not openai_api_key:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "OpenAI API key not configured"}),
            }

        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )
        chunks = event.get("chunks", [])

        if not chunks:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No chunks provided for indexing"}),
            }

        logger.info(f"Indexing {len(chunks)} chunks into {collection_name}")

        # Process chunks in smaller batches to avoid timeouts (matching vector_database.py exactly)
        batch_size = 50  # Match vector_database.py batch size
        indexed_count = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points = []

            for j, chunk in enumerate(batch):
                try:
                    # Extract text from chunks for embedding
                    text = chunk.get("text", "")
                    if not text.strip():
                        continue

                    # Get embedding from OpenAI
                    embedding = get_embedding(text, openai_api_key)

                    # Create point (matching vector_database.py structure exactly)
                    # Ensure chunk_id is a string for consistent handling
                    chunk_id = str(chunk.get("chunk_id", f"{i + j}"))

                    # Convert chunk_id to a unique integer ID for Qdrant
                    # Just using the position for now, but could use a hash function
                    point_id = i + j

                    point = Point(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            "metadata": chunk.get("metadata", {}),
                            "chunk_id": chunk_id,
                            # Store page number as a separate field for easy filtering
                            "page_number": chunk.get("metadata", {}).get(
                                "page_number", 0
                            ),
                        },
                    )
                    points.append(point)

                    # Brief pause to avoid hitting rate limits
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error processing chunk {i + j}: {str(e)}")
                    continue

            if points:
                # Upload batch to database
                client.upsert_points(collection_name=collection_name, points=points)
                indexed_count += len(points)
                logger.info(
                    f"Indexed batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}, total indexed: {indexed_count}"
                )

                # Sleep between batches to manage API rate limits
                time.sleep(1)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Successfully indexed {indexed_count} chunks",
                    "collection_name": collection_name,
                    "indexed_count": indexed_count,
                    "total_chunks": len(chunks),
                }
            ),
        }
    except Exception as e:
        logger.error(f"Failed to index chunks: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to index chunks: {str(e)}"}),
        }


def search_by_page(client, event, openai_api_key):
    """Search within a specific page (matching vector_database.py functionality exactly)"""
    try:
        if not openai_api_key:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "OpenAI API key not configured"}),
            }

        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )
        query_text = event.get("query_text", "")
        page_number = event.get("page_number")
        limit = event.get("limit", 5)

        if not query_text.strip():
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No query text provided"}),
            }

        logger.info(f"Searching page {page_number} for: {query_text}")

        # Generate embedding for query using OpenAI
        query_vector = get_embedding(query_text, openai_api_key)

        # Search in vector database (matching vector_database.py approach)
        # Note: qdrant_simple might not support filters like the full Qdrant client
        # So we'll get more results and filter them manually
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit * 3,  # Get more results to filter
            score_threshold=0.0,
        )

        # Format and filter results (matching vector_database.py logic exactly)
        filtered_results = []
        for result in search_results:
            result_page = result.payload.get("metadata", {}).get("page_number", 0)

            # Create filter if page number is specified
            if page_number is None or result_page == page_number:
                formatted_result = {
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "chunk_id": result.payload.get("chunk_id", ""),
                    "page_number": result_page,
                    "relevance_score": result.score,
                }
                filtered_results.append(formatted_result)

                if len(filtered_results) >= limit:
                    break

        logger.info(f"Found {len(filtered_results)} results for page {page_number}")
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "query": query_text,
                    "page_number": page_number,
                    "results": filtered_results,
                    "count": len(filtered_results),
                    "collection_name": collection_name,
                }
            ),
        }
    except Exception as e:
        logger.error(f"Failed to search by page: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to search by page: {str(e)}"}),
        }


def list_collections(client):
    """List all collections"""
    try:
        logger.info("Listing all collections")
        collections = client.list_collections()

        logger.info(f"Found {len(collections)} collections")
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "collections": collections,
                    "count": len(collections),
                    "default_collection": getattr(
                        config, "COLLECTION_NAME", "veterinary_knowledge"
                    ),
                }
            ),
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to list collections: {str(e)}"}),
        }


def collection_info(client, event):
    """Get collection information (using config collection name as default)"""
    try:
        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )

        logger.info(f"Getting info for collection: {collection_name}")
        info = client.collection_info(collection_name)

        return {
            "statusCode": 200,
            "body": json.dumps({"collection_name": collection_name, "info": info}),
        }
    except Exception as e:
        logger.error(f"Failed to get collection info: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to get collection info: {str(e)}"}),
        }


def count_chunks(client, event):
    """Count chunks in collection (matching vector_database.py functionality)"""
    try:
        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )

        logger.info(f"Counting chunks in collection: {collection_name}")

        # Get collection info which includes count (matching vector_database.py approach)
        info = client.collection_info(collection_name)
        count = info.get("result", {}).get("points_count", 0)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "collection_name": collection_name,
                    "count": count,
                    "message": f"Collection {collection_name} contains {count} chunks",
                }
            ),
        }
    except Exception as e:
        logger.error(f"Failed to count chunks: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to count chunks: {str(e)}"}),
        }


def count_pages(client, event):
    """Count the number of chunks per page number (matching vector_database.py functionality)"""
    try:
        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )

        logger.info(f"Counting pages in collection: {collection_name}")

        # For qdrant_simple, we need to implement scrolling differently
        # This is a simplified version - in production, you might want to use the full Qdrant client
        try:
            # Get collection info to see total points
            info = client.collection_info(collection_name)
            total_points = info.get("result", {}).get("points_count", 0)

            if total_points == 0:
                return {
                    "statusCode": 200,
                    "body": json.dumps(
                        {
                            "collection_name": collection_name,
                            "page_counts": {},
                            "total_pages": 0,
                            "message": "No chunks found in collection",
                        }
                    ),
                }

            # Note: This requires implementing scroll functionality like vector_database.py
            # For now, return a message that this feature requires full implementation
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "collection_name": collection_name,
                        "total_chunks": total_points,
                        "message": "Page counting requires scroll API implementation. Use full Qdrant client for detailed page statistics.",
                        "note": "This is a simplified version. For full page counting, implement scroll functionality like vector_database.py.",
                        "reference": "See vector_database.py count_pages() method for full implementation",
                    }
                ),
            }

        except Exception as e:
            logger.error(f"Error in page counting: {str(e)}")
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "collection_name": collection_name,
                        "page_counts": {},
                        "error": "Unable to count pages with current client",
                        "message": "Use vector_database.py locally for detailed page statistics",
                    }
                ),
            }

    except Exception as e:
        logger.error(f"Failed to count pages: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to count pages: {str(e)}"}),
        }


def scroll_points(client, event):
    """Scroll through points in a collection (basic implementation, needs full Qdrant client like vector_database.py)"""
    try:
        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )
        limit = event.get("limit", 1000)  # Match vector_database.py default
        offset = event.get("offset", 0)

        logger.info(f"Scrolling points in collection: {collection_name}")

        # This is a basic implementation - qdrant_simple might not support full scroll
        # You would need to implement this based on your qdrant_simple client capabilities
        # The vector_database.py uses full Qdrant client scroll functionality
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "collection_name": collection_name,
                    "message": "Scroll functionality requires full Qdrant client implementation",
                    "note": "Use vector_database.py locally for full scroll capabilities",
                    "reference": "See vector_database.py count_pages() method for scroll implementation example",
                    "suggested_limit": limit,
                    "suggested_offset": offset,
                }
            ),
        }

    except Exception as e:
        logger.error(f"Failed to scroll points: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to scroll points: {str(e)}"}),
        }


def delete_collection(client, event):
    """Delete a collection (using config collection name as default)"""
    try:
        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )

        logger.info(f"Deleting collection: {collection_name}")

        # Use the delete method from qdrant_simple client
        client._make_request("DELETE", f"/collections/{collection_name}")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Collection {collection_name} deleted successfully",
                    "collection_name": collection_name,
                }
            ),
        }
    except Exception as e:
        logger.error(f"Failed to delete collection: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to delete collection: {str(e)}"}),
        }


def decode_base64_pdf(base64_data):
    """
    Decode base64 PDF data and extract text using PyMuPDF
    """
    try:
        if not PDF_PROCESSING_AVAILABLE:
            return {
                "error": "PyMuPDF not available - include in Lambda layer",
                "pages": [],
            }

        # Decode base64 data
        pdf_bytes = base64.b64decode(base64_data)

        # Open PDF document
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        pages_data = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()

            pages_data.append({"page_number": page_num + 1, "text": text})

        pdf_document.close()

        return {"pages": pages_data, "total_pages": len(pages_data)}

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {"error": str(e), "pages": []}


def process_pdf_documents(client, event, openai_api_key):
    """
    Process PDF documents and index them in the vector database
    Handles both base64 PDF data and local file paths
    Use for base64-uploaded PDFs (API/S3-based)
    """
    try:
        if not openai_api_key:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "OpenAI API key not configured"}),
            }

        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )

        # Handle both base64 PDF data and local file paths
        pdf_files_data = event.get("pdf_files", [])
        pdf_file_paths = event.get("pdf_file_paths", [])

        # If no PDF data provided, check for default local file
        if not pdf_files_data and not pdf_file_paths:
            # Default to the packaged PDF file
            pdf_file_paths = ["./docs/pdf/BSAVA Drug Formulary.pdf"]

        all_chunks = []
        chunk_size = event.get("chunk_size", getattr(config, "CHUNK_SIZE", 500))
        chunk_overlap = event.get("chunk_overlap", getattr(config, "CHUNK_OVERLAP", 50))

        # Process base64 PDF data (existing logic)
        for pdf_data in pdf_files_data:
            filename = pdf_data.get("filename", "unknown.pdf")
            pages_data = pdf_data.get("pages", [])

            if not pages_data:
                logger.warning(f"No pages data found for {filename}")
                continue

            logger.info(f"Processing {filename} with {len(pages_data)} pages...")

            metadata = {
                "filename": filename,
                "file_path": pdf_data.get("file_path", filename),
                **pdf_data.get("metadata", {}),
            }

            document_chunks = create_chunks_with_page_info(
                pages_data, metadata, chunk_size, chunk_overlap
            )

            all_chunks.extend(document_chunks)
            logger.info(f"Created {len(document_chunks)} chunks from {filename}")

        # Process local PDF file paths (NEW LOGIC)
        for file_path in pdf_file_paths:
            try:
                if not PDF_PROCESSING_AVAILABLE:
                    logger.error("PyMuPDF not available for PDF processing")
                    continue

                if not os.path.exists(file_path):
                    logger.error(f"PDF file not found: {file_path}")
                    continue

                logger.info(f"Processing local PDF: {file_path}")

                # Open and process the PDF file
                pdf_document = fitz.open(file_path)
                pages_data = []

                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    text = page.get_text()

                    pages_data.append({"page_number": page_num + 1, "text": text})

                pdf_document.close()

                if not pages_data:
                    logger.warning(f"No text extracted from {file_path}")
                    continue

                filename = os.path.basename(file_path)
                logger.info(f"Processing {filename} with {len(pages_data)} pages...")

                metadata = {
                    "filename": filename,
                    "file_path": file_path,
                    "total_pages": len(pages_data),
                }

                document_chunks = create_chunks_with_page_info(
                    pages_data, metadata, chunk_size, chunk_overlap
                )

                all_chunks.extend(document_chunks)
                logger.info(f"Created {len(document_chunks)} chunks from {filename}")

            except Exception as e:
                logger.error(f"Error processing PDF file {file_path}: {str(e)}")
                continue

        if not all_chunks:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"error": "No chunks were created from the provided PDF files"}
                ),
            }

        logger.info(f"Total chunks created: {len(all_chunks)}")

        # Index all chunks in the vector database
        result = index_chunks_internal(
            client, all_chunks, collection_name, openai_api_key
        )

        if result["success"]:
            processed_files = []
            processed_files.extend([pdf["filename"] for pdf in pdf_files_data])
            processed_files.extend([os.path.basename(path) for path in pdf_file_paths])

            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message": f"Successfully processed PDF files",
                        "total_chunks_created": len(all_chunks),
                        "chunks_indexed": result["indexed_count"],
                        "collection_name": collection_name,
                        "files_processed": processed_files,
                    }
                ),
            }
        else:
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "error": f"Failed to index chunks: {result['error']}",
                        "chunks_created": len(all_chunks),
                    }
                ),
            }

    except Exception as e:
        logger.error(f"Failed to process PDF documents: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to process PDF documents: {str(e)}"}),
        }


def create_chunks_with_page_info(
    pages_data, metadata, chunk_size=500, chunk_overlap=50
):
    """
    Create chunks with page information
    Replicates the functionality from VeterinaryDocumentChunker.create_chunks_with_page_info()
    """
    import re

    all_chunks = []
    chunk_counter = 0

    # Simple text splitter since we don't have LangChain in Lambda
    # This is a simplified version - you might want to implement recursive splitting

    for page_data in pages_data:
        text = page_data.get("text", "")
        page_number = page_data.get("page_number", 1)

        # Skip empty pages
        if not text.strip():
            continue

        # Clean text by removing excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Add page number to the metadata for this page's chunks
        page_metadata = metadata.copy()
        page_metadata["page_number"] = page_number

        # Simple chunking logic (splitting by sentences and then by size)
        chunks = simple_text_splitter(text, chunk_size, chunk_overlap)

        # Create chunk objects
        for chunk_text in chunks:
            if chunk_text.strip():  # Only add non-empty chunks
                chunk = {
                    "text": chunk_text,
                    "metadata": page_metadata,
                    "chunk_id": f"{metadata['filename']}_p{page_number}_{chunk_counter}",
                }
                all_chunks.append(chunk)
                chunk_counter += 1

    return all_chunks


def simple_text_splitter(text, chunk_size=500, chunk_overlap=50):
    """
    Simple text splitter that tries to break on sentences first, then words
    This is a simplified version of what LangChain's RecursiveCharacterTextSplitter does
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Determine the end position for this chunk
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break

        # Try to find a good breaking point
        chunk_text = text[start:end]

        # Look for sentence endings
        sentence_endings = [". ", "! ", "? ", "\n\n"]
        best_break = -1

        for ending in sentence_endings:
            last_occurrence = chunk_text.rfind(ending)
            if last_occurrence > best_break and last_occurrence > len(chunk_text) * 0.5:
                best_break = last_occurrence + len(ending)

        # If no good sentence break found, look for word boundaries
        if best_break == -1:
            last_space = chunk_text.rfind(" ")
            if (
                last_space > len(chunk_text) * 0.8
            ):  # Only if it's reasonably close to the end
                best_break = last_space + 1

        # If still no good break, just cut at chunk_size
        if best_break == -1:
            best_break = chunk_size

        # Add the chunk
        chunks.append(text[start : start + best_break].strip())

        # Move start position with overlap
        start = start + best_break - chunk_overlap

        # Ensure we don't go backwards
        if start <= 0:
            start = best_break

    return [chunk for chunk in chunks if chunk.strip()]


def index_chunks_internal(client, chunks, collection_name, openai_api_key):
    """
    Internal method to index chunks (similar to the existing index_chunks but returns dict)
    """
    try:
        if not chunks:
            return {
                "success": False,
                "error": "No chunks provided for indexing",
                "indexed_count": 0,
            }

        logger.info(f"Indexing {len(chunks)} chunks into {collection_name}")

        # Process chunks in smaller batches to avoid timeouts
        batch_size = 50  # Match vector_database.py batch size
        indexed_count = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points = []

            for j, chunk in enumerate(batch):
                try:
                    # Extract text from chunks for embedding
                    text = chunk.get("text", "")
                    if not text.strip():
                        continue

                    # Get embedding from OpenAI
                    embedding = get_embedding(text, openai_api_key)

                    # Create point (matching vector_database.py structure exactly)
                    chunk_id = str(chunk.get("chunk_id", f"{i + j}"))
                    point_id = i + j

                    point = Point(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            "metadata": chunk.get("metadata", {}),
                            "chunk_id": chunk_id,
                            "page_number": chunk.get("metadata", {}).get(
                                "page_number", 0
                            ),
                        },
                    )
                    points.append(point)

                    # Brief pause to avoid hitting rate limits
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error processing chunk {i + j}: {str(e)}")
                    continue

            if points:
                # Upload batch to database
                client.upsert_points(collection_name=collection_name, points=points)
                indexed_count += len(points)
                logger.info(
                    f"Indexed batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}, total indexed: {indexed_count}"
                )

                # Sleep between batches to manage API rate limits
                time.sleep(1)

        return {"success": True, "indexed_count": indexed_count, "error": None}

    except Exception as e:
        logger.error(f"Failed to index chunks: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "indexed_count": 0}


def index_pdf_documents(client, event, openai_api_key):
    """
    Index PDF documents from local file paths
    This function reads PDF files and processes them for indexing
    """
    try:
        if not openai_api_key:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "OpenAI API key not configured"}),
            }

        collection_name = event.get(
            "collection_name",
            getattr(config, "COLLECTION_NAME", "veterinary_knowledge"),
        )

        # Get PDF file paths from the event
        pdf_file_paths = event.get("pdf_file_paths", [])

        if not pdf_file_paths:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No PDF file paths provided"}),
            }

        logger.info(f"Processing {len(pdf_file_paths)} PDF file paths")

        # Since this is running in Lambda, we need to process files differently
        # For now, return an error indicating files need to be uploaded via S3 or base64
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "error": "Direct file path processing not supported in Lambda environment",
                    "suggestion": "Use process_pdf_documents operation with base64 encoded PDF data or S3 references",
                    "received_paths": pdf_file_paths,
                }
            ),
        }

    except Exception as e:
        logger.error(f"Failed to index PDF documents: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to index PDF documents: {str(e)}"}),
        }


def deployment_check(client, event):
    """
    Check Lambda deployment status and dependencies
    """
    try:
        checks = {
            "qdrant_connection": False,
            "openai_api_key": False,
            "pdf_processing": PDF_PROCESSING_AVAILABLE,
            "environment_vars": False,
        }

        # Check Qdrant connection
        try:
            cluster_info = client.get_cluster_info()
            checks["qdrant_connection"] = True
        except:
            pass

        # Check OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        checks["openai_api_key"] = bool(openai_api_key)

        # Check environment variables
        env_validation = validate_environment()
        checks["environment_vars"] = env_validation["valid"]

        status_code = 200 if all(checks.values()) else 206

        return {
            "statusCode": status_code,
            "body": json.dumps(
                {
                    "deployment_status": "ready" if all(checks.values()) else "partial",
                    "checks": checks,
                    "recommendations": [
                        (
                            "Include PyMuPDF in Lambda layer for PDF processing"
                            if not checks["pdf_processing"]
                            else None
                        ),
                        (
                            "Set OPENAI_API_KEY environment variable"
                            if not checks["openai_api_key"]
                            else None
                        ),
                        (
                            "Check Qdrant connection settings"
                            if not checks["qdrant_connection"]
                            else None
                        ),
                        (
                            "Set required environment variables"
                            if not checks["environment_vars"]
                            else None
                        ),
                    ],
                }
            ),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Deployment check failed: {str(e)}"}),
        }


def validate_environment():
    """
    Validate required environment variables
    """
    required_vars = ["QDRANT_URL", "QDRANT_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)

    if missing_vars:
        return {
            "valid": False,
            "missing_vars": missing_vars,
            "message": f"Missing required environment variables: {', '.join(missing_vars)}",
        }

    return {"valid": True, "message": "All required environment variables are set"}


def main():
    """
    Main function to run the Lambda handler for local semantic search
    pdf is already processed and indexed, i wanted to search for a specific query

    search_query = "common symptoms of diabetes in dogs"
    """
    print("Starting Lambda function for local semantic search...")
    # Initialize Qdrant client
    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")

    client = QdrantHTTPClient(
        url=qdrant_url,
        api_key=qdrant_api_key if qdrant_api_key else None,
        timeout=30,
    )
    search_query = "common symptoms of diabetes in dogs"

    # Example event for testing
    event = {
        "operation": "search_by_page",
        "collection_name": "veterinary_knowledge",
        "query_text": search_query,
        "score_threshold": 0.0,
        "limit": 5,
    }

    # Call the search function
    response = search_by_page(
        client, event, openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    print(response)


if __name__ == "__main__":
    main()
