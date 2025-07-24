# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY", ""
)  # Set your OPENAI_API_KEY key here if not using env variable


QDRANT_URL = os.environ.get(
    "QDRANT_URL", ""
)  # Set your QDRANT_URL key here if not using env variable

QDRANT_API_KEY = os.environ.get(
    "QDRANT_API_KEY", ""
)  # Set your QDRANT_API_KEY key here if not using env variable

# Qdrant database configuration
QDRANT_HOST = "localhost"  # Use actual host if not running locally
QDRANT_PORT = 6333  # Default Qdrant port
COLLECTION_NAME = "veterinary_knowledge"

# Document processing configuration
DOCS_FOLDER = "./docs"  # Folder to store documents
CHUNKS_FILE = "./data/chunks.json"  # Path to store processed chunks
CHUNK_SIZE = 500  # Size of text chunks in characters (matching your chunker default)
CHUNK_OVERLAP = 50  # Overlap between chunks (matching your chunker default)
