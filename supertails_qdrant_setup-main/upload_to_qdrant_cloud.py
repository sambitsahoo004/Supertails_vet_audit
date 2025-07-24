import json
import os
import time
from pathlib import Path
import fitz  # PyMuPDF
import logging

# Import the lambda functions
from lambda_function import (
    create_collection,
    process_pdf_documents,
    semantic_search,
    search_by_page,
    count_chunks,
    collection_info,
    QdrantHTTPClient,
)

# Configure logging
# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Show INFO and above

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Ensure it handles INFO level logs
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Import configuration
try:
    import config

    logger.info("✓ Configuration loaded from config.py")
except ImportError:
    logger.info("✗ config.py not found, using default values")

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


class LocalPDFProcessor:
    """
    Local PDF processor that uses lambda functions for vector operations
    """

    def __init__(self):
        """Initialize the processor with config values"""
        self.collection_name = getattr(
            config, "COLLECTION_NAME", "veterinary_knowledge"
        )
        self.docs_folder = getattr(config, "DOCS_FOLDER", "./docs")
        self.chunks_file = getattr(config, "CHUNKS_FILE", "./data/chunks.json")
        self.chunk_size = getattr(config, "CHUNK_SIZE", 500)
        self.chunk_overlap = getattr(config, "CHUNK_OVERLAP", 50)

        # Display configuration
        logger.info(f"Configuration:")
        logger.info(f"  Collection Name: {self.collection_name}")
        logger.info(f"  Docs Folder: {self.docs_folder}")
        logger.info(f"  Chunks File: {self.chunks_file}")
        logger.info(f"  Chunk Size: {self.chunk_size}")
        logger.info(f"  Chunk Overlap: {self.chunk_overlap}")

    def extract_pdf_pages(self, pdf_path):
        """
        Extract text from PDF pages locally using PyMuPDF (fitz)
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            pages_data = []

            # Open PDF document
            pdf_document = fitz.open(str(pdf_path))

            try:
                for page_num in range(len(pdf_document)):
                    try:
                        page = pdf_document.load_page(page_num)
                        text = page.get_text()

                        if text.strip():  # Only add pages with content
                            pages_data.append(
                                {
                                    "page_number": page_num
                                    + 1,  # 1-based page numbering
                                    "text": text,
                                }
                            )
                            logger.info(
                                f"Extracted page {page_num + 1}: {len(text)} characters"
                            )
                        else:
                            logger.info(f"Page {page_num + 1}: No text content found")

                    except Exception as e:
                        logger.info(f"Error extracting page {page_num + 1}: {str(e)}")
                        continue

                logger.info(
                    f"Successfully extracted {len(pages_data)} pages from {pdf_path.name}"
                )

            finally:
                # Always close the document
                pdf_document.close()

            return pages_data

        except Exception as e:
            logger.info(f"Error reading PDF file: {str(e)}")
            return []

    def create_local_qdrant_client(self):
        """
        Create a local Qdrant client for testing
        """
        # You can set these environment variables or modify as needed
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")  # Empty for local Qdrant

        logger.info(f"Connecting to Qdrant at: {qdrant_url}")

        try:
            client = QdrantHTTPClient(
                url=qdrant_url,
                api_key=qdrant_api_key if qdrant_api_key else None,
                timeout=30,
            )

            # Test connection
            cluster_info = client.get_cluster_info()
            logger.info("✓ Successfully connected to Qdrant")
            return client

        except Exception as e:
            logger.info(f"✗ Failed to connect to Qdrant: {str(e)}")
            logger.info("Make sure Qdrant is running locally on port 6333")
            logger.info("You can start it with: docker run -p 6333:6333 qdrant/qdrant")
            return None

    def find_pdf_files(self):
        """
        Find PDF files in the configured docs folder
        """
        docs_path = Path(self.docs_folder)
        if not docs_path.exists():
            logger.info(f"✗ Docs folder not found: {docs_path}")
            return []

        pdf_files = list(docs_path.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {docs_path}")

        for pdf_file in pdf_files:
            logger.info(f"  - {pdf_file.relative_to(docs_path)}")

        return pdf_files

    def process_single_pdf(self, pdf_path, client, openai_api_key):
        """
        Process a single PDF file
        """
        logger.info(f"\n=== Processing PDF: {pdf_path} ===")

        # Extract PDF content
        pages_data = self.extract_pdf_pages(pdf_path)

        if not pages_data:
            logger.info("✗ No content extracted from PDF")
            return False

        # Process PDF and create chunks using lambda function
        pdf_files_data = [
            {
                "filename": Path(pdf_path).name,
                "file_path": str(pdf_path),
                "pages": pages_data,
                "metadata": {
                    "source": "local_pdf",
                    "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            }
        ]

        process_event = {
            "collection_name": self.collection_name,
            "pdf_files": pdf_files_data,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

        result = process_pdf_documents(client, process_event, openai_api_key)
        result_data = json.loads(result["body"])

        if result["statusCode"] == 200:
            logger.info(f"✓ Successfully processed PDF:")
            logger.info(
                f"  - Total chunks created: {result_data['total_chunks_created']}"
            )
            logger.info(f"  - Chunks indexed: {result_data['chunks_indexed']}")
            return True
        else:
            logger.info(
                f"✗ Error processing PDF: {result_data.get('error', 'Unknown error')}"
            )
            return False

    def perform_search(self, client, openai_api_key, query):
        """
        Perform semantic search using lambda function
        """
        logger.info(f"\n=== Performing Semantic Search ===")
        logger.info(f"Query: '{query}'")

        search_event = {
            "collection_name": self.collection_name,
            "query_text": query,
            "limit": 5,
            "score_threshold": 0.0,
        }

        result = semantic_search(client, search_event, openai_api_key)
        search_data = json.loads(result["body"])

        if result["statusCode"] == 200:
            results = search_data["results"]
            logger.info(f"✓ Found {len(results)} relevant results:")

            for i, result in enumerate(results, 1):
                logger.info(f"\n--- Result {i} ---")
                logger.info(f"Page: {result['page_number']}")
                logger.info(f"Relevance Score: {result['relevance_score']:.4f}")
                logger.info(f"Text Preview: {result['text'][:200]}...")
                if len(result["text"]) > 200:
                    logger.info("(truncated)")

            return results
        else:
            logger.info(f"✗ Search failed: {search_data.get('error', 'Unknown error')}")
            return []

    def perform_page_search(self, client, openai_api_key, query, page_number):
        """
        Perform page-specific search using lambda function
        """
        logger.info(f"\n=== Performing Page-Specific Search ===")
        logger.info(f"Query: '{query}'")
        logger.info(f"Page: {page_number}")

        page_search_event = {
            "collection_name": self.collection_name,
            "query_text": query,
            "page_number": page_number,
            "limit": 3,
        }

        result = search_by_page(client, page_search_event, openai_api_key)
        page_data = json.loads(result["body"])

        if result["statusCode"] == 200:
            page_results = page_data["results"]
            logger.info(f"✓ Found {len(page_results)} results in page {page_number}:")

            for i, result in enumerate(page_results, 1):
                logger.info(f"\n--- Page {page_number} Result {i} ---")
                logger.info(f"Relevance Score: {result['relevance_score']:.4f}")
                logger.info(f"Text: {result['text'][:150]}...")

            return page_results
        else:
            logger.info(
                f"✗ Page search failed: {page_data.get('error', 'Unknown error')}"
            )
            return []

    def get_collection_stats(self, client):
        """
        Get collection statistics using lambda functions
        """
        logger.info(f"\n=== Collection Statistics ===")

        # Get chunk count
        count_event = {"collection_name": self.collection_name}
        result = count_chunks(client, count_event)
        count_data = json.loads(result["body"])
        logger.info(f"Total chunks: {count_data['count']}")

        # Get collection info
        info_event = {"collection_name": self.collection_name}
        result = collection_info(client, info_event)
        info_data = json.loads(result["body"])
        logger.info(f"Collection info: {info_data.get('info', {})}")

    def interactive_search(self, client, openai_api_key):
        """
        Interactive search function for testing
        """
        logger.info(f"\n=== Interactive Search Mode ===")
        logger.info("Commands:")
        logger.info("  - Enter search query for semantic search")
        logger.info(
            "  - 'page:N query' for page-specific search (e.g., 'page:5 diabetes symptoms')"
        )
        logger.info("  - 'stats' to show collection statistics")
        logger.info("  - 'quit' to exit")

        while True:
            query = input("\nEnter command: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            if query.lower() == "stats":
                self.get_collection_stats(client)
                continue

            # Check for page-specific search
            if query.lower().startswith("page:"):
                try:
                    parts = query.split(" ", 1)
                    page_part = parts[0][5:]  # Remove 'page:'
                    page_number = int(page_part)
                    search_query = parts[1] if len(parts) > 1 else ""

                    if search_query:
                        self.perform_page_search(
                            client, openai_api_key, search_query, page_number
                        )
                    else:
                        logger.info("Please provide a search query after page number")
                except (ValueError, IndexError):
                    logger.info("Invalid page search format. Use: page:N query")
                continue

            # Regular semantic search
            self.perform_search(client, openai_api_key, query)

    def run(
        self,
        pdf_file_name="BSAVA Drug Formulary.pdf",
        search_query="common symptoms of diabetes in dogs",
    ):
        """
        Main execution method
        """
        logger.info("=== Local PDF Processing and Vector Search ===\n")

        # Get OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.info("✗ OPENAI_API_KEY environment variable not set")
            logger.info("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
            return

        # Create Qdrant client
        logger.info("1. Connecting to Qdrant...")
        client = self.create_local_qdrant_client()
        if not client:
            return

        # Create collection
        logger.info(f"\n2. Creating collection '{self.collection_name}'...")
        create_event = {
            "collection_name": self.collection_name,
            "vector_size": 1536,  # OpenAI text-embedding-ada-002 dimensions
            "distance": "Cosine",
        }

        result = create_collection(client, create_event)
        logger.info(
            f"Collection creation result: {json.loads(result['body'])['message']}"
        )

        # Find PDF file
        pdf_path = Path(self.docs_folder) / pdf_file_name
        if not pdf_path.exists():
            # Try to find any PDF files
            pdf_files = self.find_pdf_files()
            if pdf_files:
                pdf_path = pdf_files[0]  # Use the first PDF found
                logger.info(f"Using PDF file: {pdf_path}")
            else:
                logger.info(
                    f"✗ No PDF files found. Please place a PDF file in {self.docs_folder}"
                )
                return

        # Process PDF
        logger.info(f"\n3. Processing PDF file...")
        success = self.process_single_pdf(pdf_path, client, openai_api_key)
        if not success:
            return

        # Show collection statistics
        self.get_collection_stats(client)

        # Perform initial search
        results = self.perform_search(client, openai_api_key, search_query)

        # Demonstrate page-specific search if results found
        if results:
            target_page = results[0]["page_number"]
            self.perform_page_search(client, openai_api_key, search_query, target_page)

        logger.info(f"\n=== Processing Complete ===")
        logger.info("Starting interactive search mode...")

        # Start interactive mode
        self.interactive_search(client, openai_api_key)


def main():
    """
    Main function to run the PDF processor
    """
    """
    Step 1 : Uncomment the following lines to upload the document locally and perform a search based search_query.
    Step 2 : Comment the rest of the code 
    processor = LocalPDFProcessor()

    # You can customize these parameters
    PDF_FILE_NAME = "BSAVA Drug Formulary.pdf"  # Will look in config.DOCS_FOLDER
    search_query = "common symptoms of diabetes in dogs"

    processor.run(pdf_file_name=PDF_FILE_NAME, search_query=search_query)

    """
    processor = LocalPDFProcessor()

    search_query = "common symptoms of diabetes in dogs"

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.info("✗ OPENAI_API_KEY environment variable not set")
        return

    logger.info("Connecting to Qdrant...")
    client = processor.create_local_qdrant_client()
    if not client:
        return

    logger.info("Perform initial search...")
    results = processor.perform_search(client, openai_api_key, search_query)

    logger.info("Demonstrate page-specific search if results found...")
    if results:
        target_page = results[0]["page_number"]
        processor.perform_page_search(client, openai_api_key, search_query, target_page)

    print(f"\n=== Processing Complete ===")

    # print("Starting interactive search mode...")
    # processor.interactive_search(client, openai_api_key)


if __name__ == "__main__":
    main()
