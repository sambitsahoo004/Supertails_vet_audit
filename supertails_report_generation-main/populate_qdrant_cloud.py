#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to populate Qdrant Cloud with veterinary knowledge chunks.
This script processes PDF files and indexes them in Qdrant Cloud for use in Lambda functions.
"""
import os
import json
import time
from pathlib import Path

from document_chunker import VeterinaryDocumentChunker
from knowledge_base import CloudVeterinaryKnowledgeBase
import config


def populate_qdrant_cloud():
    """Populate Qdrant Cloud with veterinary knowledge chunks."""
    print("=" * 80)
    print("QDrant Cloud Population Script")
    print("=" * 80)

    # Check required environment variables
    required_vars = ["QDRANT_URL", "QDRANT_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these environment variables before running the script.")
        return False

    try:
        # Initialize knowledge base
        print("Initializing Qdrant Cloud knowledge base...")
        kb = CloudVeterinaryKnowledgeBase(
            collection_name=config.COLLECTION_NAME,
            qdrant_url=os.environ.get("QDRANT_URL"),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Check if collection exists and get info
        collection_info = kb.get_collection_info()
        if collection_info:
            print(
                f"Collection '{config.COLLECTION_NAME}' exists with {collection_info.get('points_count', 0)} points"
            )

            # Ask user if they want to recreate
            response = (
                input("Do you want to recreate the collection? (y/N): ").strip().lower()
            )
            if response == "y":
                print("Deleting existing collection...")
                kb.delete_collection()
                time.sleep(2)  # Wait for deletion to complete
            else:
                print("Using existing collection. Adding new chunks...")

        # Define PDF directory
        pdf_dir = os.path.join(config.DOCS_FOLDER, "pdf")
        if not os.path.exists(pdf_dir):
            print(f"❌ PDF directory not found: {pdf_dir}")
            return False

        # Get PDF files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_dir}")
            return False

        print(f"Found {len(pdf_files)} PDF files to process")

        # Initialize chunker
        chunker = VeterinaryDocumentChunker(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )

        # Process each PDF file
        all_chunks = []
        total_start_time = time.time()

        for i, pdf_file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"\nProcessing file {i}/{len(pdf_files)}: {pdf_file}")

            # Process PDF
            chunks = chunker.process_pdf_file(pdf_path)
            if chunks:
                all_chunks.extend(chunks)
                print(f"Added {len(chunks)} chunks from {pdf_file}")
            else:
                print(f"⚠️ No chunks created from {pdf_file}")

        if not all_chunks:
            print("❌ No chunks were created from any PDF files")
            return False

        print(f"\nTotal chunks created: {len(all_chunks)}")

        # Index chunks in Qdrant Cloud
        print("\nIndexing chunks in Qdrant Cloud...")
        indexing_start_time = time.time()

        kb.index_chunks(all_chunks)

        indexing_time = time.time() - indexing_start_time
        total_time = time.time() - total_start_time

        # Verify indexing
        final_count = kb.count_chunks()
        print(f"\n✅ Indexing completed!")
        print(f"   Total chunks indexed: {final_count}")
        print(f"   Indexing time: {indexing_time:.1f} seconds")
        print(f"   Total processing time: {total_time:.1f} seconds")

        # Test search
        print("\nTesting search functionality...")
        test_query = "medicine dosage"
        results = kb.search(test_query, limit=3)
        print(f"Test search for '{test_query}' returned {len(results)} results")

        if results:
            print("Sample result:")
            print(f"  Text: {results[0]['text'][:200]}...")
            print(f"  Score: {results[0]['score']}")

        print("\n✅ Qdrant Cloud population completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error populating Qdrant Cloud: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_qdrant_cloud_connection():
    """Test Qdrant Cloud connection and search."""
    print("\n" + "=" * 80)
    print("Testing Qdrant Cloud Connection")
    print("=" * 80)

    try:
        kb = CloudVeterinaryKnowledgeBase(
            collection_name=config.COLLECTION_NAME,
            qdrant_url=os.environ.get("QDRANT_URL"),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Get collection info
        info = kb.get_collection_info()
        print(f"✅ Collection info: {info}")

        # Test search
        test_queries = [
            "medicine dosage",
            "hip dysplasia treatment",
            "Golden Retriever care",
        ]

        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            results = kb.search(query, limit=2)
            print(f"  Results: {len(results)}")
            if results:
                print(f"  Top result score: {results[0]['score']:.4f}")
                print(f"  Top result text: {results[0]['text'][:100]}...")

        print("\n✅ Qdrant Cloud connection test successful!")
        return True

    except Exception as e:
        print(f"❌ Qdrant Cloud connection test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        success = test_qdrant_cloud_connection()
    else:
        # Population mode
        success = populate_qdrant_cloud()

    sys.exit(0 if success else 1)
