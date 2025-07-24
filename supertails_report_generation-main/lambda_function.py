#!/usr/bin/env python3

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import traceback

from single_parse import UnifiedTranscriptScorer
from feedback import generate_excel_report
from s3_handler import S3Handler

from config import (
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    GCP_SERVICE_ACCOUNT_KEY,
    AWS_S3_BUCKET_NAME,
    AWS_S3_CLIENT_STORE_DIRECTORY,
    DOCS_FOLDER,
    CHUNKS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PARAMETERS_CONFIG,
)

# Initialize S3 handler
s3_handler = S3Handler()


def setup_logging():
    """Setup logging configuration for Lambda."""
    # Lambda automatically handles log files through CloudWatch
    # Configure logging for CloudWatch
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def process_single_transcript(transcript_file, scorer, output_dir, excel_file, logger):
    """Process a single transcript file through scoring and feedback generation."""
    logger.info("=" * 60)
    logger.info(f"Processing: {os.path.basename(transcript_file)}")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        # Initialize scorer if not provided
        if scorer is None:
            # Use API keys from config.py, fallback to environment variables
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            openai_api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")

            if not api_key:
                logger.error("Error: ANTHROPIC_API_KEY environment variable not set")
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")

            if not openai_api_key:
                logger.warning(
                    "Warning: OPENAI_API_KEY not found in config or environment variables. Category 4 scoring may be limited."
                )

            logger.info("Initializing scorer...")
            scorer = UnifiedTranscriptScorer(
                api_key=api_key, openai_api_key=openai_api_key
            )

        # Load transcript
        transcript = scorer.load_transcript(transcript_file)
        if not transcript:
            logger.error(f"Failed to load transcript: {transcript_file}")
            return False, None

        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(transcript_file))[0]
        json_output = os.path.join(output_dir, f"{base_name}_scores.json")

        logger.info("Scoring transcript...")
        results = scorer.score_transcript(transcript)

        # Check if validation failed (e.g., duration too short)
        if results.get("validation_failed"):
            logger.warning(
                f"Validation failed for {transcript_file}: {results.get('message', 'Unknown validation error')}"
            )

            # Save the validation failure result
            scorer.save_results(results, json_output)

            # You can choose to either:
            # 1. Return False to count as failed
            # 2. Return True to count as processed but with validation failure
            # 3. Skip Excel generation for failed validations

            logger.info("Skipping Excel report generation due to validation failure")
            return False  # Counting as failed processing

        # Save JSON results
        scorer.save_results(results, json_output)

        logger.info("Generating Excel feedback...")
        # Generate Excel report (append to main Excel file)
        generate_excel_report(results, transcript, excel_file)

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info("Processing completed successfully")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Results saved to: {json_output}")
        logger.info(f"Excel report updated: {excel_file}")

        return True, results

    except Exception as e:
        logger.error(f"Error processing {transcript_file}: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None


# def batch_process_transcripts(transcript_folder, output_dir, excel_file, logger):
#     """Process transcript files."""

#     # Use API keys from config.py, fallback to environment variables
#     api_key = os.environ.get("ANTHROPIC_API_KEY")
#     openai_api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")

#     if not api_key:
#         logger.error("Error: ANTHROPIC_API_KEY environment variable not set")
#         raise ValueError("ANTHROPIC_API_KEY environment variable not set")

#     if not openai_api_key:
#         logger.warning(
#             "Warning: OPENAI_API_KEY not found in config or environment variables. Category 4 scoring may be limited."
#         )

#     logger.info("Initializing scorer...")
#     scorer = UnifiedTranscriptScorer(api_key=api_key, openai_api_key=openai_api_key)

#     os.makedirs(output_dir, exist_ok=True)

#     transcript_files = []
#     supported_extensions = [".txt", ".md", ".transcript"]

#     for ext in supported_extensions:
#         transcript_files.extend(Path(transcript_folder).glob(f"*{ext}"))

#     for file_path in Path(transcript_folder).iterdir():
#         if file_path.is_file() and not file_path.suffix:
#             transcript_files.append(file_path)

#     transcript_files = [str(f) for f in transcript_files]
#     transcript_files.sort()  # Process in alphabetical order

#     if not transcript_files:
#         logger.error("No transcript files provided")
#         raise ValueError("No transcript files provided")

#     logger.info(f"Found {len(transcript_files)} transcript files to process")

#     # Process each file
#     successful = 0
#     failed = 0
#     batch_start_time = time.time()
#     all_results = []

#     for i, transcript_file in enumerate(transcript_files, 1):
#         logger.info(f"Processing file {i}/{len(transcript_files)}")

#         success, results = process_single_transcript(
#             transcript_file=transcript_file,
#             scorer=scorer,
#             output_dir=output_dir,
#             excel_file=excel_file,
#             logger=logger,
#         )

#         if success:
#             successful += 1
#             all_results.append(
#                 {"file": os.path.basename(transcript_file), "results": results}
#             )
#         else:
#             failed += 1

#         # Add a small delay to avoid API rate limits
#         if i < len(transcript_files):  # Don't sleep after the last file
#             logger.info("Waiting 2 seconds before next file...")
#             time.sleep(2)

#     # Print final summary using logging
#     batch_end_time = time.time()
#     total_time = batch_end_time - batch_start_time

#     logger.info("=" * 80)
#     logger.info("BATCH PROCESSING COMPLETE")
#     logger.info("=" * 80)
#     logger.info("Summary:")
#     logger.info(f"   Total files processed: {len(transcript_files)}")
#     logger.info(f"   Successful: {successful}")
#     logger.info(f"   Failed: {failed}")
#     logger.info(
#         f"   Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
#     )
#     logger.info(
#         f"   Average time per file: {total_time/len(transcript_files):.2f} seconds"
#     )
#     logger.info("Results saved to:")
#     logger.info(f"   JSON files: {output_dir}")
#     logger.info(f"   Excel report: {excel_file}")

#     return {
#         "successful": successful,
#         "failed": failed,
#         "total_time": total_time,
#         "results": all_results,
#     }


def batch_process_transcripts(transcript_files, output_dir, excel_file, logger):
    """Process transcript files."""

    # Use API keys from config.py, fallback to environment variables
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.error("Error: ANTHROPIC_API_KEY environment variable not set")
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    if not openai_api_key:
        logger.warning(
            "Warning: OPENAI_API_KEY not found in config or environment variables. Category 4 scoring may be limited."
        )

    logger.info("Initializing scorer...")
    scorer = UnifiedTranscriptScorer(api_key=api_key, openai_api_key=openai_api_key)

    os.makedirs(output_dir, exist_ok=True)

    # transcript_files is already a list of file paths from lambda_handler
    # No need to search for files - they're already provided
    if not transcript_files:
        logger.error("No transcript files provided")
        raise ValueError("No transcript files provided")

    logger.info(f"Found {len(transcript_files)} transcript files to process")

    # Process each file
    successful = 0
    failed = 0
    validation_failed = 0
    batch_start_time = time.time()
    all_results = []

    for i, transcript_file in enumerate(transcript_files, 1):
        logger.info(f"Processing file {i}/{len(transcript_files)}")

        success, results = process_single_transcript(
            transcript_file=transcript_file,
            scorer=scorer,
            output_dir=output_dir,
            excel_file=excel_file,
            logger=logger,
        )

        if success:
            successful += 1
            all_results.append(
                {"file": os.path.basename(transcript_file), "results": results}
            )
        else:
            failed += 1
            # Check if it was a validation failure by looking at the JSON output
            base_name = os.path.splitext(os.path.basename(transcript_file))[0]
            json_output = os.path.join(output_dir, f"{base_name}_scores.json")
            if os.path.exists(json_output):
                try:
                    with open(json_output, "r") as f:
                        result_data = json.load(f)
                        if result_data.get("validation_failed"):
                            validation_failed += 1
                except:
                    pass  # If we can't read the file, just continue

        # Add a small delay to avoid API rate limits
        if i < len(transcript_files):  # Don't sleep after the last file
            logger.info("Waiting 2 seconds before next file...")
            time.sleep(2)

    # Print final summary using logging
    batch_end_time = time.time()
    total_time = batch_end_time - batch_start_time

    logger.info("=" * 80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info(f"   Total files processed: {len(transcript_files)}")
    logger.info(f"   Successful: {successful}")
    logger.info(f"   Failed: {failed}")
    if validation_failed > 0:
        logger.info(
            f"   Failed due to validation (duration/other): {validation_failed}"
        )
        logger.info(f"   Failed due to processing errors: {failed - validation_failed}")
    logger.info(
        f"   Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    logger.info(
        f"   Average time per file: {total_time/len(transcript_files):.2f} seconds"
    )
    logger.info("Results saved to:")
    logger.info(f"   JSON files: {output_dir}")
    logger.info(f"   Excel report: {excel_file}")

    return {
        "successful": successful,
        "failed": failed,
        "total_time": total_time,
        "results": all_results,
    }


def lambda_handler(event, context):
    """
    AWS Lambda handler function.

    Expected event structure:
    {
        "source_bucket": "your-input-bucket",
        "source_prefix": "transcripts/",  # Optional: folder prefix
        "destination_bucket": "your-output-bucket",
        "destination_prefix": "results/",  # Optional: output folder prefix
        "transcript_files": ["file1.txt", "file2.txt"],  # Optional: specific files
        "excel_filename": "custom_report.xlsx",  # Optional: custom Excel filename
        "use_config_defaults": true  # Optional: use config.py defaults for S3 settings
    }
    """

    # Setup logging
    logger = setup_logging()

    try:
        logger.info(
            f"Lambda function started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(f"Event received: {json.dumps(event, indent=2)}")

        # Health check endpoint
        if event.get("operation") == "health_check":
            return {
                "statusCode": 200,
                "body": {
                    "message": "Lambda function is healthy",
                    "timestamp": datetime.now().isoformat(),
                    "configuration": {
                        "aws_s3_bucket": AWS_S3_BUCKET_NAME,
                        "aws_s3_client_store_directory": AWS_S3_CLIENT_STORE_DIRECTORY,
                        "parameters_count": len(PARAMETERS_CONFIG["parameters"]),
                    },
                },
            }

        # Log configuration parameters being used
        logger.info("Configuration loaded:")
        logger.info(f"   AWS S3 Bucket: {AWS_S3_BUCKET_NAME}")
        logger.info(f"   AWS S3 Directory: {AWS_S3_CLIENT_STORE_DIRECTORY}")
        logger.info(f"   Chunk Size: {CHUNK_SIZE}")
        logger.info(f"   Chunk Overlap: {CHUNK_OVERLAP}")
        logger.info(f"   Total Parameters: {len(PARAMETERS_CONFIG['parameters'])}")

        # Extract parameters from event with config.py defaults
        use_config_defaults = event.get("use_config_defaults", False)

        if use_config_defaults:
            source_bucket = event.get("source_bucket", AWS_S3_BUCKET_NAME)
            destination_bucket = event.get("destination_bucket", AWS_S3_BUCKET_NAME)
            source_prefix = event.get(
                "source_prefix", f"{AWS_S3_CLIENT_STORE_DIRECTORY}/transcripts/"
            )
            destination_prefix = event.get(
                "destination_prefix", f"{AWS_S3_CLIENT_STORE_DIRECTORY}/results/"
            )
        else:
            source_bucket = event.get("source_bucket")
            destination_bucket = event.get("destination_bucket", source_bucket)
            source_prefix = event.get("source_prefix", "")
            destination_prefix = event.get("destination_prefix", "results/")

        specific_files = event.get("transcript_files", [])

        # Validate required parameters
        if not source_bucket:
            raise ValueError(
                "source_bucket is required in event or set use_config_defaults=true"
            )

        # Generate timestamped filenames
        timestamp = datetime.now().strftime("%d%m%Y_%H%M")
        excel_filename = event.get(
            "excel_filename", f"veterinary_call_report_{timestamp}.xlsx"
        )

        # Create temporary directories in Lambda's /tmp space
        local_input_dir = "/tmp/input"
        local_output_dir = "/tmp/output"
        os.makedirs(local_input_dir, exist_ok=True)
        os.makedirs(local_output_dir, exist_ok=True)

        excel_file_path = os.path.join(local_output_dir, excel_filename)

        # Get list of files to process
        transcript_files = []

        if specific_files:
            # Process specific files mentioned in event
            for filename in specific_files:
                s3_key = f"{source_prefix}{filename}" if source_prefix else filename
                local_file_path = os.path.join(local_input_dir, filename)

                if s3_handler.download_file(source_bucket, s3_key, local_file_path):
                    transcript_files.append(local_file_path)
                else:
                    logger.warning(f"Failed to download {s3_key}")
        else:
            # List all files in the S3 prefix
            supported_extensions = [".txt", ".md", ".transcript"]

            try:
                objects = s3_handler.list_objects(source_bucket, source_prefix)

                for obj in objects:
                    key = obj["Key"]
                    filename = os.path.basename(key)

                    # Check if file has supported extension or no extension
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in supported_extensions or not file_ext:
                        local_file_path = os.path.join(local_input_dir, filename)

                        if s3_handler.download_file(
                            source_bucket, key, local_file_path
                        ):
                            transcript_files.append(local_file_path)
                        else:
                            logger.warning(f"Failed to download {key}")

            except Exception as e:
                logger.error(f"Error listing objects in S3: {str(e)}")
                raise

        if not transcript_files:
            raise ValueError("No transcript files found to process")

        logger.info(f"Input bucket: {source_bucket}")
        logger.info(f"Output bucket: {destination_bucket}")
        logger.info(f"Excel file: {excel_filename}")

        # Process transcripts
        if len(transcript_files) == 1:
            # Single file processing
            logger.info("Processing single transcript file...")
            success, results = process_single_transcript(
                transcript_file=transcript_files[0],
                scorer=None,  # Will be initialized inside the function
                output_dir=local_output_dir,
                excel_file=excel_file_path,
                logger=logger,
            )

            if success:
                processing_results = {
                    "successful": 1,
                    "failed": 0,
                    "total_time": 0,  # Will be calculated
                    "results": [
                        {
                            "file": os.path.basename(transcript_files[0]),
                            "results": results,
                        }
                    ],
                }
            else:
                processing_results = {
                    "successful": 0,
                    "failed": 1,
                    "total_time": 0,
                    "results": [],
                }
        else:
            # Batch processing for multiple files
            logger.info(
                f"Processing {len(transcript_files)} transcript files in batch..."
            )
            processing_results = batch_process_transcripts(
                transcript_files=transcript_files,
                output_dir=local_output_dir,
                excel_file=excel_file_path,
                logger=logger,
            )

        # Upload results back to S3
        logger.info("Uploading results to S3...")

        # Upload Excel file
        excel_s3_key = f"{destination_prefix}{excel_filename}"
        if s3_handler.upload_file(excel_file_path, destination_bucket, excel_s3_key):
            logger.info(
                f"Excel report uploaded to s3://{destination_bucket}/{excel_s3_key}"
            )
        else:
            logger.error("Failed to upload Excel report")

        # Upload JSON files
        for json_file in os.listdir(local_output_dir):
            if json_file.endswith(".json"):
                local_json_path = os.path.join(local_output_dir, json_file)
                json_s3_key = f"{destination_prefix}json/{json_file}"

                if s3_handler.upload_file(
                    local_json_path, destination_bucket, json_s3_key
                ):
                    logger.info(
                        f"JSON file uploaded to s3://{destination_bucket}/{json_s3_key}"
                    )
                else:
                    logger.warning(f"Failed to upload {json_file}")

        # Prepare response with configuration info
        response = {
            "statusCode": 200,
            "body": {
                "message": "Batch processing completed successfully",
                "summary": processing_results,
                "configuration": {
                    "parameters_count": len(PARAMETERS_CONFIG["parameters"]),
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "aws_s3_bucket": AWS_S3_BUCKET_NAME,
                    "aws_s3_directory": AWS_S3_CLIENT_STORE_DIRECTORY,
                },
                "output_files": {
                    "excel_report": f"s3://{destination_bucket}/{excel_s3_key}",
                    "json_results": f"s3://{destination_bucket}/{destination_prefix}json/",
                },
                "timestamp": datetime.now().isoformat(),
            },
        }

        logger.info("Lambda function completed successfully")
        return response

    except Exception as e:
        logger.error(f"Lambda function failed: {str(e)}")
        logger.error(traceback.format_exc())

        return {
            "statusCode": 500,
            "body": {"error": str(e), "timestamp": datetime.now().isoformat()},
        }


# For local testing
if __name__ == "__main__":
    # Example event for local testing with config defaults
    test_event = {
        "use_config_defaults": True,  # Use config.py defaults
        "source_prefix": "transcripts/",
        "destination_prefix": "results/",
        "transcript_files": [
            "anusree_s_supertails_in__Pharmacy_OB__320__-1__6389215727__2025-05-04_17-22-58_diarization_1746871972.txt"
        ],  # Optional
    }

    # Alternative test event with explicit buckets
    # test_event_explicit = {
    #     "source_bucket": "your-transcript-bucket",
    #     "source_prefix": "transcripts/",
    #     "destination_bucket": "your-results-bucket",
    #     "destination_prefix": "results/",
    #     "transcript_files": ["sample1.txt", "sample2.txt"],  # Optional
    # }

    test_context = {}

    print("Testing with config defaults...")
    result = lambda_handler(test_event, test_context)
    print(json.dumps(result, indent=2))
