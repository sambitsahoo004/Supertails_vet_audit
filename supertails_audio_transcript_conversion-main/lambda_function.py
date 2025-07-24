#!/usr/bin/env python3

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import traceback
import glob

from s3_handler import S3Handler
from audioconversion import transcribe_and_diarize_mp3_two_speakers

from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    AWS_S3_BUCKET_NAME,
    AWS_S3_CLIENT_STORE_DIRECTORY,
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


def process_audio_file(mp3_file_path, local_output_dir, logger):
    """
    Process a single audio file using the audioconversion functionality.

    Args:
        mp3_file_path: Path to the MP3 file to process
        local_output_dir: Directory to save output files
        logger: Logger instance

    Returns:
        dict: Processing results including file paths and metadata
    """
    try:
        logger.info(f"Processing audio file: {os.path.basename(mp3_file_path)}")

        # Process the audio file using the audioconversion function
        results = transcribe_and_diarize_mp3_two_speakers(
            mp3_file_path, local_output_dir
        )

        if results:
            # Find the generated diarization file
            audio_filename = os.path.basename(mp3_file_path)
            audio_filename_no_ext = os.path.splitext(audio_filename)[0]

            # Look for the diarization file in the output directory
            diarization_pattern = os.path.join(
                local_output_dir, f"{audio_filename_no_ext}_diarization_*.txt"
            )
            diarization_files = glob.glob(diarization_pattern)

            if diarization_files:
                diarization_file_path = diarization_files[0]  # Take the first match
                logger.info(
                    f"Diarization file generated: {os.path.basename(diarization_file_path)}"
                )

                return {
                    "success": True,
                    "input_file": mp3_file_path,
                    "diarization_file": diarization_file_path,
                    "transcription_file": None,  # Will be handled separately if needed
                    "duration_seconds": results.get("duration_seconds", 0),
                    "num_speakers": results.get("processing_info", {}).get(
                        "num_speakers", 0
                    ),
                    "segments_count": len(results.get("segments", [])),
                    "processing_info": results.get("processing_info", {}),
                }
            else:
                logger.error(f"No diarization file found for {audio_filename}")
                return {
                    "success": False,
                    "input_file": mp3_file_path,
                    "error": "No diarization file generated",
                }
        else:
            logger.error(f"Audio processing failed for {mp3_file_path}")
            return {
                "success": False,
                "input_file": mp3_file_path,
                "error": "Audio processing returned no results",
            }

    except Exception as e:
        logger.error(f"Error processing {mp3_file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "input_file": mp3_file_path, "error": str(e)}


def lambda_handler(event, context):
    """
    AWS Lambda handler function.

    Expected event structure:
    {
        "source_bucket": "your-input-bucket",
        "source_prefix": "mp3/",  # Optional: folder prefix
        "destination_bucket": "your-output-bucket",
        "destination_prefix": "dializarition/",  # Optional: output folder prefix
        "mp3_files": ["audio1.mp3", "audio2.mp3"],  # Optional: specific files
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
                    },
                },
            }

        # Log configuration parameters being used
        logger.info("Configuration loaded:")
        logger.info(f"   AWS S3 Bucket: {AWS_S3_BUCKET_NAME}")
        logger.info(f"   AWS S3 Directory: {AWS_S3_CLIENT_STORE_DIRECTORY}")

        # Extract parameters from event with config.py defaults
        use_config_defaults = event.get("use_config_defaults", False)

        if use_config_defaults:
            source_bucket = event.get("source_bucket", AWS_S3_BUCKET_NAME)
            destination_bucket = event.get("destination_bucket", AWS_S3_BUCKET_NAME)
            source_prefix = event.get(
                "source_prefix", f"{AWS_S3_CLIENT_STORE_DIRECTORY}/mp3/"
            )
            destination_prefix = event.get(
                "destination_prefix", f"{AWS_S3_CLIENT_STORE_DIRECTORY}/dializarition/"
            )
        else:
            source_bucket = event.get("source_bucket")
            destination_bucket = event.get("destination_bucket", source_bucket)
            source_prefix = event.get("source_prefix", "")
            destination_prefix = event.get("destination_prefix", "dializarition/")

        specific_files = event.get("mp3_files", [])

        # Validate required parameters
        if not source_bucket:
            raise ValueError(
                "source_bucket is required in event or set use_config_defaults=true"
            )

        # Create temporary directories in Lambda's /tmp space
        local_input_dir = "/tmp/input"
        local_output_dir = "/tmp/output"
        os.makedirs(local_input_dir, exist_ok=True)
        os.makedirs(local_output_dir, exist_ok=True)

        # Get list of files to process
        mp3_files = []

        if specific_files:
            # Process specific files mentioned in event
            for filename in specific_files:
                s3_key = f"{source_prefix}{filename}" if source_prefix else filename
                local_file_path = os.path.join(local_input_dir, filename)

                if s3_handler.download_file(source_bucket, s3_key, local_file_path):
                    mp3_files.append(local_file_path)
                    logger.info(f"Downloaded {s3_key} to {local_file_path}")
                else:
                    logger.warning(f"Failed to download {s3_key}")
        else:
            # List all files in the S3 prefix
            supported_extensions = [".mp3", ".wav", ".flac"]

            try:
                objects = s3_handler.list_objects(source_bucket, source_prefix)

                for obj in objects:
                    key = obj["Key"]
                    filename = os.path.basename(key)

                    # Check if file has supported extension
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in supported_extensions:
                        local_file_path = os.path.join(local_input_dir, filename)

                        if s3_handler.download_file(
                            source_bucket, key, local_file_path
                        ):
                            mp3_files.append(local_file_path)
                            logger.info(f"Downloaded {key} to {local_file_path}")
                        else:
                            logger.warning(f"Failed to download {key}")

            except Exception as e:
                logger.error(f"Error listing objects in S3: {str(e)}")
                raise

        if not mp3_files:
            raise ValueError("No audio files found to process")

        logger.info(f"Found {len(mp3_files)} audio files to process")
        logger.info(f"Input bucket: {source_bucket}")
        logger.info(f"Output bucket: {destination_bucket}")

        # Process audio files
        start_time = time.time()
        processing_results = {
            "successful": 0,
            "failed": 0,
            "total_time": 0,
            "results": [],
            "uploaded_files": [],
        }

        for i, mp3_file in enumerate(mp3_files):
            logger.info(
                f"Processing file {i+1}/{len(mp3_files)}: {os.path.basename(mp3_file)}"
            )

            # Process the audio file
            result = process_audio_file(mp3_file, local_output_dir, logger)

            if result["success"]:
                processing_results["successful"] += 1

                # Upload the diarization file to S3 at the correct destination prefix
                diarization_file_path = result["diarization_file"]
                diarization_filename = os.path.basename(diarization_file_path)
                # Ensure destination_prefix ends with a slash
                if not destination_prefix.endswith("/"):
                    destination_prefix += "/"
                s3_key_diar = f"{destination_prefix}{diarization_filename}"

                # Always upload to source_bucket (per user requirement)
                if s3_handler.upload_file(
                    diarization_file_path, source_bucket, s3_key_diar
                ):
                    logger.info(
                        f"Uploaded diarization file to s3://{source_bucket}/{s3_key_diar}"
                    )
                    processing_results["uploaded_files"].append(
                        {
                            "file": diarization_filename,
                            "s3_location": f"s3://{source_bucket}/{s3_key_diar}",
                            "type": "diarization",
                        }
                    )
                else:
                    logger.error(
                        f"Failed to upload diarization file: {diarization_filename}"
                    )

                # Upload the plain transcript file to S3 at the same destination prefix
                transcription_file_path = result.get("transcription_file")
                if transcription_file_path and os.path.exists(transcription_file_path):
                    transcription_filename = os.path.basename(transcription_file_path)
                    s3_key_trans = f"{destination_prefix}{transcription_filename}"
                    if s3_handler.upload_file(
                        transcription_file_path, source_bucket, s3_key_trans
                    ):
                        logger.info(
                            f"Uploaded transcript file to s3://{source_bucket}/{s3_key_trans}"
                        )
                        processing_results["uploaded_files"].append(
                            {
                                "file": transcription_filename,
                                "s3_location": f"s3://{source_bucket}/{s3_key_trans}",
                                "type": "transcript",
                            }
                        )
                    else:
                        logger.error(
                            f"Failed to upload transcript file: {transcription_filename}"
                        )

                # Add to results
                processing_results["results"].append(
                    {
                        "file": os.path.basename(mp3_file),
                        "success": True,
                        "diarization_file": diarization_filename,
                        "transcription_file": (
                            transcription_filename if transcription_file_path else None
                        ),
                        "duration_seconds": result.get("duration_seconds", 0),
                        "num_speakers": result.get("num_speakers", 0),
                        "segments_count": result.get("segments_count", 0),
                    }
                )
            else:
                processing_results["failed"] += 1
                processing_results["results"].append(
                    {
                        "file": os.path.basename(mp3_file),
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                    }
                )

        # Calculate total processing time
        processing_results["total_time"] = time.time() - start_time

        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        for mp3_file in mp3_files:
            try:
                os.remove(mp3_file)
            except:
                pass

        # Prepare response with configuration info
        response = {
            "statusCode": 200,
            "body": {
                "message": "Audio processing completed successfully",
                "summary": processing_results,
                "configuration": {
                    "aws_s3_bucket": AWS_S3_BUCKET_NAME,
                    "aws_s3_directory": AWS_S3_CLIENT_STORE_DIRECTORY,
                },
                "output_files": {
                    "dializarition_files": f"s3://{destination_bucket}/{destination_prefix}",
                    "uploaded_count": len(processing_results["uploaded_files"]),
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
        "source_prefix": "mp3/",
        "destination_prefix": "dializarition/",
        "mp3_files": [
            "anusree_s_supertails_in__Pharmacy_OB__320__-1__6389215727__2025-05-04_17-22-58.mp3"
        ],  # Optional
    }

    # Context is not used in this example, but can be provided if needed
    test_context = {}

    print("Testing with config defaults...")
    result = lambda_handler(test_event, test_context)
    print(json.dumps(result, indent=2))
