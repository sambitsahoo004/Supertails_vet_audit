#!/usr/bin/env python3

import os
import json
import time
import argparse
import logging  ## change made here imported logging
from datetime import datetime
from pathlib import Path
import traceback
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

from single_parse import UnifiedTranscriptScorer
from feedback import generate_excel_report


### code for log file creating and configure log
def setup_logging():
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    log_filename = f"logs/batch_processing_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d%m%Y_%H%M",
        handlers=[
            logging.FileHandler(log_filename),  # Sends logs to a file
            logging.StreamHandler(),  # This will still output to console
        ],
    )

    return logging.getLogger(__name__)


def process_single_transcript(transcript_file, scorer, output_dir, excel_file, logger):
    """Process a single transcript file through scoring and feedback generation."""
    logger.info("=" * 60)
    logger.info(f"Processing: {os.path.basename(transcript_file)}")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        # Load transcript
        transcript = scorer.load_transcript(transcript_file)
        if not transcript:
            logger.error(f"Failed to load transcript: {transcript_file}")
            return False

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

        logger.info(f"Successfully processed in {processing_time:.2f} seconds")
        logger.info(f"   Overall Score: {results['overall']['percentage_score']}%")

        for cat, data in results["categories"].items():
            if isinstance(data, dict) and "percentage_score" in data:
                logger.info(
                    f"   {cat.replace('_', ' ').title()}: {data['percentage_score']}%"
                )

        return True

    except Exception as e:
        logger.error(f"Error processing {transcript_file}: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def batch_process_transcripts(transcript_folder, output_dir, excel_file, logger):
    """Process all transcript files in a folder."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.error("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    if not openai_api_key:
        logger.warning(
            "Warning: OPENAI_API_KEY environment variable not set. Category 4 scoring may be limited."
        )

    logger.info("Initializing scorer...")
    scorer = UnifiedTranscriptScorer(api_key=api_key, openai_api_key=openai_api_key)

    os.makedirs(output_dir, exist_ok=True)

    transcript_files = []
    supported_extensions = [".txt", ".md", ".transcript"]

    for ext in supported_extensions:
        transcript_files.extend(Path(transcript_folder).glob(f"*{ext}"))

    for file_path in Path(transcript_folder).iterdir():
        if file_path.is_file() and not file_path.suffix:
            transcript_files.append(file_path)

    transcript_files = [str(f) for f in transcript_files]
    transcript_files.sort()  # Process in alphabetical order

    if not transcript_files:
        logger.error(f"No transcript files found in {transcript_folder}")
        logger.error(
            f"   Looking for files with extensions: {supported_extensions} or no extension"
        )
        return

    logger.info(f"Found {len(transcript_files)} transcript files to process")

    # Process each file
    successful = 0
    failed = 0
    validation_failed = 0
    batch_start_time = time.time()

    for i, transcript_file in enumerate(transcript_files, 1):
        logger.info(f"Processing file {i}/{len(transcript_files)}")

        success = process_single_transcript(
            transcript_file=transcript_file,
            scorer=scorer,
            output_dir=output_dir,
            excel_file=excel_file,
            logger=logger,
        )

        if success:
            successful += 1
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


def main():
    """Main function for batch processing."""
    # Setup logging first
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Batch process veterinary call transcripts"
    )
    parser.add_argument(
        "transcript_folder", help="Path to folder containing transcript files"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for JSON results",
        default="batch_results",
    )

    # Generate timestamped default filename
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    default_output_dir = "output"  ##this will help us to save results in output folder
    default_filename = f"veterinary_call_report_{timestamp}.xlsx"

    parser.add_argument(
        "--excel-file",
        "-e",
        type=str,
        default=default_filename,
        help=f"Path to save the Excel report (default: {default_filename})",
    )

    parser.add_argument(
        "--model", help="LLM model to use", default="claude-3-5-sonnet-latest"
    )

    args = parser.parse_args()

    # Set default Excel file path inside output-dir if not specified
    if not args.excel_file:
        args.excel_file = os.path.join(args.output_dir, default_filename)

    # Validate input folder
    if not os.path.exists(args.transcript_folder):
        logger.error(
            f"Error: Transcript folder '{args.transcript_folder}' does not exist"
        )
        return

    if not os.path.isdir(args.transcript_folder):
        logger.error(f"Error: '{args.transcript_folder}' is not a directory")
        return

    logger.info(
        f"Starting batch processing at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Input folder: {args.transcript_folder}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Excel file: {args.excel_file}")

    # Start batch processing
    batch_process_transcripts(
        transcript_folder=args.transcript_folder,
        output_dir=args.output_dir,
        excel_file=args.excel_file,
        logger=logger,
    )


if __name__ == "__main__":
    main()
