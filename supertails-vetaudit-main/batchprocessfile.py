#!/usr/bin/env python3

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from single_parse import UnifiedTranscriptScorer
from feedback import generate_excel_report


def process_single_transcript(transcript_file, scorer, output_dir, excel_file):
    """Process a single transcript file through scoring and feedback generation."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(transcript_file)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Load transcript
        transcript = scorer.load_transcript(transcript_file)
        if not transcript:
            print(f"❌ Failed to load transcript: {transcript_file}")
            return False

        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(transcript_file))[0]
        json_output = os.path.join(output_dir, f"{base_name}_scores.json")

        print(f"📊 Scoring transcript...")
        results = scorer.score_transcript(transcript)

        # Save JSON results
        scorer.save_results(results, json_output)

        print(f"📝 Generating Excel feedback...")
        # Generate Excel report (append to main Excel file)
        generate_excel_report(results, transcript, excel_file)

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"✅ Successfully processed in {processing_time:.2f} seconds")
        print(f"   Overall Score: {results['overall']['percentage_score']}%")

        for cat, data in results["categories"].items():
            if isinstance(data, dict) and "percentage_score" in data:
                print(
                    f"   {cat.replace('_', ' ').title()}: {data['percentage_score']}%"
                )

        return True

    except Exception as e:
        print(f"❌ Error processing {transcript_file}: {str(e)}")
        traceback.print_exc()
        return False


def batch_process_transcripts(transcript_folder, output_dir, excel_file):
    """Process all transcript files in a folder."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        return

    if not openai_api_key:
        print(
            "⚠️ Warning: OPENAI_API_KEY environment variable not set. Category 4 scoring may be limited."
        )

    print("🔧 Initializing scorer...")
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
        print(f"❌ No transcript files found in {transcript_folder}")
        print(
            f"   Looking for files with extensions: {supported_extensions} or no extension"
        )
        return

    print(f"📁 Found {len(transcript_files)} transcript files to process")

    # Process each file
    successful = 0
    failed = 0
    batch_start_time = time.time()

    for i, transcript_file in enumerate(transcript_files, 1):
        print(f"\n🔄 Processing file {i}/{len(transcript_files)}")

        success = process_single_transcript(
            transcript_file=transcript_file,
            scorer=scorer,
            output_dir=output_dir,
            excel_file=excel_file,
        )

        if success:
            successful += 1
        else:
            failed += 1

        # Add a small delay to avoid API rate limits
        if i < len(transcript_files):  # Don't sleep after the last file
            print("⏳ Waiting 2 seconds before next file...")
            time.sleep(2)

    # Print final summary
    batch_end_time = time.time()
    total_time = batch_end_time - batch_start_time

    print(f"\n{'='*80}")
    print(f"🏁 BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Summary:")
    print(f"   Total files processed: {len(transcript_files)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"   Average time per file: {total_time/len(transcript_files):.2f} seconds")
    print(f"📄 Results saved to:")
    print(f"   JSON files: {output_dir}")
    print(f"   Excel report: {excel_file}")


def main():
    """Main function for batch processing."""
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
        default="output/batch_results",
    )

    # Generate timestamped default filename
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    default_output_dir = "output"  ##this will help us to save results in output folder
    default_excel_filename = f"veterinary_call_report_{timestamp}.xlsx"

    default_excel_filepath = os.path.join(default_output_dir, default_excel_filename)

    parser.add_argument(
        "--excel-file",
        "-e",
        type=str,
        default=default_excel_filepath,
        help=f"Path to save the Excel report (default: {default_excel_filepath})",
    )

    parser.add_argument(
        "--model", help="LLM model to use", default="claude-3-5-sonnet-latest"
    )

    args = parser.parse_args()

    # Validate input folder
    if not os.path.exists(args.transcript_folder):
        print(f"❌ Error: Transcript folder '{args.transcript_folder}' does not exist")
        return

    if not os.path.isdir(args.transcript_folder):
        print(f"❌ Error: '{args.transcript_folder}' is not a directory")
        return

    print(
        f"🚀 Starting batch processing at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"📁 Input folder: {args.transcript_folder}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📄 Excel file: {args.excel_file}")

    # Start batch processing
    batch_process_transcripts(
        transcript_folder=args.transcript_folder,
        output_dir=args.output_dir,
        excel_file=args.excel_file,
    )


if __name__ == "__main__":
    main()
