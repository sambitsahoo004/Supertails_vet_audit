#!/usr/bin/env python3

"""
Simple script to check what files are currently in your S3 bucket.
This helps you verify what files are available for testing your Lambda function.
"""

import boto3
import logging
from config import AWS_S3_BUCKET_NAME, AWS_S3_CLIENT_STORE_DIRECTORY

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_s3_bucket():
    """Check what files are in your S3 bucket."""

    logger.info("🔍 Checking S3 bucket contents...")
    logger.info(f"📦 Bucket: {AWS_S3_BUCKET_NAME}")
    logger.info(f"📁 Directory: {AWS_S3_CLIENT_STORE_DIRECTORY}")

    # Initialize S3 client
    s3_client = boto3.client("s3")

    try:
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=AWS_S3_BUCKET_NAME)
            logger.info(f"✅ Bucket '{AWS_S3_BUCKET_NAME}' exists")
        except Exception as e:
            logger.error(
                f"❌ Bucket '{AWS_S3_BUCKET_NAME}' does not exist or is not accessible"
            )
            logger.error(f"   Error: {str(e)}")
            return

        # List all objects in the bucket
        logger.info("\n📋 All files in bucket:")
        logger.info("=" * 60)

        paginator = s3_client.get_paginator("list_objects_v2")
        total_files = 0

        for page in paginator.paginate(Bucket=AWS_S3_BUCKET_NAME):
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    size = obj["Size"]
                    last_modified = obj["LastModified"]

                    # Highlight transcript files
                    if "transcript" in key.lower() or key.endswith((".txt", ".md")):
                        logger.info(
                            f"🎯 TRANSCRIPT: {key} ({size} bytes, {last_modified})"
                        )
                    else:
                        logger.info(f"   📄 {key} ({size} bytes, {last_modified})")

                    total_files += 1

        if total_files == 0:
            logger.warning("⚠️ No files found in the bucket")
        else:
            logger.info(f"\n📊 Total files found: {total_files}")

        # Check specific transcripts directory
        logger.info(
            f"\n🎯 Checking transcripts directory: {AWS_S3_CLIENT_STORE_DIRECTORY}/transcripts/"
        )
        logger.info("=" * 60)

        transcript_prefix = f"{AWS_S3_CLIENT_STORE_DIRECTORY}/transcripts/"
        response = s3_client.list_objects_v2(
            Bucket=AWS_S3_BUCKET_NAME, Prefix=transcript_prefix
        )

        if "Contents" in response:
            transcript_files = response["Contents"]
            logger.info(f"✅ Found {len(transcript_files)} transcript files:")

            for obj in transcript_files:
                key = obj["Key"]
                size = obj["Size"]
                last_modified = obj["LastModified"]
                filename = key.replace(transcript_prefix, "")

                logger.info(f"   📄 {filename} ({size} bytes, {last_modified})")

                # Check file extension
                if (
                    filename.endswith((".txt", ".md", ".transcript"))
                    or "." not in filename
                ):
                    logger.info(f"      ✅ Supported format")
                else:
                    logger.warning(f"      ⚠️ Unsupported format (will be skipped)")
        else:
            logger.warning(f"⚠️ No files found in transcripts directory")
            logger.info(
                f"💡 Upload transcript files to: s3://{AWS_S3_BUCKET_NAME}/{transcript_prefix}"
            )

        # Check results directory
        logger.info(
            f"\n📊 Checking results directory: {AWS_S3_CLIENT_STORE_DIRECTORY}/results/"
        )
        logger.info("=" * 60)

        results_prefix = f"{AWS_S3_CLIENT_STORE_DIRECTORY}/results/"
        response = s3_client.list_objects_v2(
            Bucket=AWS_S3_BUCKET_NAME, Prefix=results_prefix
        )

        if "Contents" in response:
            result_files = response["Contents"]
            logger.info(f"✅ Found {len(result_files)} result files:")

            for obj in result_files:
                key = obj["Key"]
                size = obj["Size"]
                last_modified = obj["LastModified"]
                filename = key.replace(results_prefix, "")

                if filename.endswith(".xlsx"):
                    logger.info(
                        f"   📊 Excel: {filename} ({size} bytes, {last_modified})"
                    )
                elif filename.endswith(".json"):
                    logger.info(
                        f"   📄 JSON: {filename} ({size} bytes, {last_modified})"
                    )
                else:
                    logger.info(f"   📄 {filename} ({size} bytes, {last_modified})")
        else:
            logger.info(
                f"ℹ️ No result files found yet (this is normal before first test)"
            )

        logger.info("\n" + "=" * 60)
        logger.info("🎉 S3 bucket check completed!")

        # Summary
        logger.info("\n📋 Summary:")
        logger.info(f"   Bucket: {AWS_S3_BUCKET_NAME}")
        logger.info(f"   Total files: {total_files}")

        # Check for transcript files
        transcript_response = s3_client.list_objects_v2(
            Bucket=AWS_S3_BUCKET_NAME,
            Prefix=f"{AWS_S3_CLIENT_STORE_DIRECTORY}/transcripts/",
        )
        transcript_count = len(transcript_response.get("Contents", []))
        logger.info(f"   Transcript files: {transcript_count}")

        if transcript_count > 0:
            logger.info("   ✅ Ready for Lambda testing!")
        else:
            logger.warning("   ⚠️ No transcript files found - upload some files first")

    except Exception as e:
        logger.error(f"❌ Error checking S3 bucket: {str(e)}")


if __name__ == "__main__":
    check_s3_bucket()
