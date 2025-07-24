#!/usr/bin/env python3

import os
import boto3
import logging
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError, NoCredentialsError


class S3Handler:
    """
    A dedicated handler for AWS S3 operations.
    Handles file upload, download, listing, and management operations.
    """

    def __init__(self, region_name: Optional[str] = None):
        """
        Initialize S3 handler with boto3 client.

        Args:
            region_name: AWS region name (optional, uses default if not specified)
        """
        try:
            self.s3_client = boto3.client("s3", region_name=region_name)
            self.logger = logging.getLogger(__name__)
            self.logger.info("✅ S3 handler initialized successfully")
        except NoCredentialsError:
            self.logger.error(
                "❌ AWS credentials not found. Please configure AWS credentials."
            )
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize S3 handler: {str(e)}")
            raise

    def download_file(self, bucket_name: str, key: str, local_path: str) -> bool:
        """
        Download a file from S3 to local path.

        Args:
            bucket_name: S3 bucket name
            key: S3 object key (file path in bucket)
            local_path: Local file path to save the downloaded file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            self.s3_client.download_file(bucket_name, key, local_path)
            self.logger.info(f"✅ Downloaded s3://{bucket_name}/{key} to {local_path}")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                self.logger.error(f"❌ File not found: s3://{bucket_name}/{key}")
            elif error_code == "NoSuchBucket":
                self.logger.error(f"❌ Bucket not found: {bucket_name}")
            else:
                self.logger.error(
                    f"❌ Error downloading s3://{bucket_name}/{key}: {str(e)}"
                )
            return False

        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error downloading s3://{bucket_name}/{key}: {str(e)}"
            )
            return False

    def upload_file(self, local_path: str, bucket_name: str, key: str) -> bool:
        """
        Upload a file from local path to S3.

        Args:
            local_path: Local file path to upload
            bucket_name: S3 bucket name
            key: S3 object key (file path in bucket)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(local_path):
                self.logger.error(f"❌ Local file not found: {local_path}")
                return False

            self.s3_client.upload_file(local_path, bucket_name, key)
            self.logger.info(f"✅ Uploaded {local_path} to s3://{bucket_name}/{key}")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                self.logger.error(f"❌ Bucket not found: {bucket_name}")
            else:
                self.logger.error(
                    f"❌ Error uploading {local_path} to s3://{bucket_name}/{key}: {str(e)}"
                )
            return False

        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error uploading {local_path} to s3://{bucket_name}/{key}: {str(e)}"
            )
            return False

    def list_objects(
        self, bucket_name: str, prefix: str = "", max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List objects in an S3 bucket with optional prefix filter.

        Args:
            bucket_name: S3 bucket name
            prefix: Prefix to filter objects (optional)
            max_keys: Maximum number of keys to return

        Returns:
            List of object dictionaries with keys: Key, Size, LastModified, etc.
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix=prefix, MaxKeys=max_keys
            )

            objects = response.get("Contents", [])
            self.logger.info(
                f"✅ Listed {len(objects)} objects in s3://{bucket_name}/{prefix}"
            )
            return objects

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                self.logger.error(f"❌ Bucket not found: {bucket_name}")
            else:
                self.logger.error(
                    f"❌ Error listing objects in s3://{bucket_name}/{prefix}: {str(e)}"
                )
            return []

        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error listing objects in s3://{bucket_name}/{prefix}: {str(e)}"
            )
            return []

    def get_object_metadata(
        self, bucket_name: str, key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific S3 object.

        Args:
            bucket_name: S3 bucket name
            key: S3 object key

        Returns:
            Dictionary with object metadata or None if not found
        """
        try:
            response = self.s3_client.head_object(Bucket=bucket_name, Key=key)
            self.logger.info(f"✅ Retrieved metadata for s3://{bucket_name}/{key}")
            return response

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                self.logger.warning(f"⚠️ Object not found: s3://{bucket_name}/{key}")
            elif error_code == "NoSuchBucket":
                self.logger.error(f"❌ Bucket not found: {bucket_name}")
            else:
                self.logger.error(
                    f"❌ Error getting metadata for s3://{bucket_name}/{key}: {str(e)}"
                )
            return None

        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error getting metadata for s3://{bucket_name}/{key}: {str(e)}"
            )
            return None

    def delete_object(self, bucket_name: str, key: str) -> bool:
        """
        Delete an object from S3.

        Args:
            bucket_name: S3 bucket name
            key: S3 object key

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=key)
            self.logger.info(f"✅ Deleted s3://{bucket_name}/{key}")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                self.logger.warning(
                    f"⚠️ Object not found for deletion: s3://{bucket_name}/{key}"
                )
                return True  # Object doesn't exist, consider it deleted
            elif error_code == "NoSuchBucket":
                self.logger.error(f"❌ Bucket not found: {bucket_name}")
            else:
                self.logger.error(
                    f"❌ Error deleting s3://{bucket_name}/{key}: {str(e)}"
                )
            return False

        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error deleting s3://{bucket_name}/{key}: {str(e)}"
            )
            return False

    def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        destination_bucket: str,
        destination_key: str,
    ) -> bool:
        """
        Copy an object from one location to another in S3.

        Args:
            source_bucket: Source S3 bucket name
            source_key: Source S3 object key
            destination_bucket: Destination S3 bucket name
            destination_key: Destination S3 object key

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            copy_source = {"Bucket": source_bucket, "Key": source_key}
            self.s3_client.copy_object(
                CopySource=copy_source, Bucket=destination_bucket, Key=destination_key
            )
            self.logger.info(
                f"✅ Copied s3://{source_bucket}/{source_key} to s3://{destination_bucket}/{destination_key}"
            )
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                self.logger.error(
                    f"❌ Source object not found: s3://{source_bucket}/{source_key}"
                )
            elif error_code == "NoSuchBucket":
                self.logger.error(
                    f"❌ Bucket not found: {source_bucket} or {destination_bucket}"
                )
            else:
                self.logger.error(
                    f"❌ Error copying s3://{source_bucket}/{source_key} to s3://{destination_bucket}/{destination_key}: {str(e)}"
                )
            return False

        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error copying s3://{source_bucket}/{source_key} to s3://{destination_bucket}/{destination_key}: {str(e)}"
            )
            return False

    def download_multiple_files(
        self, bucket_name: str, keys: List[str], local_dir: str
    ) -> Dict[str, bool]:
        """
        Download multiple files from S3 to a local directory.

        Args:
            bucket_name: S3 bucket name
            keys: List of S3 object keys to download
            local_dir: Local directory to save downloaded files

        Returns:
            Dictionary mapping file keys to success status
        """
        results = {}

        for key in keys:
            filename = os.path.basename(key)
            local_path = os.path.join(local_dir, filename)
            success = self.download_file(bucket_name, key, local_path)
            results[key] = success

        return results

    def upload_multiple_files(
        self, local_files: List[str], bucket_name: str, prefix: str = ""
    ) -> Dict[str, bool]:
        """
        Upload multiple files from local paths to S3.

        Args:
            local_files: List of local file paths to upload
            bucket_name: S3 bucket name
            prefix: S3 prefix to prepend to file names

        Returns:
            Dictionary mapping local file paths to success status
        """
        results = {}

        for local_file in local_files:
            if not os.path.exists(local_file):
                self.logger.warning(f"⚠️ Local file not found: {local_file}")
                results[local_file] = False
                continue

            filename = os.path.basename(local_file)
            key = f"{prefix}{filename}" if prefix else filename
            success = self.upload_file(local_file, bucket_name, key)
            results[local_file] = success

        return results

    def check_bucket_exists(self, bucket_name: str) -> bool:
        """
        Check if an S3 bucket exists.

        Args:
            bucket_name: S3 bucket name

        Returns:
            bool: True if bucket exists, False otherwise
        """
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.info(f"✅ Bucket exists: {bucket_name}")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                self.logger.warning(f"⚠️ Bucket does not exist: {bucket_name}")
                return False
            else:
                self.logger.error(f"❌ Error checking bucket {bucket_name}: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error checking bucket {bucket_name}: {str(e)}"
            )
            return False

    def get_bucket_size(self, bucket_name: str, prefix: str = "") -> int:
        """
        Get total size of objects in a bucket (with optional prefix).

        Args:
            bucket_name: S3 bucket name
            prefix: Prefix to filter objects (optional)

        Returns:
            int: Total size in bytes
        """
        try:
            total_size = 0
            paginator = self.s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        total_size += obj["Size"]

            self.logger.info(
                f"✅ Bucket size for s3://{bucket_name}/{prefix}: {total_size} bytes"
            )
            return total_size

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                self.logger.error(f"❌ Bucket not found: {bucket_name}")
            else:
                self.logger.error(
                    f"❌ Error getting bucket size for {bucket_name}: {str(e)}"
                )
            return 0

        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error getting bucket size for {bucket_name}: {str(e)}"
            )
            return 0


# Convenience functions for backward compatibility
def download_from_s3(bucket_name: str, key: str, local_path: str) -> bool:
    """
    Convenience function to download a file from S3.

    Args:
        bucket_name: S3 bucket name
        key: S3 object key
        local_path: Local file path

    Returns:
        bool: True if successful, False otherwise
    """
    s3_handler = S3Handler()
    return s3_handler.download_file(bucket_name, key, local_path)


def upload_to_s3(local_path: str, bucket_name: str, key: str) -> bool:
    """
    Convenience function to upload a file to S3.

    Args:
        local_path: Local file path
        bucket_name: S3 bucket name
        key: S3 object key

    Returns:
        bool: True if successful, False otherwise
    """
    s3_handler = S3Handler()
    return s3_handler.upload_file(local_path, bucket_name, key)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)

    # Example usage
    s3_handler = S3Handler()

    # Test bucket existence
    bucket_name = "test-bucket"
    if s3_handler.check_bucket_exists(bucket_name):
        print(f"Bucket {bucket_name} exists")

        # List objects
        objects = s3_handler.list_objects(bucket_name, prefix="test/")
        print(f"Found {len(objects)} objects")

        # Get bucket size
        size = s3_handler.get_bucket_size(bucket_name)
        print(f"Bucket size: {size} bytes")
    else:
        print(f"Bucket {bucket_name} does not exist")
