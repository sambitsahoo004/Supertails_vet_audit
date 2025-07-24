import json
import os
import tempfile
import base64
import zipfile
from io import BytesIO
from google.cloud import storage
from google.oauth2 import service_account
import boto3
import config
import sys
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # You can set to DEBUG for more verbose logs

# Load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def load_config():
    """
    Returns configuration values from config.py or default values if not found
    """
    # Debug: Check current working directory and Python path
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Files in current directory: {os.listdir('.')}")

    try:
        import config
    except ImportError:
        logger.info("✗ config.py not found, using default values")
        logger.info(
            "Available files:", [f for f in os.listdir(".") if f.endswith(".py")]
        )

        class config:
            GCS_BUCKET_NAME = ""
            GCS_PROJECT_NAME = ""
            AWS_S3_BUCKET_NAME = "supertails-lambda-output-bucket"
            AWS_S3_CLIENT_STORE_DIRECTORY = "supertails-vet-audit"

    return {
        "GCS_BUCKET_NAME": getattr(config, "GCS_BUCKET_NAME", ""),
        "GCS_PROJECT_NAME": getattr(config, "GCS_PROJECT_NAME", ""),
        "AWS_S3_BUCKET_NAME": getattr(config, "AWS_S3_BUCKET_NAME", ""),
        "AWS_S3_CLIENT_STORE_DIRECTORY": getattr(
            config, "AWS_S3_CLIENT_STORE_DIRECTORY", ""
        ),
    }


def lambda_handler(event, context):
    """
    AWS Lambda function to interact with Google Cloud Storage
    Supports both single file and directory operations with S3 integration
    """

    try:
        # Add debugging to see what event is received
        logger.info(f"Received event: {json.dumps(event, indent=2)}")
        logger.info(f"Event type: {type(event)}")
        logger.info(f"Operation requested: {event.get('operation', 'NOT SPECIFIED')}")

        # Get environment variables
        gcp_service_account_key = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
        if not gcp_service_account_key:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    "GCP_SERVICE_ACCOUNT_KEY environment variable not set"
                ),
            }

        config_values = load_config()

        gcs_bucket_name = config_values["GCS_BUCKET_NAME"]
        if not gcs_bucket_name:
            return {
                "statusCode": 400,
                "body": json.dumps("GCS_BUCKET_NAME not set in config.py"),
            }

        gcs_project_name = config_values["GCS_PROJECT_NAME"]
        if not gcs_project_name:
            return {
                "statusCode": 400,
                "body": json.dumps("GCS_PROJECT_NAME not set in config.py"),
            }

        aws_s3_bucket = config_values["AWS_S3_BUCKET_NAME"]
        if not aws_s3_bucket:
            return {
                "statusCode": 400,
                "body": json.dumps("AWS_S3_BUCKET_NAME not set in config.py"),
            }

        aws_s3_client_store_directory = config_values["AWS_S3_CLIENT_STORE_DIRECTORY"]
        if not aws_s3_client_store_directory:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    "AWS_S3_CLIENT_STORE_DIRECTORY not set in config.py"
                ),
            }

        # Decode the base64 encoded service account key
        try:
            gcp_service_account_decoded_key = base64.b64decode(
                gcp_service_account_key
            ).decode("utf-8")
            service_account_info = json.loads(gcp_service_account_decoded_key)
        except Exception as e:
            return {
                "statusCode": 400,
                "body": json.dumps(f"Error decoding service account key: {str(e)}"),
            }

        # Create credentials from service account info
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        # Initialize the GCS client
        client = storage.Client(credentials=credentials, project=gcs_project_name)
        gcs_bucket = client.bucket(gcs_bucket_name)

        # Initialize S3 client
        aws_s3_client = boto3.client("s3")

        # Route to appropriate operation
        operation = event.get("operation", "list")
        logger.info(f"Routing to operation: {operation}")

        if operation == "list":
            logger.info("Executing list_contents function")
            return list_contents(gcs_bucket, event)
        elif operation == "download_single_file_GCS_to_S3":
            logger.info("Executing download_single_file function from GCS to S3")
            return download_single_file_GCS_to_S3(
                gcs_bucket,
                event,
                aws_s3_client,
                aws_s3_bucket,
                aws_s3_client_store_directory,
            )
        elif operation == "download_directory_GCS_to_S3":
            logger.info("Executing download_directory function from GCS to S3")
            return download_directory_GCS_to_S3(
                gcs_bucket,
                event,
                aws_s3_client,
                aws_s3_bucket,
                aws_s3_client_store_directory,
            )
        elif operation == "upload_single_file_S3_to_GCS":
            logger.info("Executing upload_single_file function from S3 to GCS")
            return upload_single_file_S3_to_GCS(
                gcs_bucket,
                event,
                aws_s3_client,
                aws_s3_bucket,
                aws_s3_client_store_directory,
            )
        elif operation == "upload_directory_S3_to_GCS":
            logger.info("Executing upload_directory function from S3 to GCS")
            return upload_directory_S3_to_GCS(
                gcs_bucket,
                event,
                aws_s3_client,
                aws_s3_bucket,
                aws_s3_client_store_directory,
            )
        elif operation == "delete_single_file_in_gcs":
            logger.info("delete a single file in gcs bucket")
            return delete_single_file_in_gcs(gcs_bucket, event)
        elif operation == "delete_directory_in_gcs":
            logger.info("delete a directory in gcs bucket")
            return delete_directory_in_gcs(gcs_bucket, event)
        elif operation == "move_single_file_in_gcs":
            logger.info("Move or rename source_file to destination_file in gcs bucket")
            return move_single_file_in_gcs(gcs_bucket, event)
        elif operation == "move_directory_in_gcs":
            logger.info(
                "Rename source_directory to destination_directory in gcs bucket"
            )
            return move_directory_in_gcs(gcs_bucket, event)
        elif operation == "delete_single_file_in_s3":
            logger.info("Delete a single file in S3 bucket")
            return delete_single_file_in_s3(aws_s3_client, event, aws_s3_bucket)
        elif operation == "delete_directory_in_s3":
            logger.info("Delete a directory in S3 bucket")
            return delete_directory_in_s3(aws_s3_client, event, aws_s3_bucket)
        elif operation == "move_single_file_in_s3":
            logger.info("Move or rename a single file in S3 bucket")
            return move_single_file_in_s3(aws_s3_client, event, aws_s3_bucket)
        elif operation == "move_directory_in_s3":
            logger.info("Move or rename a directory in S3 bucket")
            return move_directory_in_s3(aws_s3_client, event, aws_s3_bucket)
        elif operation == "copy_single_file_in_s3":
            logger.info("Copy a single file in S3 bucket")
            return copy_single_file_in_s3(aws_s3_client, event, aws_s3_bucket)
        elif operation == "copy_directory_in_s3":
            logger.info("Copy a directory in S3 bucket")
            return copy_directory_in_s3(aws_s3_client, event, aws_s3_bucket)
        else:
            logger.info(f"Invalid operation received: {operation}")
            return {
                "statusCode": 400,
                "body": json.dumps(
                    "Invalid operation. Supported operations: list, download_file, download_directory, "
                    "upload_file, upload_directory, delete_file, delete_directory, rename_file, rename_directory"
                ),
            }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps(f"Error: {str(e)}")}


def list_contents(bucket, event):
    """List all contents or contents of a specific directory"""
    try:
        prefix = event.get("prefix", "")  # Directory path to list
        delimiter = event.get("delimiter", "/")  # Use "/" for directory-like listing

        if prefix and not prefix.endswith("/"):
            prefix += "/"

        blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

        files = []
        directories = []

        for blob in blobs:
            files.append(
                {
                    "name": blob.name,
                    "size": blob.size,
                    "created": (
                        blob.time_created.isoformat() if blob.time_created else None
                    ),
                    "updated": blob.updated.isoformat() if blob.updated else None,
                    "content_type": blob.content_type,
                }
            )

        # Get directory prefixes (subdirectories)
        for prefix in blobs.prefixes:
            directories.append(prefix.rstrip("/"))

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Successfully listed contents of bucket: {bucket.name}",
                    "prefix": prefix,
                    "files": files,
                    "directories": directories,
                    "file_count": len(files),
                    "directory_count": len(directories),
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error listing contents: {str(e)}"),
        }


def download_single_file_GCS_to_S3(
    bucket, event, s3_client, aws_s3_bucket_name, client_store_name
):
    """Download a single file as ZIP and save to S3, then unzip in S3"""
    try:
        file_name = event.get("file_name")
        if not file_name:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    "file_name parameter required for download operation"
                ),
            }

        blob = bucket.blob(file_name)
        if not blob.exists():
            return {
                "statusCode": 404,
                "body": json.dumps(f"File {file_name} not found in bucket"),
            }

        # Create ZIP file with single file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            file_content = blob.download_as_bytes()
            # Use just the filename without path for the archive
            archive_name = os.path.basename(file_name)
            zip_file.writestr(archive_name, file_content)

        zip_buffer.seek(0)
        zip_content = zip_buffer.getvalue()

        # Generate ZIP filename (use original filename without extension + .zip)
        file_basename = os.path.splitext(os.path.basename(file_name))[0]
        zip_filename = f"{file_basename}.zip"
        s3_zip_key = f"{client_store_name}/{zip_filename}"

        # Upload ZIP to S3
        s3_client.put_object(
            Bucket=aws_s3_bucket_name,
            Key=s3_zip_key,
            Body=zip_content,
            ContentType="application/zip",
        )

        # Unzip and save individual file to S3
        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            for file_info in zip_file.filelist:
                if not file_info.is_dir():
                    file_content = zip_file.read(file_info.filename)
                    s3_file_key = f"{client_store_name}/{file_info.filename}"
                    s3_client.put_object(
                        Bucket=aws_s3_bucket_name,
                        Key=s3_file_key,
                        Body=file_content,
                    )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Successfully downloaded {file_name} as ZIP and unzipped to S3",
                    "file_name": file_name,
                    "zip_location": f"s3://{aws_s3_bucket_name}/{s3_zip_key}",
                    "unzipped_location": f"s3://{aws_s3_bucket_name}/{client_store_name}/",
                    "size": blob.size,
                    "zip_size": len(zip_content),
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error downloading file: {str(e)}"),
        }


def download_directory_GCS_to_S3(
    bucket, event, s3_client, aws_s3_bucket_name, client_store_name
):
    """Download all files in a directory as a ZIP file and save to S3, then unzip"""
    try:
        directory_name = event.get("directory_name", "")

        if directory_name and not directory_name.endswith("/"):
            directory_name += "/"

        # List all blobs with the directory prefix
        blobs = list(bucket.list_blobs(prefix=directory_name))
        if not blobs:
            return {
                "statusCode": 404,
                "body": json.dumps(f"Directory {directory_name} not found or is empty"),
            }

        # Create a ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for blob in blobs:
                if not blob.name.endswith("/"):  # Skip directory markers
                    file_content = blob.download_as_bytes()
                    # Remove directory prefix from the file name in the zip
                    archive_name = (
                        blob.name[len(directory_name) :]
                        if directory_name
                        else blob.name
                    )
                    zip_file.writestr(archive_name, file_content)

        zip_buffer.seek(0)
        zip_content = zip_buffer.getvalue()

        # Generate ZIP filename (use directory name)
        dir_basename = (
            directory_name.rstrip("/").split("/")[-1]
            if directory_name.strip("/")
            else "root"
        )
        zip_filename = f"{dir_basename}.zip"
        s3_zip_key = f"{client_store_name}/{zip_filename}"

        # Upload ZIP to S3
        s3_client.put_object(
            Bucket=aws_s3_bucket_name,
            Key=s3_zip_key,
            Body=zip_content,
            ContentType="application/zip",
        )

        # Unzip and save individual files to S3 under directory structure
        zip_buffer.seek(0)
        unzipped_files = []
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            for file_info in zip_file.filelist:
                if not file_info.is_dir():
                    file_content = zip_file.read(file_info.filename)
                    # Place unzipped files under CLIENT_STORE_NAME/directory_name/
                    s3_file_key = (
                        f"{client_store_name}/{dir_basename}/{file_info.filename}"
                    )
                    s3_client.put_object(
                        Bucket=aws_s3_bucket_name,
                        Key=s3_file_key,
                        Body=file_content,
                    )
                    unzipped_files.append(s3_file_key)

        response_data = {
            "message": f"Successfully downloaded directory {directory_name} as ZIP and unzipped to S3",
            "directory_name": directory_name,
            "zip_location": f"s3://{aws_s3_bucket_name}/{s3_zip_key}",
            "unzipped_location": f"s3://{aws_s3_bucket_name}/{client_store_name}/",
            "file_count": len([b for b in blobs if not b.name.endswith("/")]),
            "zip_size": len(zip_content),
            "unzipped_files": unzipped_files,
            "unzipped_count": len(unzipped_files),
        }

        return {"statusCode": 200, "body": json.dumps(response_data)}
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error downloading directory: {str(e)}"),
        }


def upload_single_file_S3_to_GCS(
    bucket, event, s3_client, aws_s3_bucket_name, client_store_name
):
    """Upload a single file from S3 to GCS"""
    try:
        file_name = event.get("file_name")
        s3_file_key = event.get(
            "s3_file_key"
        )  # Optional: specify S3 key, otherwise use CLIENT_STORE_NAME/file_name

        if not file_name:
            return {
                "statusCode": 400,
                "body": json.dumps("file_name parameter required for upload operation"),
            }

        # Determine S3 key
        if not s3_file_key:
            s3_file_key = f"{client_store_name}/{file_name}"

        # Download file from S3
        try:
            s3_response = s3_client.get_object(
                Bucket=aws_s3_bucket_name, Key=s3_file_key
            )
            file_content = s3_response["Body"].read()
        except Exception as e:
            return {
                "statusCode": 404,
                "body": json.dumps(
                    f"File {s3_file_key} not found in S3 bucket: {str(e)}"
                ),
            }

        # Upload to GCS
        blob = bucket.blob(file_name)
        blob.upload_from_string(file_content)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Successfully uploaded {file_name} from S3 to GCS",
                    "file_name": file_name,
                    "s3_source": f"s3://{aws_s3_bucket_name}/{s3_file_key}",
                    "gcs_destination": f"gs://{bucket.name}/{file_name}",
                    "size": blob.size,
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error uploading file: {str(e)}"),
        }


def upload_directory_S3_to_GCS(
    bucket, event, s3_client, aws_s3_bucket_name, client_store_name
):
    """Upload multiple files from S3 directory to GCS"""
    try:
        directory_name = event.get("directory_name", "")
        s3_prefix = event.get(
            "s3_prefix"
        )  # Optional: specify S3 prefix, otherwise use CLIENT_STORE_NAME/directory_name

        # Determine S3 prefix
        if not s3_prefix:
            if directory_name:
                # Remove trailing slash from directory_name for S3 prefix construction
                clean_directory_name = directory_name.rstrip("/")
                s3_prefix = f"{client_store_name}/{clean_directory_name}"
            else:
                s3_prefix = client_store_name

        if not s3_prefix.endswith("/"):
            s3_prefix += "/"

        # List all objects in S3 with the prefix
        try:
            response = s3_client.list_objects_v2(
                Bucket=aws_s3_bucket_name, Prefix=s3_prefix
            )
            if "Contents" not in response:
                return {
                    "statusCode": 404,
                    "body": json.dumps(f"No files found in S3 with prefix {s3_prefix}"),
                }
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps(f"Error listing S3 objects: {str(e)}"),
            }

        if directory_name and not directory_name.endswith("/"):
            directory_name += "/"

        uploaded_files = []
        errors = []

        for obj in response["Contents"]:
            s3_key = obj["Key"]

            # Skip directory markers
            if s3_key.endswith("/"):
                continue

            try:
                # Get file content from S3
                s3_response = s3_client.get_object(
                    Bucket=aws_s3_bucket_name, Key=s3_key
                )
                file_content = s3_response["Body"].read()

                # Calculate GCS path
                relative_path = s3_key[len(s3_prefix) :]  # Remove S3 prefix
                gcs_path = directory_name + relative_path

                # Upload to GCS
                blob = bucket.blob(gcs_path)
                blob.upload_from_string(file_content)

                uploaded_files.append(
                    {
                        "s3_source": f"s3://{aws_s3_bucket_name}/{s3_key}",
                        "gcs_destination": f"gs://{bucket.name}/{gcs_path}",
                        "size": blob.size,
                    }
                )

            except Exception as e:
                errors.append(f"Error uploading {s3_key}: {str(e)}")

        return {
            "statusCode": (
                200 if not errors else 207
            ),  # 207 Multi-Status if there were some errors
            "body": json.dumps(
                {
                    "message": f"Upload operation completed from S3 prefix {s3_prefix} to GCS directory {directory_name}",
                    "s3_prefix": s3_prefix,
                    "gcs_directory": directory_name,
                    "uploaded_files": uploaded_files,
                    "uploaded_count": len(uploaded_files),
                    "errors": errors,
                    "error_count": len(errors),
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error uploading directory: {str(e)}"),
        }


def delete_single_file_in_gcs(bucket, event):
    """Delete a single file"""
    try:
        file_name = event.get("file_name")
        if not file_name:
            return {
                "statusCode": 400,
                "body": json.dumps("file_name parameter required for delete operation"),
            }

        blob = bucket.blob(file_name)
        if not blob.exists():
            return {
                "statusCode": 404,
                "body": json.dumps(f"File {file_name} not found in bucket"),
            }

        blob.delete()

        return {
            "statusCode": 200,
            "body": json.dumps(
                {"message": f"Successfully deleted {file_name} from {bucket.name}"}
            ),
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps(f"Error deleting file: {str(e)}")}


def delete_directory_in_gcs(bucket, event):
    """Delete all files in a directory"""
    try:
        directory_name = event.get("directory_name", "")
        if directory_name and not directory_name.endswith("/"):
            directory_name += "/"

        # List all blobs with the directory prefix
        blobs = list(bucket.list_blobs(prefix=directory_name))
        if not blobs:
            return {
                "statusCode": 404,
                "body": json.dumps(f"Directory {directory_name} not found or is empty"),
            }

        deleted_files = []
        errors = []

        for blob in blobs:
            try:
                blob.delete()
                deleted_files.append(blob.name)
            except Exception as e:
                errors.append(f"Error deleting {blob.name}: {str(e)}")

        return {
            "statusCode": 200 if not errors else 207,
            "body": json.dumps(
                {
                    "message": f"Delete operation completed for directory {directory_name}",
                    "directory_name": directory_name,
                    "deleted_files": deleted_files,
                    "deleted_count": len(deleted_files),
                    "errors": errors,
                    "error_count": len(errors),
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error deleting directory: {str(e)}"),
        }


def move_single_file_in_gcs(bucket, event):
    """Move/rename a single file"""
    try:
        source_file = event.get("source_file")
        destination_file = event.get("destination_file")

        if not source_file or not destination_file:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    "Both source_file and destination_file parameters required"
                ),
            }

        source_blob = bucket.blob(source_file)
        if not source_blob.exists():
            return {
                "statusCode": 404,
                "body": json.dumps(f"Source file {source_file} not found in bucket"),
            }

        # Copy to new location
        destination_blob = bucket.copy_blob(source_blob, bucket, destination_file)

        # Delete original file
        source_blob.delete()

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Successfully moved {source_file} to {destination_file}",
                    "source_file": source_file,
                    "destination_file": destination_file,
                    "size": destination_blob.size,
                }
            ),
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps(f"Error moving file: {str(e)}")}


def move_directory_in_gcs(bucket, event):
    """Move/rename a directory and all its contents"""
    try:
        source_directory = event.get("source_directory", "")
        destination_directory = event.get("destination_directory", "")

        if not source_directory or not destination_directory:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    "Both source_directory and destination_directory parameters required"
                ),
            }

        if not source_directory.endswith("/"):
            source_directory += "/"
        if not destination_directory.endswith("/"):
            destination_directory += "/"

        # List all blobs with the source directory prefix
        source_blobs = list(bucket.list_blobs(prefix=source_directory))
        if not source_blobs:
            return {
                "statusCode": 404,
                "body": json.dumps(
                    f"Source directory {source_directory} not found or is empty"
                ),
            }

        moved_files = []
        errors = []

        for source_blob in source_blobs:
            try:
                # Calculate new destination path
                relative_path = source_blob.name[len(source_directory) :]
                destination_path = destination_directory + relative_path

                # Copy to new location
                destination_blob = bucket.copy_blob(
                    source_blob, bucket, destination_path
                )

                # Delete original
                source_blob.delete()

                moved_files.append(
                    {
                        "source": source_blob.name,
                        "destination": destination_path,
                        "size": destination_blob.size,
                    }
                )

            except Exception as e:
                errors.append(f"Error moving {source_blob.name}: {str(e)}")

        return {
            "statusCode": 200 if not errors else 207,
            "body": json.dumps(
                {
                    "message": f"Move operation completed from {source_directory} to {destination_directory}",
                    "source_directory": source_directory,
                    "destination_directory": destination_directory,
                    "moved_files": moved_files,
                    "moved_count": len(moved_files),
                    "errors": errors,
                    "error_count": len(errors),
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error moving directory: {str(e)}"),
        }


def delete_single_file_in_s3(s3_client, event, aws_s3_bucket_name):
    """Delete a single file in S3 bucket"""
    try:
        file_name = event.get("file_name")
        if not file_name:
            return {
                "statusCode": 400,
                "body": json.dumps("file_name parameter required for delete operation"),
            }

        # Check if file exists
        try:
            s3_client.head_object(Bucket=aws_s3_bucket_name, Key=file_name)
        except s3_client.exceptions.NoSuchKey:
            return {
                "statusCode": 404,
                "body": json.dumps(f"File {file_name} not found in S3 bucket"),
            }

        # Delete the file
        s3_client.delete_object(Bucket=aws_s3_bucket_name, Key=file_name)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Successfully deleted {file_name} from S3 bucket {aws_s3_bucket_name}"
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error deleting file from S3: {str(e)}"),
        }


def delete_directory_in_s3(s3_client, event, aws_s3_bucket_name):
    """Delete all files in a directory in S3 bucket"""
    try:
        directory_name = event.get("directory_name", "")
        if directory_name and not directory_name.endswith("/"):
            directory_name += "/"

        # List all objects with the directory prefix
        try:
            response = s3_client.list_objects_v2(
                Bucket=aws_s3_bucket_name, Prefix=directory_name
            )
            if "Contents" not in response:
                return {
                    "statusCode": 404,
                    "body": json.dumps(
                        f"Directory {directory_name} not found or is empty"
                    ),
                }
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps(f"Error listing S3 objects: {str(e)}"),
            }

        deleted_files = []
        errors = []

        # Delete objects in batches (S3 allows up to 1000 objects per batch)
        objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]

        # Process in chunks of 1000
        for i in range(0, len(objects_to_delete), 1000):
            batch = objects_to_delete[i : i + 1000]
            try:
                delete_response = s3_client.delete_objects(
                    Bucket=aws_s3_bucket_name, Delete={"Objects": batch}
                )

                # Track successfully deleted files
                if "Deleted" in delete_response:
                    for deleted_obj in delete_response["Deleted"]:
                        deleted_files.append(deleted_obj["Key"])

                # Track errors
                if "Errors" in delete_response:
                    for error_obj in delete_response["Errors"]:
                        errors.append(
                            f"Error deleting {error_obj['Key']}: {error_obj['Message']}"
                        )

            except Exception as e:
                errors.append(f"Error deleting batch: {str(e)}")

        return {
            "statusCode": 200 if not errors else 207,
            "body": json.dumps(
                {
                    "message": f"Delete operation completed for directory {directory_name}",
                    "directory_name": directory_name,
                    "deleted_files": deleted_files,
                    "deleted_count": len(deleted_files),
                    "errors": errors,
                    "error_count": len(errors),
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error deleting directory from S3: {str(e)}"),
        }


def move_single_file_in_s3(s3_client, event, aws_s3_bucket_name):
    """Move a single file in S3 bucket (copy + delete)"""
    try:
        source_file = event.get("source_file_name")
        destination_file = event.get("destination_file_name")

        if not source_file or not destination_file:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    "Both source_file_name and destination_file_name parameters required"
                ),
            }

        # Check if source file exists
        try:
            s3_client.head_object(Bucket=aws_s3_bucket_name, Key=source_file)
        except s3_client.exceptions.NoSuchKey:
            return {
                "statusCode": 404,
                "body": json.dumps(f"Source file {source_file} not found in S3 bucket"),
            }

        # Copy to new location
        copy_source = {"Bucket": aws_s3_bucket_name, "Key": source_file}
        s3_client.copy_object(
            CopySource=copy_source, Bucket=aws_s3_bucket_name, Key=destination_file
        )

        # Delete original file
        s3_client.delete_object(Bucket=aws_s3_bucket_name, Key=source_file)

        # Get file size for response
        response = s3_client.head_object(
            Bucket=aws_s3_bucket_name, Key=destination_file
        )
        file_size = response["ContentLength"]

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Successfully moved {source_file} to {destination_file}",
                    "source_file": source_file,
                    "destination_file": destination_file,
                    "size": file_size,
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error moving file in S3: {str(e)}"),
        }


def move_directory_in_s3(s3_client, event, aws_s3_bucket_name):
    """Move a directory in S3 bucket (copy all files + delete originals)"""
    try:
        source_directory = event.get("source_directory_name", "")
        destination_directory = event.get("destination_directory_name", "")

        if not source_directory or not destination_directory:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    "Both source_directory_name and destination_directory_name parameters required"
                ),
            }

        if not source_directory.endswith("/"):
            source_directory += "/"
        if not destination_directory.endswith("/"):
            destination_directory += "/"

        # List all objects with the source directory prefix
        try:
            response = s3_client.list_objects_v2(
                Bucket=aws_s3_bucket_name, Prefix=source_directory
            )
            if "Contents" not in response:
                return {
                    "statusCode": 404,
                    "body": json.dumps(
                        f"Source directory {source_directory} not found or is empty"
                    ),
                }
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps(f"Error listing S3 objects: {str(e)}"),
            }

        moved_files = []
        errors = []

        for obj in response["Contents"]:
            source_key = obj["Key"]
            try:
                # Skip directory markers
                if source_key.endswith("/"):
                    continue

                # Calculate new destination path
                relative_path = source_key[len(source_directory) :]
                destination_key = destination_directory + relative_path

                # Copy to new location
                copy_source = {"Bucket": aws_s3_bucket_name, "Key": source_key}
                s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=aws_s3_bucket_name,
                    Key=destination_key,
                )

                # Delete original
                s3_client.delete_object(Bucket=aws_s3_bucket_name, Key=source_key)

                moved_files.append(
                    {
                        "source": source_key,
                        "destination": destination_key,
                        "size": obj["Size"],
                    }
                )

            except Exception as e:
                errors.append(f"Error moving {source_key}: {str(e)}")

        return {
            "statusCode": 200 if not errors else 207,
            "body": json.dumps(
                {
                    "message": f"Move operation completed from {source_directory} to {destination_directory}",
                    "source_directory": source_directory,
                    "destination_directory": destination_directory,
                    "moved_files": moved_files,
                    "moved_count": len(moved_files),
                    "errors": errors,
                    "error_count": len(errors),
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error moving directory in S3: {str(e)}"),
        }


def copy_single_file_in_s3(s3_client, event, aws_s3_bucket_name):
    """Copy a single file in S3 bucket"""
    try:
        source_file = event.get("source_file_name")
        destination_file = event.get("destination_file_name")

        if not source_file or not destination_file:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    "Both source_file_name and destination_file_name parameters required"
                ),
            }

        # Check if source file exists
        try:
            s3_client.head_object(Bucket=aws_s3_bucket_name, Key=source_file)
        except s3_client.exceptions.NoSuchKey:
            return {
                "statusCode": 404,
                "body": json.dumps(f"Source file {source_file} not found in S3 bucket"),
            }

        # Copy to new location
        copy_source = {"Bucket": aws_s3_bucket_name, "Key": source_file}
        s3_client.copy_object(
            CopySource=copy_source, Bucket=aws_s3_bucket_name, Key=destination_file
        )

        # Get file size for response
        response = s3_client.head_object(
            Bucket=aws_s3_bucket_name, Key=destination_file
        )
        file_size = response["ContentLength"]

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Successfully copied {source_file} to {destination_file}",
                    "source_file": source_file,
                    "destination_file": destination_file,
                    "size": file_size,
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error copying file in S3: {str(e)}"),
        }


def copy_directory_in_s3(s3_client, event, aws_s3_bucket_name):
    """Copy a directory in S3 bucket (copy all files)"""
    try:
        source_directory = event.get("source_directory_name", "")
        destination_directory = event.get("destination_directory_name", "")

        if not source_directory or not destination_directory:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    "Both source_directory_name and destination_directory_name parameters required"
                ),
            }

        if not source_directory.endswith("/"):
            source_directory += "/"
        if not destination_directory.endswith("/"):
            destination_directory += "/"

        # List all objects with the source directory prefix
        try:
            response = s3_client.list_objects_v2(
                Bucket=aws_s3_bucket_name, Prefix=source_directory
            )
            if "Contents" not in response:
                return {
                    "statusCode": 404,
                    "body": json.dumps(
                        f"Source directory {source_directory} not found or is empty"
                    ),
                }
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps(f"Error listing S3 objects: {str(e)}"),
            }

        copied_files = []
        errors = []

        for obj in response["Contents"]:
            source_key = obj["Key"]
            try:
                # Skip directory markers
                if source_key.endswith("/"):
                    continue

                # Calculate new destination path
                relative_path = source_key[len(source_directory) :]
                destination_key = destination_directory + relative_path

                # Copy to new location
                copy_source = {"Bucket": aws_s3_bucket_name, "Key": source_key}
                s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=aws_s3_bucket_name,
                    Key=destination_key,
                )

                copied_files.append(
                    {
                        "source": source_key,
                        "destination": destination_key,
                        "size": obj["Size"],
                    }
                )

            except Exception as e:
                errors.append(f"Error copying {source_key}: {str(e)}")

        return {
            "statusCode": 200 if not errors else 207,
            "body": json.dumps(
                {
                    "message": f"Copy operation completed from {source_directory} to {destination_directory}",
                    "source_directory": source_directory,
                    "destination_directory": destination_directory,
                    "copied_files": copied_files,
                    "copied_count": len(copied_files),
                    "errors": errors,
                    "error_count": len(errors),
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error copying directory in S3: {str(e)}"),
        }


# Test function for local development
if __name__ == "__main__":
    # Test different operations - uncomment the one you want to test
    # pip install google-cloud-storage boto3 python-dotenv
    # Create the config.py file with your actual bucket names and project ID
    # Ensure your .env file has the GCP_SERVICE_ACCOUNT_KEY (base64 encoded JSON)
    # Set up AWS credentials using AWS CLI: aws configure

    # 1. List contents (basic test)
    test_event_list = {
        "operation": "list",
        "prefix": "",  # Optional: specify directory prefix like "folder1/"
        "delimiter": "/",
    }

    # 2. Download single file from GCS to S3
    test_event_download_file = {
        "operation": "download_single_file_GCS_to_S3",
        "file_name": "07-May-2025/test2.txt",
    }

    # 3. Rename single file in S3
    test_event_move_file_s3 = {
        "operation": "move_single_file_in_s3",
        "source_file_name": "supertails-vet-audit/07-May-2025/sample4.txt",
        "destination_file_name": "supertails-vet-audit/08-May-2025/test1.txt",
    }

    # Choose which test to run (start with 'list' for basic connectivity test)
    test_event = test_event_move_file_s3

    print("Testing Lambda function locally...")
    print(f"Test event: {json.dumps(test_event, indent=2)}")
    print("-" * 50)

    try:
        result = lambda_handler(test_event, None)
        print("Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback

        traceback.print_exc()

    print("-" * 50)
    print("Test completed.")

# Note: The above code is designed to run in an AWS Lambda environment.
# For local testing, ensure you have the necessary environment variables set and config.py
