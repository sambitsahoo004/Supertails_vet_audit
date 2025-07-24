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
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def load_config():
    """
    Returns configuration values from config.py or default values if not found
    """
    # Debug: Check current working directory and Python path
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Files in current directory: {os.listdir('.')}")

    try:
        import config
    except ImportError:
        print("✗ config.py not found, using default values")
        print("Available files:", [f for f in os.listdir(".") if f.endswith(".py")])

        class config:
            GCS_BUCKET_NAME = ""
            GCS_PROJECT_NAME = ""
            AWS_S3_BUCKET_NAME = ""
            AWS_S3_CLIENT_STORE_DIRECTORY = ""

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
        print(f"Received event: {json.dumps(event, indent=2)}")
        print(f"Event type: {type(event)}")
        print(f"Operation requested: {event.get('operation', 'NOT SPECIFIED')}")

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
        print(f"Routing to operation: {operation}")

        if operation == "list":
            print("Executing list_contents function")
            return list_contents(gcs_bucket, event)
        elif operation == "download_file":
            print("Executing download_single_file function")
            return download_single_file(
                gcs_bucket,
                event,
                aws_s3_client,
                aws_s3_bucket,
                aws_s3_client_store_directory,
            )
        elif operation == "download_directory":
            print("Executing download_directory function")
            return download_directory(
                gcs_bucket,
                event,
                aws_s3_client,
                aws_s3_bucket,
                aws_s3_client_store_directory,
            )
        elif operation == "upload_file":
            print("Executing upload_single_file function")
            return upload_single_file(
                gcs_bucket,
                event,
                aws_s3_client,
                aws_s3_bucket,
                aws_s3_client_store_directory,
            )
        elif operation == "upload_directory":
            print("Executing upload_directory function")
            return upload_directory(
                gcs_bucket,
                event,
                aws_s3_client,
                aws_s3_bucket,
                aws_s3_client_store_directory,
            )
        elif operation == "delete_file":
            print("delete a single file in gcs bucket")
            return delete_single_file(gcs_bucket, event)
        elif operation == "delete_directory":
            print("delete a directory in gcs bucket")
            return delete_directory(gcs_bucket, event)
        elif operation == "rename_file":
            print("Rename source_file to destination_file in gcs bucket")
            return rename_single_file(gcs_bucket, event)
        elif operation == "rename_directory":
            print("Rename source_directory to destination_directory in gcs bucket")
            return rename_directory(gcs_bucket, event)
        else:
            print(f"Invalid operation received: {operation}")
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


def download_single_file(
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


def download_directory(bucket, event, s3_client, aws_s3_bucket_name, client_store_name):
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


def upload_single_file(bucket, event, s3_client, aws_s3_bucket_name, client_store_name):
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


def upload_directory(bucket, event, s3_client, aws_s3_bucket_name, client_store_name):
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


def delete_single_file(bucket, event):
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


def delete_directory(bucket, event):
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


def rename_single_file(bucket, event):
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


def rename_directory(bucket, event):
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


# Test function for local development
if __name__ == "__main__":
    # Test event
    test_event = {"operation": "list"}

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))

    print("Test completed. Check the output above for results.")

# Note: The above code is designed to run in an AWS Lambda environment.
# For local testing, ensure you have the necessary environment variables set
