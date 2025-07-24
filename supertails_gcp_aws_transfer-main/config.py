# config.py
import os
from dotenv import load_dotenv

load_dotenv()
# GCP_SERVICE_ACCOUNT_KEY - set via environment variable or directly here
GCP_SERVICE_ACCOUNT_KEY = os.environ.get(
    "GCP_SERVICE_ACCOUNT_KEY", ""
)  # Set your API key here if not using env variable

# Cloud storage configuration

# Google Cloud Storage (GCS) configuration - local setup
# Uncomment the following lines to use local GCS configuration
# GCS_BUCKET_NAME = "reetesh-bucket-2025-v1"
# GCS_PROJECT_NAME = "my-python-project"

# Google Cloud Storage (GCS) configuration - supertails setup
# Uncomment the following lines to use supertails GCS configuration
GCS_BUCKET_NAME = "cx-call-recordings"
GCS_PROJECT_NAME = "GA4 Data API"

# AWS S3 configuration
AWS_S3_BUCKET_NAME = "supertails-lambda-output-bucket"
AWS_S3_CLIENT_STORE_DIRECTORY = "supertails-vet-audit"
