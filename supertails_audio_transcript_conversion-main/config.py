# config.py
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY", ""
)  # Set your OPENAI_API_KEY key here if not using env variable

ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY", ""
)  # Set your ANTHROPIC_API_KEY key here if not using env variable

# AWS S3 configuration
AWS_S3_BUCKET_NAME = "supertails-lambda-output-bucket"
AWS_S3_CLIENT_STORE_DIRECTORY = "supertails-vet-audit"
