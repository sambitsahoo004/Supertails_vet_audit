# config.py
import os

from dotenv import load_dotenv

load_dotenv()

GCP_SERVICE_ACCOUNT_KEY = os.environ.get(
    "GCP_SERVICE_ACCOUNT_KEY", ""
)  # Set your GCP_SERVICE_ACCOUNT_KEY key here if not using env variable

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY", ""
)  # Set your OPENAI_API_KEY key here if not using env variable

ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY", ""
)  # Set your ANTHROPIC_API_KEY key here if not using env variable

QDRANT_URL = os.environ.get(
    "QDRANT_URL", ""
)  # Set your QDRANT_URL key here if not using env variable

QDRANT_API_KEY = os.environ.get(
    "QDRANT_API_KEY", ""
)  # Set your QDRANT_API_KEY key here if not using env variable

# Qdrant database configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "veterinary_knowledge"

FEEDBACK_THRESHOLD = 70

# Cloud storage configuration
# Google Cloud Storage (GCS) configuration - local setup
# Uncomment the following lines to use local GCS configuration
GCS_BUCKET_NAME = "reetesh-bucket-2025-v1"
GCS_PROJECT_NAME = "my-python-project"

# Google Cloud Storage (GCS) configuration - supertails setup
# Uncomment the following lines to use supertails GCS configuration
# GCS_BUCKET_NAME = "cx-call-recordings"
# GCS_PROJECT_NAME = "GA4 Data API"

# AWS S3 configuration
AWS_S3_BUCKET_NAME = "supertails-lambda-output-bucket"
AWS_S3_CLIENT_STORE_DIRECTORY = "supertails-vet-audit"

# Document processing configuration
DOCS_FOLDER = "./docs"  # Folder to store documents
CHUNKS_FILE = "./data/chunks.json"  # Path to store processed chunks
CHUNK_SIZE = 500  # Size of text chunks in characters (matching your chunker default)
CHUNK_OVERLAP = 50  # Overlap between chunks (matching your chunker default)

# Call Duration Validation Settings
MIN_CALL_DURATION_SECONDS = 15  # Minimum call duration in seconds
PROCEED_WITHOUT_DURATION = True  # Whether to proceed if no duration found in transcript


# Parameters and their details
PARAMETERS_CONFIG = {
    "parameters": [
        # Category 1: Introduction parameters (Total: 20 points)
        {
            "name": "Self Introduction",
            "max_score": 3,
            "subparameters": [
                {"name": "Greeting", "weight": 1.5},
                {"name": "Introduction of self", "weight": 1.5},
            ],
        },
        {
            "name": "Company Introduction",
            "max_score": 3,
            "subparameters": [
                {"name": "Company name mentioned", "weight": 3},
            ],
        },
        {
            "name": "Customer Name Confirmation",
            "max_score": 3,
            "subparameters": [
                {"name": "Asked for customer name", "weight": 3},
            ],
        },
        {
            "name": "Order Confirmation",
            "max_score": 3,
            "subparameters": [
                {"name": "Asked for order details", "weight": 3},
            ],
        },
        {
            "name": "Purpose Of the Call (Context)",
            "max_score": 8,
            "subparameters": [
                {"name": "Explained purpose of the call", "weight": 8},
            ],
        },
        # Category 2: Pet Information parameters (Total: 15 points)
        {
            "name": "Pet Name, Age and Gender",
            "max_score": 5,
            "subparameters": [
                {"name": "Asked for pet's name", "weight": 2},
                {"name": "Asked for pet's age", "weight": 1.5},
                {"name": "Asked for pet's gender", "weight": 1.5},
            ],
        },
        {
            "name": "Pet Body Weight",
            "max_score": 5,
            "subparameters": [
                {"name": "Vet's questions about body weight", "weight": 2.5},
                {"name": "User's responses about body weight", "weight": 2.5},
            ],
        },
        {
            "name": "Health Concern",
            "max_score": 5,
            "subparameters": [
                {"name": "Vet's questions about health concern", "weight": 3},
                {"name": "User's description of health concern", "weight": 2},
            ],
        },
        # Category 3 - Communication Skills parameters (Total: 25 points)
        {
            "name": "Tone & Voice Modulation",
            "max_score": 5,
            "subparameters": [
                {"name": "Maintains friendly tone", "weight": 2.5},
                {"name": "Maintains engaging tone", "weight": 2.5},
            ],
        },
        {
            "name": "Clear Communication",
            "max_score": 5,
            "subparameters": [
                {"name": "Uses simple language", "weight": 2.5},
                {"name": "Avoids jargon", "weight": 2.5},
            ],
        },
        {
            "name": "Engagement & Rapport Building",
            "max_score": 5,
            "subparameters": [
                {"name": "Connects personally using pet's name", "weight": 2.5},
                {"name": "Acknowledges pet parent's feelings", "weight": 2.5},
            ],
        },
        {
            "name": "Patience & Attentiveness",
            "max_score": 5,
            "subparameters": [
                {"name": "Remains calm", "weight": 2.5},
                {"name": "Addresses concerns fully", "weight": 2.5},
            ],
        },
        {
            "name": "Empathy & Compassion",
            "max_score": 5,
            "subparameters": [
                {"name": "Shows genuine care", "weight": 2.5},
                {"name": "Shows concern for pet's well-being", "weight": 2.5},
            ],
        },
        # Category 4: Technical Assessment parameters (Total: 25 points)
        {
            "name": "Diet Confirmation",
            "max_score": 3,
            "subparameters": [
                {"name": "Asked about diet confirmation", "weight": 1.5},
                {"name": "Confirmed diet information", "weight": 1.5},
            ],
        },
        {
            "name": "Food Brand Name",
            "max_score": 3,
            "subparameters": [
                {"name": "Confirmed food brand", "weight": 3},
            ],
        },
        {
            "name": "Technical Parameters",
            "max_score": 4,
            "subparameters": [
                {"name": "Technical assessment", "weight": 3},
            ],
        },
        {
            "name": "Treatment Plan and Instructions PSP",
            "max_score": 3,
            "subparameters": [
                {"name": "Explains treatment plan", "weight": 1.5},
                {"name": "Provides PSP instructions", "weight": 1.5},
            ],
        },
        {
            "name": "Medicine Name",
            "max_score": 4,
            "subparameters": [
                {"name": "Appropriate medication naming", "weight": 3},
            ],
        },
        {
            "name": "Medicine Prescribed",
            "max_score": 4,
            "subparameters": [
                {"name": "Proper prescription protocols", "weight": 3},
            ],
        },
        {
            "name": "Medicine Dosage",
            "max_score": 4,
            "subparameters": [
                {"name": "Explains dosage", "weight": 1},
                {"name": "Explains frequency", "weight": 1},
            ],
        },
        # Category 5: Call Conclusion parameters (Total: 15 points)
        {
            "name": "Provided 'Chat with the Vet' Reminder at Call Closing",
            "max_score": 5,
            "subparameters": [
                {"name": "Explicitly mentions chat with vet option", "weight": 2.5},
                {"name": "Explains how to access the service", "weight": 2.5},
            ],
        },
        {
            "name": "Previous Prescription",
            "max_score": 5,
            "subparameters": [
                {"name": "Inquires about previous medications", "weight": 2.5},
                {
                    "name": "Discusses effectiveness of previous treatments",
                    "weight": 2.5,
                },
            ],
        },
        {
            "name": "Medicine Usage",
            "max_score": 5,
            "subparameters": [
                {
                    "name": "Provides clear instructions on medication usage",
                    "weight": 2.5,
                },
                {
                    "name": "Confirms customer understands the instructions",
                    "weight": 2.5,
                },
            ],
        },
    ]
}
