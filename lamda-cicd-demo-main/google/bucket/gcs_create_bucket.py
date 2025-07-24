from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "D:/PythonGithubAWSLambda/google-cloud/my-python-project-461822-97e81e0b11ae.json"
)


def create_bucket(bucket_name, location="ASIA-SOUTH1"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    new_bucket = client.create_bucket(bucket, location=location)
    print(f"Bucket {new_bucket.name} created in {location}")


create_bucket("reetesh-bucket-2025-v1")  # must be globally unique
