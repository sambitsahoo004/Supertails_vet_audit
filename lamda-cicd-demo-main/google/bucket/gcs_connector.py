from google.cloud import storage
import os


class GCSConnector:
    def __init__(self, bucket_name, key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def list_files(self):
        """List all files in the bucket"""
        blobs = self.bucket.list_blobs()
        return [blob.name for blob in blobs]

    def download_file(self, source_blob_name, destination_file_name):
        """Download a file from the bucket"""
        blob = self.bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} to {destination_file_name}")

    def upload_file(self, source_file_name, destination_blob_name):
        """Upload a file to the bucket"""
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"Uploaded {source_file_name} to {destination_blob_name}")

    def delete_file(self, blob_name):
        """Delete a file from the bucket"""
        blob = self.bucket.blob(blob_name)
        blob.delete()
        print(f"Deleted {blob_name} from the bucket")

    def copy_file(self, source_blob_name, destination_blob_name):
        """Copy a file within the bucket"""
        source_blob = self.bucket.blob(source_blob_name)
        destination_blob = self.bucket.blob(destination_blob_name)
        destination_blob.rewrite(source_blob)
        print(f"Copied {source_blob_name} to {destination_blob_name}")

    def move_file(self, source_blob_name, destination_blob_name):
        """Move a file within the bucket (copy and delete)"""
        self.copy_file(source_blob_name, destination_blob_name)
        self.delete_file(source_blob_name)
        print(f"Moved {source_blob_name} to {destination_blob_name}")

    def exists(self, blob_name):
        """Check if a file exists in the bucket"""
        blob = self.bucket.blob(blob_name)
        return blob.exists()

    def get_bucket_name(self):
        """Get the name of the bucket"""
        return self.bucket.name

    def get_bucket_location(self):
        """Get the location of the bucket"""
        return self.bucket.location if self.bucket.exists() else None
