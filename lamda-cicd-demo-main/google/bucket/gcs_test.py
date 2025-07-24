from gcs_connector import GCSConnector

if __name__ == "__main__":
    gcs = GCSConnector(
        bucket_name="reetesh-bucket-2025-v1",
        key_path="D:/PythonGithubAWSLambda/google-cloud/my-python-project-461822-97e81e0b11ae.json",
    )

    print("Files in bucket 1:")
    print(gcs.list_files())

    # upload a specific file
    gcs.upload_file(
        source_file_name="D:/PythonGithubAWSLambda/google-cloud/test.txt",
        destination_blob_name="blob_test.txt",
    )

    print("Files in bucket 2:")
    print(gcs.list_files())

    # download a specific file
    gcs.download_file(
        source_blob_name="blob_test.txt",
        destination_file_name="D:/PythonGithubAWSLambda/google-cloud/test_downloaded.txt",
    )
