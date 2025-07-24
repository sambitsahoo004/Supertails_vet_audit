from google.cloud import spanner
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "D:/PythonGithubAWSLambda/google-cloud/my-python-project-461822-97e81e0b11ae.json"
)

client = spanner.Client(project="my-python-project-461822")
instance = client.instance("freetrial-2025")
database = instance.database("my-google-db")

operation = database.update_ddl(
    [
        """
    CREATE TABLE users (
        id INT64 NOT NULL,
        name STRING(100)
    ) PRIMARY KEY(id)
    """
    ]
)
operation.result()  # Wait for completion
print("Table created.")
