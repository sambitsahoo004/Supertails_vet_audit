from spanner_connector import SpannerConnector

if __name__ == "__main__":
    spanner = SpannerConnector(
        project_id="my-python-project-461822",
        instance_id="freetrial-2025",
        database_id="my-google-db",
        key_path="D:/PythonGithubAWSLambda/google-cloud/my-python-project-461822-97e81e0b11ae.json",
    )

    # Insert example
    spanner.insert_row(table="users", columns=["id", "name"], values=[1, "Alice"])

    # Read example
    result = spanner.read_rows(table="users", columns=["id", "name"])
    print("Result:", result)

    # Update example
    spanner.update_row(
        table="users", columns=["id", "name"], values=[1, "Alice Updated"]
    )

    # Delete example
    spanner.delete_row(table="users", key_columns=["id"], key_values=[1])
