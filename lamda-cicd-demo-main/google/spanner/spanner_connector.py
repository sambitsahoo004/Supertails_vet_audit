import os
from google.cloud import spanner
from google.cloud.spanner_v1 import param_types


class SpannerConnector:
    def __init__(self, project_id, instance_id, database_id, key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

        self.client = spanner.Client(project=project_id)
        self.instance = self.client.instance(instance_id)
        self.database = self.instance.database(database_id)

    def insert_row(self, table, columns, values):
        with self.database.batch() as batch:
            batch.insert(table=table, columns=columns, values=[values])

    def read_rows(self, table, columns):
        with self.database.snapshot() as snapshot:
            results = snapshot.read(
                table=table, columns=columns, keyset=spanner.KeySet(all_=True)
            )
            return [list(row) for row in results]

    def execute_query(self, query, params=None, param_types_=None):
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                query, params=params, param_types=param_types_
            )
            return [list(row) for row in results]

    def update_row(self, table, columns, values):
        with self.database.batch() as batch:
            batch.update(table=table, columns=columns, values=[values])

    def delete_row(self, table, key_columns, key_values):
        key = spanner.KeySet(keys=[tuple(key_values)])
        with self.database.batch() as batch:
            batch.delete(table=table, keyset=key)
