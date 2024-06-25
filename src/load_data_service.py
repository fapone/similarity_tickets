import json
import base64
import os
from google.oauth2.service_account import Credentials
from google.cloud import bigquery


class LoadData:
    
    def run_select_bigquery(select_statement: str):
        if google_credential := os.getenv('GOOGLE_CREDENTIAL'):
            credentials = Credentials.from_service_account_info(json.loads(base64.b64decode(google_credential)))
        elif service_account_file := os.getenv('SERVICE_ACCOUNT_FILE'):
            credentials = Credentials.from_service_account_file(service_account_file)
        else:
            raise Exception('No credentials found. Please set GOOGLE_CREDENTIAL or SERVICE_ACCOUNT_FILE environment variable.')

        project_id = 'labs-poc'
        client = bigquery.Client(credentials= credentials,project=project_id)

        try:
            # Perform a query.
            query_job = client.query(select_statement)  # API request
            result = query_job.result()
        except Exception as e:
            print(f'Fetching resuls from database failed. {e}\nSelect statement: {select_statement}')
            raise
    
        return result