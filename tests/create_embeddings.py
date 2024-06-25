import sys
import os
from dotenv import load_dotenv
from embeddings import create_embeddings_and_sent_to_db

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

if __name__ == "__main__":
    load_dotenv()
    table_destination = os.getenv('EMBEDDINGS_TABLE_DESTINATION')
    create_embeddings_and_sent_to_db(table_destination)