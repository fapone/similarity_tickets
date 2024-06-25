import os
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

class DatabaseService:

    def __init__(self):
        db_user=os.getenv('DBUSER')
        db_password=os.getenv('DBPASSWORD')
        db_database=os.getenv('DBDATABASE')
        db_port=os.getenv('DBPORT')
        db_host=os.getenv('DBHOST')
        self.connection_str = f"host='{db_host}' port='{db_port}' dbname='{db_database}' user='{db_user}' password='{db_password}'"

    def _get_database_connection(self):
        return psycopg2.connect(self.connection_str)

    def run_select_statement(self, select_statement: str, vars=None):
        
        try:
            conn = self._get_database_connection()
            register_vector(conn)
        except Exception as e:
            print(f'Connecting to database failed. {e}')
            return []
        try:
            cursor = conn.cursor()
            cursor.execute(select_statement, vars=vars)
            fields = [field_md[0] for field_md in cursor.description]
            result = cursor.fetchall()
            result = [dict(zip(fields, row)) for row in result]
        except Exception as e:
            print(f'Fetching resuls from database failed. {e}\nSelect statement: {select_statement}')
            conn.rollback()
            result = []

        return result
    
    # Delete all records from a table
    def run_dml_delete_statement(self, table: str):
        
        try:
            conn = self._get_database_connection()
            register_vector(conn)
        except Exception as e:
            print(f'Connecting to database failed. {e}')
            return []
        
        # SQL query to execute 
        query = f'delete from {table}' 
        cursor = conn.cursor() 
        try: 
            cursor.execute(query)
            conn.commit() 
        except (Exception, psycopg2.DatabaseError) as error: 
            print("Error: %s" % error) 
            conn.rollback()  
            
        cursor.close()  
    
    def run_dml_statement(self, df: str, table: str, vars=None): 
        
        try:
            conn = self._get_database_connection()
            register_vector(conn)
        except Exception as e:
            print(f'Connecting to database failed. {e}')
            return []
        
        tuples = [tuple(x) for x in df.to_numpy()] 
  
        cols = ','.join(list(df.columns)) 
        # SQL query to execute 
        query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols) 
        cursor = conn.cursor() 
        try: 
            execute_values(cursor, query, tuples) 
            conn.commit() 
        except (Exception, psycopg2.DatabaseError) as error: 
            print("Error: %s" % error) 
            conn.rollback() 

        cursor.close()  
        
    def run_insert_sentence_embeddings_statement(self, table_name: str, data_list: list):
        try:
            conn = self._get_database_connection()
            register_vector(conn)
        except Exception as e:
            print(f'Connecting to database failed. {e}')
            return
        try:
            cursor = conn.cursor()
            execute_values(cursor, f"""INSERT INTO {table_name} (ticket_id, ticket_comment, ticket_sentence_hash, module, product,
                                   sentence_source, ticket_status, sentence_embedding, created_at, updated_at, sentence) VALUES %s ON CONFLICT DO NOTHING""", data_list)
            conn.commit()
            print(f'Sentences added to dabatase successfully.')
        except Exception as e:
            print(f'Inserting {len(data_list)} rows into database failed. {e}')
            conn.rollback()

    def run_insert_chunks_embeddings_statement(self, table_name: str, data_list: list):
        try:
            conn = self._get_database_connection()
            register_vector(conn)
        except Exception as e:
            print(f'Connecting to database failed. {e}')
            return
        try:
            cursor = conn.cursor()
            execute_values(cursor, f"""INSERT INTO {table_name} (ticket_id, ticket_comment, ticket_chunk_hash, module, product,
                                   ticket_status, sentence_embedding, created_at, updated_at, chunk, subject) VALUES %s ON CONFLICT DO NOTHING""", data_list)
            conn.commit()
            print(f'Sentences added to dabatase successfully.')
        except Exception as e:
            print(f'Inserting {len(data_list)} rows into database failed. {e}')
            conn.rollback()
            raise e