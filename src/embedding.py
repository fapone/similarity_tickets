import sys
import os

# Adiciona o diretório 'src' ao caminho do sistema
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

import copy
import requests
import numpy as np
import pandas as pd
import db_dtypes
# import ftfy
# import re
# import six
# import tiktoken
# import pandas_gbq
# from unidecode import unidecode
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.question_answering import load_qa_chain
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
# from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import psycopg2
from pgvector.psycopg2 import register_vector
import json
import os
from google.oauth2 import service_account
from google.cloud import bigquery
#from asyncio.log import logger
#from util import transform_sentence
#from logger import get_logger
import psycopg2.extras as extras 
import hashlib
from langchain.docstore.document import Document
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import textwrap
import openai
import warnings
from joblib import Parallel, delayed
import pandas_gbq
import time

warnings.filterwarnings("ignore")


class LoadData:
    
    def run_select_bigquery(select_statement: str):
        credentials = service_account.Credentials.from_service_account_file('/Users/rodrigomoraes/Library/CloudStorage/GoogleDrive-rg.moraes@totvs.com.br/My Drive/TOTVS LABS/key_SA_GCP/labs-poc-09feb4e7688e.json')

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


class DatabaseService:

    def __init__(self):
        db_user='tecsupport'
        db_password='?Hi((<={}F{nI=jp'
        db_database='tecsupport'
        db_port='5432'
        db_host='34.123.172.21'
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
            return 1
        
        return
    
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
            extras.execute_values(cursor, query, tuples) 
            conn.commit() 
        except (Exception, psycopg2.DatabaseError) as error: 
            print("Error: %s" % error) 
            conn.rollback() 
            cursor.close() 
            return 1
        
        # query = "SELECT COUNT(*) as cnt FROM %s;" % (table) 
        # cursor.execute(query)
        # num_records = cursor.fetchone()[0]

        # print("Number of vector records in table: ", num_records,"\n")

        # Create an index on the data for faster retrieval

        #calculate the index parameters according to best practices
        # num_lists = num_records / 1000
        # if num_lists < 10:
        #     num_lists = 10
        # if num_records > 1000000:
        #     num_lists = math.sqrt(num_records)

        #use the cosine distance measure, which is what we'll later use for querying
        # cursor.execute(f'CREATE INDEX ON {table} USING ivfflat (sentence_embedding vector_cosine_ops) WITH (lists = {num_lists});')
        # cursor.execute(f'CREATE INDEX idx ON {table} USING hnsw (sentence_embedding vector_l2_ops);')
        # conn.commit()

        cursor.close()  

        return
    
    def save_dataframe_to_bigquery(self, df, table_id, if_exists='replace'):
        
        # Create credentials
        credentials = service_account.Credentials.from_service_account_file('/Users/rodrigomoraes/Library/CloudStorage/GoogleDrive-rg.moraes@totvs.com.br/My Drive/TOTVS LABS/key_SA_GCP/labs-poc-09feb4e7688e.json')

        project_id = 'labs-poc'
        dataset_id = 'custom_data'

        # Save the DataFrame to BigQuery
        pandas_gbq.to_gbq(df, f'{dataset_id}.{table_id}', project_id=project_id, if_exists=if_exists, credentials=credentials)

class DocumentSearchService:

    def __init__(self):
        self.database_service = DatabaseService()
        self.threshold = 65

    def ticket_summarization(self, sentence: str, llm):
        """Returns ticket summarization by comment.

        :param sentence: Sentence to summarization.
        :param llm: LLMChain object.
        """

        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(sentence)
        docs = [Document(page_content=t) for t in texts[:4]]
        
        # prompt_template = """Write a concise topic based on the following passages, in Brazilian Portuguese, 
        #                      disregarding all personal names, date, attach and url of the following:

        # Perguntar ao prompt sobre requisicao e dúvida, fornecendo os comentarios, explicar qual a duvida/problema, 
        # com os demais itens (nomes, url...)
        # with a custom prompt
        prompt_template = """Write a concise summary, in chronological order, in Brazilian Portuguese, 
                            disregarding all personal names, date, attach and url, of the following:


        {text}


        CONCISE SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template,
                                input_variables=["text"])

        ## with intermediate steps
        chain = load_summarize_chain(llm,
                                    chain_type="map_reduce",
                                    return_intermediate_steps=True,
                                    map_prompt=PROMPT,
                                    combine_prompt=PROMPT)

        output_summary = chain({"input_documents": docs}, return_only_outputs=True)
        wrapped_text = textwrap.fill(output_summary['output_text'],
                                    width=100,
                                    break_long_words=False,
                                    replace_whitespace=False)
    
        return wrapped_text

    def embeddings_search_on_database(self, query_vec: np.array, product: str, module: str,
                                      threshold: int, similarity: str, ticket_id: int, batch: bool):

        table_name = 'tickets_embeddings'
        
        if batch == False:
            select_statement = f'''SELECT * FROM
                                    (
                                        SELECT *, 1 - (sentence_embedding {similarity} %s) as score 
                                        FROM public.{table_name}
                                        WHERE (product ~ 'ˆ{product}.*' OR 
                                                module ~ '{module}')
                                            and ticket_id <> {ticket_id}
                                    ) as filtered_kb
                                WHERE score > {threshold/100};
                            '''
        else:
            select_statement = f'''SELECT * FROM
                                    (
                                        SELECT te.ticket_id, 1 - (sentence_embedding {similarity} %s) as score,
                                        (
                                            SELECT
                                                expected_id
                                            FROM
                                                public.tickets_similares
                                            WHERE ticket_id = {ticket_id}
                                        ) as expected_id 
                                        FROM public.{table_name} te
                                        WHERE te.ticket_id <> {ticket_id}
                                    ) as filtered_kb
                                WHERE score > {threshold/100};
                            '''

        result = self.database_service.run_select_statement(select_statement, (query_vec,))
        
        return pd.DataFrame(result)

    def find_tickets_for_query(self, query: str, product: str, module: str, k: int, similarity: str, ticket_id: int, batch: bool):
        
        if batch == False:
            # Searching tickets using similarity of OpenAPI embeddings
            query_vec =  OpenAIEmbeddings(
                openai_api_base="https://proxy.dta.totvs.ai/",
                openai_api_key="sk-axyZ_tPhqNPbbywhdhhhKQ",
                model="text-embedding-3-small"  
                ).embed_query(query)
            query_vec = np.array(query_vec)
        else:
            query_vec = query

        results = self.embeddings_search_on_database(query_vec, product, module, self.threshold, similarity, ticket_id, batch)

        if results.empty:
            return results
        
        # Getting only results with score higher than threshold
        results = results[results["score"] >= self.threshold / 100].copy()

        # Ordering results by score
        results.sort_values(by="score", ascending=False, inplace=True)

        # Keeping only the highest rank per ticket'
        results.drop_duplicates(subset=['ticket_id'], keep="first", inplace=True)
        results = results.head(k)

        return results
    

    # Create similar tickets from the sent spreadsheet
    def similarity_ticket():

        select_statement = ("""
                            SELECT
                                CAST(ticket_id AS INT64) AS ticket_id,
                                expected_id
                            FROM
                                `labs-poc.custom_data.tickets_similares`
                            """)
        
        df = LoadData.run_select_bigquery(select_statement).to_dataframe()

        # Grava o resultado dos dados coletados no BQ e faz insert do Dataframe diretamente no Banco Vetorizado
        DatabaseService().run_dml_statement(df, 'tickets_similares')

        return df

    def embedding_sentence(result_df, i, llm, embedding):

        time.sleep(2)

        # Cria os embeddings para inserir os registros.
        #for i, row in result_df.iterrows():
        sentence = result_df.at[i,'ticket_comment']
        summary = DocumentSearchService().ticket_summarization(sentence, llm)

        # Cria embeddings para ticket_comment
        query_vec = embedding.embed_query(sentence)
        query_vec = np.array(query_vec)

        # Atualiza os valores da linha atual
        hash_concat = str(result_df.at[i,'ticket_id']) + sentence
        hash_id = hashlib.md5(hash_concat.encode('utf-8')).hexdigest()
        result_df.at[i,'ticket_sentence_hash'] =  hash_id
        result_df.at[i,'summary'] = summary
        result_df.at[i,'sentence_embedding'] = query_vec
        result_df.at[i,'sentence_source'] = 'ticket_comment'

        # Grava o resultado dos dados coletados no BQ e faz insert do Dataframe diretamente no Banco Vetorizado
        DatabaseService().run_dml_statement(result_df.iloc[[i]], 'tickets_embeddings')

        # Cria nova linha com dados do Assunto
        ultima_linha = result_df.loc[i]
        nova_linha = ultima_linha.copy()
        sentence = nova_linha['subject']
        hash_concat = str(nova_linha['ticket_id']) + sentence
        hash_id = hashlib.md5(hash_concat.encode('utf-8')).hexdigest()

        # Cria embeddings para a sentença
        query_vec = embedding.embed_query(sentence)
        query_vec = np.array(query_vec)

        # Atualiza os valores da linha atual
        nova_linha['ticket_sentence_hash'] =  hash_id
        nova_linha['summary'] = summary
        nova_linha['sentence_embedding'] = query_vec
        nova_linha['sentence_source'] = 'subject'
        result_df = result_df.append(nova_linha, ignore_index=True)

        # Grava o resultado dos dados coletados no BQ e faz insert do Dataframe diretamente no Banco Vetorizado
        DatabaseService().run_dml_statement(result_df.tail(1), 'tickets_embeddings')

        # Cria nova linha com dados do Resumo
        sentence = summary
        ultima_linha = result_df.loc[i]
        nova_linha = ultima_linha.copy()
        hash_concat = str(nova_linha['ticket_id']) + sentence
        hash_id = hashlib.md5(hash_concat.encode('utf-8')).hexdigest()

        # Cria embeddings para a sentença
        query_vec = embedding.embed_query(sentence)
        query_vec = np.array(query_vec)

        nova_linha['ticket_sentence_hash'] =  hash_id
        nova_linha['summary'] = summary
        nova_linha['sentence_embedding'] = query_vec
        nova_linha['sentence_source'] = 'summary'
        result_df = result_df.append(nova_linha, ignore_index=True)

        # Grava o resultado dos dados coletados no BQ e faz insert do Dataframe diretamente no Banco Vetorizado
        DatabaseService().run_dml_statement(result_df.tail(1), 'tickets_embeddings')

        # Cria nova linha com dados do primeiro comentario
        ultima_linha = result_df.loc[i]
        nova_linha = ultima_linha.copy()
        sentence = nova_linha['first_comment']
        hash_concat = str(nova_linha['ticket_id']) + sentence + summary
        hash_id = hashlib.md5(hash_concat.encode('utf-8')).hexdigest()

        # Cria embeddings para a sentença
        query_vec = embedding.embed_query(sentence)
        query_vec = np.array(query_vec)

        # Atualiza os valores da linha atual
        nova_linha['ticket_sentence_hash'] =  hash_id
        nova_linha['summary'] = summary
        nova_linha['sentence_embedding'] = query_vec
        nova_linha['sentence_source'] = 'first_comment'
        result_df = result_df.append(nova_linha, ignore_index=True)

        # Grava o resultado dos dados coletados no BQ e faz insert do Dataframe diretamente no Banco Vetorizado
        DatabaseService().run_dml_statement(result_df.tail(1), 'tickets_embeddings')

        return result_df

    def create_embeddings():
        select_statement = ("""
                            WITH tickets_all AS (
                                SELECT
                                    ticket_id,
                                    STRING_AGG(ticket_comment, '\u2561') AS ticket_comment,
                                    MAX(subject) AS subject,
                                    '' AS summary,
                                    '' ticket_sentence_hash,
                                    MAX(module_name) AS module,
                                    MAX(product_name) AS product,
                                    '' AS sentence_source,
                                    MAX(ticket_status) AS ticket_status,
                                    '[1,2,3]' AS sentence_embedding,
                                    MAX(created_at) AS created_at,
                                    MAX(updated_at) AS updated_at,
                                FROM
                                    `labs-poc`.custom_data.tickets tr
                                GROUP BY
                                    ticket_id
                                ),

                                first_contact AS (
                                SELECT
                                    ts.ticket_comment AS first_comment,
                                    ts.ticket_id
                                FROM
                                    `labs-poc`.custom_data.tickets ts
                                INNER JOIN
                                    tickets_all tr
                                ON
                                    ts.ticket_id = tr.ticket_id
                                QUALIFY ROW_NUMBER() OVER(PARTITION BY ts.ticket_id ORDER BY ts.comment_created_at) = 1
                                )

                                SELECT
                                    ta.*,
                                    tr.first_comment
                                FROM
                                    tickets_all ta
                                INNER JOIN
                                    first_contact tr
                                    ON tr.ticket_id = ta.ticket_id

                            """)
        # Faz a leitura dos daods no BQ
        df = LoadData.run_select_bigquery(select_statement).to_dataframe()

        dados = df

        # Instancia llm e embedding
        llm = OpenAI(
                    openai_api_base="https://proxy.dta.totvs.ai/",
                    openai_api_key="sk-axyZ_tPhqNPbbywhdhhhKQ",
                    temperature=0,
                    model="gpt-4o",
                    )

        embedding =  OpenAIEmbeddings(
                    openai_api_base="https://proxy.dta.totvs.ai/",
                    openai_api_key="sk-axyZ_tPhqNPbbywhdhhhKQ",
                    model="text-embedding-3-small"  
                    )

        # Usa o joblib para paralelizar o processamento. 10 jobs por vez usando o backend threading.
        Parallel(n_jobs=10, backend='threading')(delayed(DocumentSearchService().embedding_sentence)(
                dados,i, llm, embedding)
                for i, row in dados.iterrows())

        # Gera dados na tabela de tickets similares, estes dados vem da planilha enviada para busca de ticket similares
        DocumentSearchService().similarity_ticket()

        return 