import sys
import os

# Adiciona o diretório 'src' ao caminho do sistema
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

import numpy as np
import pandas as pd
from langchain_openai import OpenAIEmbeddings
import warnings
from database_service import DatabaseService

warnings.filterwarnings("ignore")

class DocumentSearchService:

    def __init__(self):
        self.database_service = DatabaseService()

    def embeddings_search_on_database(self, table_name: str, query_vec: np.array, sentence_field: str, filter_field_destination: str, product: str, module: str,
                                      use_product: bool, use_module: bool, threshold: int,
                                      similarity: str, ticket_id: int, batch: bool):

        product_statement = f"AND product ilike '{product}'" if use_product else ''
        module_statement = f"AND module ilike '{module}'" if use_module else ''
        field_statement = f"AND sentence_source = '{filter_field_destination}'" if filter_field_destination else ""
        
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
            sentence_field = 'chunk as sentence' if table_name == 'tickets_embeddings_chunks' else sentence_field
            select_statement = f'''SELECT * FROM
                                    (
                                        SELECT te.ticket_id, te.{sentence_field}, 1 - (sentence_embedding {similarity} %s) as score,
                                        (
                                            SELECT
                                                expected_id
                                            FROM
                                                public.tickets_similares
                                            WHERE ticket_id = {ticket_id}
                                        ) as expected_id 
                                        FROM public.{table_name} te
                                        WHERE te.ticket_id <> {ticket_id} 
                                        {product_statement} {module_statement}
                                        {field_statement}
                                    ) as filtered_kb;
                            '''

        result = self.database_service.run_select_statement(select_statement, (query_vec,))
        
        return pd.DataFrame(result)

    def find_tickets_for_query(self, table_name: str, query: str, sentence_field: str, filter_field_destination: str, product: str, module: str, use_product: bool, use_module: bool, threshold: int, k: int, similarity: str, ticket_id: int, batch: bool):
        
        if batch == False:
            # Searching tickets using similarity of OpenAPI embeddings
            query_vec =  OpenAIEmbeddings(
                openai_api_base="https://proxy.dta.totvs.ai/",
                openai_api_key=os.getenv("DTA_PROXY_SECRET_KEY"),
                model="text-embedding-3-small"  
                ).embed_query(query)
            query_vec = np.array(query_vec)
        else:
            query_vec = query

        results = self.embeddings_search_on_database(table_name, query_vec, sentence_field, filter_field_destination, product, module, use_product, use_module, threshold, similarity, ticket_id, batch)

        if results.empty:
            return results
        
        # Getting only results with score higher than threshold
        # results = results[results["score"] >= self.threshold / 100].copy()

        # Ordering results by score
        results.sort_values(by="score", ascending=False, inplace=True)

        # Keeping only the highest rank per ticket'
        results.drop_duplicates(subset=['ticket_id'], keep="first", inplace=True)
        results = results.head(k)

        return results
