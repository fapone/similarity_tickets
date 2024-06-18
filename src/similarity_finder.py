import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from embedding import DatabaseService
from embedding import DocumentSearchService
import time
from google.oauth2 import service_account
import pandas_gbq

class SimilarityFinder:
    def __init__(self):
        pass

    def find_similarity(self, table_name, sentence_field, filter_field_destination, row, use_product, use_module, threshold):

        ticket_id = row['ticket_id']
        sentence = row['sentence']
        product = row['product']
        module = row['module']
        query_vec = np.array(row['sentence_embedding'])

        # Find similar tickets using cosine similarity
        documents_df = DocumentSearchService().find_tickets_for_query(table_name, query_vec, sentence_field, filter_field_destination, product, module, use_product, use_module, threshold, k=20, similarity='<=>', ticket_id=ticket_id, batch=True)
        
        if not documents_df.empty:
            results_df = documents_df

            # Add top ticket IDs to the DataFrame
            results_df['sentence'] = sentence
            results_df['target'] = ticket_id

            # Sort the final DataFrame by score
            df_sorted_final = results_df.sort_values(by='score', ascending=False)

            return df_sorted_final

        return pd.DataFrame()  # Return an empty DataFrame if no results are found

    def process_data(self, table_name_source, table_name_destination, sentence_source, filter_field_destination, use_product, use_module, threshold, full_test: bool = False):

        sentence_source_statement = f"WHERE sentence_source = '{sentence_source}'" if sentence_source else ""

        # Select the ticket_id and sentence_embedding from the Vectorized Database
        if table_name_source == 'tickets_embeddings_summary':
            sentence_field = 'sentence'
            select_statement = (f"""
                        select te.ticket_id, te.sentence, te.product, te.module, te.sentence_embedding 
                        from tickets_embeddings_summary te
                        INNER JOIN tickets_similares ts
                            ON ts.ticket_id = te.ticket_id
                        {sentence_source_statement}
                        """)
        elif table_name_source == 'tickets_embeddings_chunks':
            sentence_field = 'subject'
            select_statement = (f"""
                select te.ticket_id, te.{sentence_field} as sentence, te.product, te.module, te.subject_embedding as sentence_embedding
                from tickets_embeddings_chunks te
                INNER JOIN tickets_similares ts
                    ON ts.ticket_id = te.ticket_id
                """)
        else:
            sentence_field = sentence_source if sentence_source else 'subject'
            select_statement = (f"""
                select te.ticket_id, te.{sentence_field} as sentence, te.product, te.module, te.sentence_embedding 
                from tickets_embeddings te
                INNER JOIN tickets_similares ts
                    ON ts.ticket_id = te.ticket_id
                {sentence_source_statement}
                """)
        
        ticket_inputs = DatabaseService().run_select_statement(select_statement, None)

        df = pd.DataFrame(ticket_inputs, columns=['ticket_id', 'product', 'module', 'sentence', 'sentence_embedding'])

        if table_name_source == 'tickets_embeddings_chunks':
            df = df.drop_duplicates(subset=['ticket_id'])

        print(f'Reading {len(df)} rows from the database...')

        # Initialize the progress bar
        with tqdm(total=len(df), desc="Processing") as pbar:
            # Define a function to wrap find_similarity and update the progress bar
            def process_row(row):
                result = self.find_similarity(table_name_destination, sentence_field, filter_field_destination, row, use_product, use_module, threshold)
                pbar.update(1)
                return result

            # Use Parallel to process each row in parallel
            results = Parallel(n_jobs=-1, backend='threading')(delayed(process_row)(row) for _, row in df.iterrows())

        # Concatenate the DataFrames
        final_df = pd.concat(results, ignore_index=True)
        if final_df.empty:
            print('No results found')
            return {}
        
        final_df.sort_values(by=['target','ticket_id', 'score'], ascending=False, inplace=True)
        final_df.drop_duplicates(subset=['target','ticket_id'], keep="first", inplace=True)
        final_df['hit'] = final_df.apply(lambda row: 1 if self.check_found(row) else 0, axis=1)
        final_df['num_expected_ids'] = final_df['expected_id'].apply(lambda x: len(x.split(',')) if x else 0)

        final_df.to_csv('final_df.csv', index=False)

        result = {'table_name': table_name_destination,
                  'field_query': sentence_source,
                  'field_search': filter_field_destination if filter_field_destination else 'ALL',
                  'use_product': use_product,
                  'use_module': use_module}

        top1 = self.compute_topk(final_df, 1)
        top3 = self.compute_topk(final_df, 3)
        top5 = self.compute_topk(final_df, 5)

        result['top1'] = top1
        result['top3'] = top3
        result['top5'] = top5

        if not full_test:
            print(f'Table name source: {table_name_source.upper()}')
            print(f'Table name destination: {table_name_destination.upper()}')

            print(f"top 1: {top1}")
            print(f"top 3: {top3}")
            print(f"top 5: {top5}")

            print('\n-------------------\n')

        top1_float = self.compute_topk_float(final_df, 1)
        top3_float = self.compute_topk_float(final_df, 3)
        top5_float = self.compute_topk_float(final_df, 5)

        result['top1_float'] = top1_float
        result['top3_float'] = top3_float
        result['top5_float'] = top5_float

        if not full_test:
            print(f"top 1 float: {top1_float}")
            print(f"top 3 float: {top3_float}")
            print(f"top 5 float: {top5_float}")

        return result

        # # # Apply function and create found and not found columns
        # final_df['hit'] = final_df.apply(lambda row: 1 if self.check_found(row) else 0, axis=1)

        # # List of all ticket_id for reference
        # all_ticket_ids = set(group_df['ticket_id'])

        # # Apply function and create found and not found columns
        # group_df[['found']] = group_df.apply(lambda row: pd.Series(self.check_expected(row, all_ticket_ids)), axis=1)

        # # Keep only the necessary columns
        # group_df = group_df[['expected_id', 'sentence', 'target', 'found']]

        # # Group by target and aggregate the results
        # df_grouped = group_df.groupby('target').agg({
        #     'expected_id': 'first',
        #     'sentence': 'first',
        #     'found': 'first'
        # }).reset_index()

        # # Calculate calculate_top_k - top 3 e Top 5
        # df_grouped['ptop3'], df_grouped['ptop5'] = zip(*df_grouped.apply(lambda row: self.calculate_ptopk(final_df, row['target'], [3, 5]), axis=1))

        # # Apply the function to create columns 'top_1' and 'top_3'
        # df_grouped[['top_1', 'top_3']] = df_grouped.apply(lambda row: pd.Series(self.calculate_topk(final_df, row['target'])), axis=1)

        # # Save results to BigQuery
        # DatabaseService().save_dataframe_to_bigquery(df=df_grouped, table_id='result_1_mendes', if_exists='replace')
        # DatabaseService().save_dataframe_to_bigquery(df=final_df, table_id='result_2_mendes', if_exists='replace')

        # return df_grouped
    
    # Function to check if the ticket_id is in the expected_id
    def check_found(self, row):
        expected_ids = [int(x.strip()) for x in row['expected_id'].split(',') if x.strip().isdigit()]
        return row['ticket_id'] in expected_ids

    def compute_topk(self, data: pd.DataFrame, k: int) -> float:
        data_sorted = data.sort_values(["target", "score"], ascending=False)
        data_topk = data_sorted.groupby("target").head(k)
        hits_topk = data_topk.groupby("target")["hit"].apply(lambda x: x.sum() > 0)
        return hits_topk.sum() / len(hits_topk)

    def compute_topk_float(self, data: pd.DataFrame, k: int) -> float:
        data_sorted = data.sort_values(["target", "score"], ascending=False)
        data_topk = data_sorted.groupby("target").head(k)
        hits_topk = data_topk.groupby("target").apply(lambda x: x['hit'].sum() / min(len(x), x['num_expected_ids'].iloc[0]))
        return hits_topk.sum() / len(hits_topk)
