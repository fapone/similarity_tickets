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

    def find_similarity(self, df, product, module, sentence, index, pbar):
        time.sleep(2)

        ticket_id = df.at[index, 'ticket_id']
        query_vec = np.array(df.at[index, 'sentence_embedding'])

        # Find similar tickets using cosine similarity
        documents_df = DocumentSearchService().find_tickets_for_query(query_vec, product, module, k=20, similarity='<=>', ticket_id=ticket_id, batch=True)
        
        if not documents_df.empty:
            results_df = documents_df

            # Sort the DataFrame by score in descending order
            df_sorted = results_df.sort_values(by='score', ascending=False)

            # # Select the top three ticket_ids
            # top_1 = df_sorted.index[0]
            # top_2 = df_sorted.index[1] if len(df_sorted) > 1 else None
            # top_3 = df_sorted.index[2] if len(df_sorted) > 2 else None

            # # Additional columns corresponding only to found indexes
            # results_df['top_1'] = results_df.index == top_1
            # results_df['top_2'] = results_df.index == top_2 if top_2 else False
            # results_df['top_3'] = results_df.index == top_3 if top_3 else False

            # Add top ticket IDs to the DataFrame
            results_df['sentence'] = sentence
            results_df['target'] = ticket_id

            # Sort the final DataFrame by score
            df_sorted_final = results_df.sort_values(by='score', ascending=False)

            return df_sorted_final

        return pd.DataFrame()  # Return an empty DataFrame if no results are found

    def process_data(self, product, module, sentence):

        # Select the ticket_id and sentence_embedding from the Vectorized Database
        select_statement = (f"""
                    select te.ticket_id, sentence_embedding 
                    from tickets_embeddings te
                    INNER JOIN tickets_similares ts
                        ON ts.ticket_id = te.ticket_id
                    where sentence_source = '{sentence}'
                    """)
        
        ticket_inputs = DatabaseService().run_select_statement(select_statement, None)

        dados = pd.DataFrame(ticket_inputs, columns=['ticket_id', 'sentence_embedding'])

        # Initialize the progress bar
        with tqdm(total=len(dados), desc="Processing") as pbar:
            # Define a function to wrap find_similarity and update the progress bar
            def process_row(index):
                result = self.find_similarity(dados, product, module, sentence, index, pbar)
                pbar.update(1)
                return result

            # Use Parallel to process each row in parallel
            results = Parallel(n_jobs=-1, backend='threading')(delayed(process_row)(i) for i in range(len(dados)))

        # Concatenate the DataFrames
        final_df = pd.concat(results, ignore_index=True)
        group_df = pd.concat(results, ignore_index=True) 

        # # Apply function and create found and not found columns
        final_df['found'] = final_df.apply(lambda row: 1 if self.check_found(row) else 0, axis=1)

        # # Delete the previous results from the Vectorized Database
        # DatabaseService().run_dml_delete_statement(table='result_1')
        # DatabaseService().run_dml_delete_statement(table='result_2')

        # List of all ticket_id for reference
        all_ticket_ids = set(group_df['ticket_id'])

        # Apply function and create found and not found columns
        group_df[['found']] = group_df.apply(lambda row: pd.Series(self.check_expected(row, all_ticket_ids)), axis=1)

        # Keep only the necessary columns
        group_df = group_df[['expected_id', 'sentence', 'target', 'found']]

        # Group by target and aggregate the results
        df_grouped = group_df.groupby('target').agg({
            'expected_id': 'first',
            'sentence': 'first',
            'found': 'first'
        }).reset_index()

        # Calculate calculate_top_k - top 3 e Top 5
        df_grouped['ptop3'], df_grouped['ptop5'] = zip(*df_grouped.apply(lambda row: self.calculate_ptopk(final_df, row['target'], [3, 5]), axis=1))

        # Apply the function to create columns 'top_1' and 'top_3'
        df_grouped[['top_1', 'top_3']] = df_grouped.apply(lambda row: pd.Series(self.calculate_topk(final_df, row['target'])), axis=1)

        # Save results to BigQuery
        DatabaseService().save_dataframe_to_bigquery(df=df_grouped, table_id='result_1', if_exists='replace')
        DatabaseService().save_dataframe_to_bigquery(df=final_df, table_id='result_2', if_exists='replace')

        return

    # This function extracts expected ticket IDs, checks their presence in all_ticket_ids (converted to a set), 
    # and returns counts of found and not found IDs.
    def check_expected(self, row, all_ticket_ids):
        try:
            expected_ids = [int(x.strip()) for x in row['expected_id'].split(',') if x.strip().isdigit()]
        except KeyError:
            raise ValueError("The expected_id key is missing.")
        except ValueError:
            raise ValueError("The expected_id field contains non-numeric values.")
        
        all_ticket_ids_set = set(all_ticket_ids)
        
        found_count = sum(1 for eid in expected_ids if eid in all_ticket_ids_set)

        return found_count
    
    # Function to check if the ticket_id is in the expected_id
    def check_found(self, row):
        expected_ids = [int(x.strip()) for x in row['expected_id'].split(',') if x.strip().isdigit()]
        return row['ticket_id'] in expected_ids

    # Function to calculate the percentage of 'found' records in the top k records
    def calculate_ptopk(self, final_df, target, k_values):
        filtered_df = final_df[final_df['target'] == target]
        filtered_df_found_1 = filtered_df[filtered_df['found'] == 1]
        
        ptopk_values = []
        for k in k_values:
            n = min(len(filtered_df_found_1), k)
            if n == 0:
                ptopk_values.append(0)
            else:
                ptopk_values.append(sum(filtered_df_found_1['found'].head(n)) / n)
        
        return ptopk_values

    # Function to calculate top_1 and top_3
    def calculate_topk(self, final_df, target):
        filtered_df = final_df[final_df['target'] == target]
        if len(filtered_df) < 3:
            top_1 = filtered_df.iloc[0]['found'] == 1 if not filtered_df.empty else False
            top_3 = False
        else:
            top_1 = filtered_df.iloc[0]['found'] == 1 if not filtered_df.empty else False
            top_3 = (filtered_df['found'].head(3).sum() >= 3) if not filtered_df.empty else False
        return top_1, top_3


        
