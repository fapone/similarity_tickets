from joblib import Parallel, delayed
import time
from tqdm import tqdm
import embedding
import numpy as np
import pandas as pd

class SearchService:
    
    def __init__(self):
        self.all_ticket_ids = set()
        
    # Function to find similar tickets
    def find_similarity(self, df, product, module, sentence, index, pbar):

        time.sleep(2)

        ticket_id = df.at[index, 'ticket_id']
        query_vec = np.array(df.at[index, 'sentence_embedding'])

        # Find similar tickets using cosine similarity
        documents_df = embedding.DocumentSearchService().find_tickets_for_query(query_vec, product, module, k=20, similarity='<=>', ticket_id=ticket_id, batch=True)
        
        if not documents_df.empty:
            results_df = documents_df

            # Sort the DataFrame by score in descending order
            df_sorted = results_df.sort_values(by='score', ascending=False)

            # Seleciona os três principais ticket_id
            top_1 = df_sorted.index[0]
            top_2 = df_sorted.index[1] if len(df_sorted) > 1 else None
            top_3 = df_sorted.index[2] if len(df_sorted) > 2 else None

            # Additional columns corresponding only to found indexes
            results_df['Top 1'] = results_df.index == top_1
            results_df['Top 2'] = results_df.index == top_2 if top_2 else False
            results_df['Top 3'] = results_df.index == top_3 if top_3 else False

            # Add top ticket IDs to the DataFrame
            results_df['sentence'] = sentence
            results_df['target'] = ticket_id

            # Sort the final DataFrame by score
            df_sorted_final = results_df.sort_values(by='score', ascending=False)

            return df_sorted_final

        return pd.DataFrame()  # Return an empty DataFrame if no results are found
    
    # Function to check if the ticket_id is in the expected_id
    def check_found(self, row):
        expected_ids = [int(x.strip()) for x in row['expected_id'].split(',') if x.strip().isdigit()]
        return row['ticket_id'] in expected_ids

    # Function to check if the expected_id is in the ticket_id summarization found and not_found
    def check_expected(self, row):
        expected_ids = [int(x.strip()) for x in row['expected_id'].split(',') if x.strip().isdigit()]
        found_count = sum(1 for eid in expected_ids if eid in self.all_ticket_ids)
        not_found_count = sum(1 for eid in expected_ids if eid not in self.all_ticket_ids)
        return found_count, not_found_count

    def fetch_embeddings(self, product, module, sentence):
  
        # Connecting to the database
        conn = embedding.DatabaseService()._get_database_connection()
        cur = conn.cursor()

        # Execute the query to fetch the embeddings for each ticket_id
        ticket_inputs = []
        cur.execute(f"""
                    select te.ticket_id, sentence_embedding 
                    from tickets_embeddings te
                    INNER JOIN tickets_similares ts
                        ON ts.ticket_id = te.ticket_id
                    where sentence_source = '{sentence}'
                    """)
        ticket_inputs = cur.fetchall()
        #ticket_inputs.extend(ticket_input)

        embedding_df = pd.DataFrame(ticket_inputs, columns=['ticket_id', 'sentence_embedding'])

        # Close the connection
        cur.close()
        conn.close()

        return embedding_df

    def accurancy_metrics(self, product, module, sentence):
        
        # Fetch the embeddings from the database
        dados = SearchService().fetch_embeddings(product, module, sentence)

        # Initialize the progress bar
        with tqdm(total=len(dados), desc="Processing") as pbar:
            # Define a function to wrap find_similarity and update the progress bar
            def process_row(index):
                result = SearchService().find_similarity(dados, product, module, sentence, index, pbar)
                pbar.update(1)
                return result

            # Use Parallel to process each row in parallel
            results = Parallel(n_jobs=-1, backend='threading')(delayed(process_row)(i) for i in range(len(dados)))

        # Concatenate the DataFrames
        final_df = pd.concat(results, ignore_index=True)
        group_df = pd.concat(results, ignore_index=True) 

        # Apply function and create found and not found columns
        final_df['found'] = final_df.apply(lambda row: 1 if SearchService().check_found(row) else 0, axis=1)
        final_df['not_found'] = final_df.apply(lambda row: 0 if SearchService().check_found(row) else 1, axis=1)

        # Export the final DataFrame to Excel
        final_df.to_excel('final_results.xlsx', index=False)

        # List of all ticket_id for reference
        self.all_ticket_ids = set(group_df['ticket_id'])
        #all_ticket_ids = set(group_df['ticket_id'])

        # Apply function and create found and not found columns
        group_df[['found', 'not_found']] = group_df.apply(lambda row: pd.Series(SearchService().check_expected(row)), axis=1)

        # Keep only the necessary columns
        group_df = group_df[['expected_id', 'sentence', 'target', 'found', 'not_found']]

        # Group by target and aggregate the results
        df_grouped = group_df.groupby('target').agg({
            'expected_id': 'first',   # or ' '.join para concatenar
            'sentence': 'first',
            'found': 'first',
            'not_found': 'first'
        }).reset_index()

        # Add a column accuracy
        df_grouped['accuracy'] = df_grouped.apply(
            lambda row: (row['found'] / (row['found'] + row['not_found'])) if (row['found'] + row['not_found']) > 0 else 0, 
            axis=1
        )

        return df_grouped
