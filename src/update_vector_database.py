import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)


def run_insert_statement(connection_str: str, table_name: str, data_list: list, product: str):
    try:
        conn = psycopg2.connect(connection_str)
        register_vector(conn)
    except Exception as e:
        logger.error(f'Connecting to database failed. {e}')
        return
    try:
        cursor = conn.cursor()
        execute_values(cursor, f"""INSERT INTO {table_name} (id, sentence, country, database, homolog, html_url,
                               module, patch_url, patch_version, product, sanitized_solution, section_html_url, section_id,
                               section_name, segment, sentence_source, situacao_requisicao, solution, summary, tags, title,
                               sentence_embedding, updated_at) VALUES %s ON CONFLICT DO NOTHING""", data_list)
        conn.commit()
        logger.info(f'Sentences of product {product} added to dabatase successfully.')
    except Exception as e:
        logger.error(f'Inserting {len(data_list)} rows of product {product} into database failed. {e}')
        conn.rollback()


def run_delete_statement(connection_str: str, delete_statement: str):
    try:
        conn = psycopg2.connect(connection_str)
        register_vector(conn)
    except Exception as e:
        logger.error(f'Connecting to database failed. {e}')
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(delete_statement)
        conn.commit()
    except Exception as e:
        logger.error(f'Deleting records in database failed. {e}\n Delete statement: {delete_statement}')
        conn.rollback()


def run_update_statement(connection_str: str, update_statement: str):
    try:
        conn = psycopg2.connect(connection_str)
        register_vector(conn)
    except Exception as e:
        logger.error(f'Connecting to database failed. {e}')
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(update_statement)
        conn.commit()
    except Exception as e:
        logger.error(f'Updating records in database failed. {e}\n Update statement: {update_statement}')
        conn.rollback()


def update_vector_database(connection_str: str, table_name: str, updated_embeddings_df: pd.DataFrame):
    updated_embeddings_df = updated_embeddings_df.drop_duplicates(['id', 'sentence'])
    logger.info(f'{updated_embeddings_df.shape[0]} sentences to be added to the database.')
    products_df = [y for _, y in updated_embeddings_df.groupby('product')]
    for product_df in products_df:
        product = product_df['product'].iloc[0]
        data_list = [(row['id'], 
                      row['sentence'],
                      row['country'], 
                      row['database'], 
                      row['homolog'], 
                      row['html_url'], 
                      row['module'], 
                      row['patch_url'], 
                      row['patch_version'], 
                      row['product'], 
                      row['sanitized_solution'], 
                      row['section_html_url'], 
                      row['section_id'], 
                      row['section_name'], 
                      row['segment'], 
                      row['sentence_source'], 
                      row['situacao_requisicao'], 
                      row['solution'], 
                      row['summary'], 
                      row['tags'], 
                      row['title'],  
                      row['sentence_embedding'],
                      row['updated_at']) for _, row in product_df.iterrows()]
        run_insert_statement(connection_str, table_name, data_list, product)


def delete_removed_articles_from_database(connection_str: str, table_name: str, expected_ids: list[int]):
    expected_ids_str = ''
    for id in expected_ids:
        expected_ids_str += f'{id}, '

    delete_statement = f'''DELETE FROM public.{table_name}
                           WHERE id in ({expected_ids_str[:-2]});'''
    return run_delete_statement(connection_str, delete_statement)

def delete_old_articles_from_database(connection_str: str, table_name: str, expected_ids: list[int]):
    logger.info(f"Size of the expected IDs list: {len(expected_ids)}")
    expected_ids_str = ''
    for id in expected_ids:
        expected_ids_str += f'{id}, '

    delete_statement = f'''DELETE FROM public.{table_name}
                           WHERE id in ({expected_ids_str[:-2]});'''
    result = run_delete_statement(connection_str, delete_statement)

    return result


def delete_removed_sentences_from_database(connection_str: str, table_name: str, removed_sentences_df: pd.DataFrame):
    removed_sentences_df = removed_sentences_df[['id', 'sentence']]
    records = list(removed_sentences_df.to_records(index=False))
    sentences_tuples = ''
    for sentence_tuple in records:
        id, sentence = sentence_tuple
        sentences_tuples += f"(CAST({id} AS BIGINT), CAST('{sentence}' AS TEXT)), "

    delete_statement = f'''DELETE FROM public.{table_name}
                           WHERE (id, sentence) = ANY(Array [{sentences_tuples[:-2]}]);'''
    return run_delete_statement(connection_str, delete_statement)


def update_homolog_flag_in_database(connection_str: str, table_name: str, catalogue_df: pd.DataFrame):
    catalogue_df = catalogue_df.drop_duplicates(subset=['product', 'module'], keep='first')
    prod = catalogue_df[catalogue_df.homolog == False][['product', 'module']]
    homolog = catalogue_df[catalogue_df.homolog == True][['product', 'module']]

    if not prod.empty:
        records = list(prod.to_records(index=False))
        product_module_tuples = ''
        for product_module_tuple in records:
            product, module = product_module_tuple
            product_module_tuples += f"(CAST('{product}' AS TEXT), CAST('{module}' AS TEXT)), "

        update_statement = f'''UPDATE public.{table_name}
                            SET homolog = true
                            WHERE (product, module) = ANY(Array [{product_module_tuples[:-2]}])
                            AND homolog = false;'''
            
        run_update_statement(connection_str, update_statement)
        logger.info(f'Homolog flag has been set to true for {homolog.shape[0]} modules successfully.')
    else:
        logger.info(f'There are not products with the homolog flag set to False in the catalogue.')

    if not homolog.empty:
        records = list(homolog.to_records(index=False))
        product_module_tuples = ''
        for product_module_tuple in records:
            product, module = product_module_tuple
            product_module_tuples += f"(CAST('{product}' AS TEXT), CAST('{module}' AS TEXT)), "
        
        update_statement = f'''UPDATE public.{table_name}
                            SET homolog = false
                                WHERE (product, module) = ANY(Array [{product_module_tuples[:-2]}])
                            AND homolog = true;'''
        
        run_update_statement(connection_str, update_statement)
        logger.info(f'Homolog flag has been set to false for {prod.shape[0]} modules successfully.')
    else:
        logger.info(f'There are not products with the homolog flag set to True in the catalogue.')
