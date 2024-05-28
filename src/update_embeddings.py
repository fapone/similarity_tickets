import json
import logging
import pandas as pd
import json
import gc
import numpy as np
from dta_utils import DtaProxy
from langchain.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def update_embeddings(df, existing_documents_df, deleted_documents, dta_proxy_url, secret_key_dta,
                      cache=True, shrink_long_strings=5000):

    if df.empty:
        return df, df, None

    df = df[~df['id'].isin(deleted_documents)]
    # Filling NAs with blank
    logger.info('Replacing NaN with \"\" to avoid crashes on fulfillment.')
    df.fillna('', inplace=True)

    # Converting tags to list
    aux = df.applymap(lambda x: isinstance(x, list)).all()
    list_columns = aux.index[aux].tolist()
    if 'tags' in df.columns and 'tags' not in list_columns:
        df['tags'] = df['tags'].apply(get_tags)

    df.dropna(subset=['sentence'], inplace=True)

    logger.info(f'There are {df.shape[0]} sentences to be processed.')

    if cache:
        merged = df.merge(existing_documents_df, how='outer', indicator=True, on=['id', 'sentence'])
        new_sentences_only = merged[merged['_merge']=='left_only']
        logger.info(f'There are {new_sentences_only.shape[0]} new sentences to be embedded.')
        removed_sentences_only = merged[merged['_merge']=='right_only'][['id', 'sentence']]
        logger.info(f'There are {removed_sentences_only.shape[0]} sentences to be removed.')
        removed_sentences_only = removed_sentences_only[['id', 'sentence']]
        sentences_remained_same = merged[merged['_merge']=='both']
        logger.info(f'There are {sentences_remained_same.shape[0]} sentences that remained the same.')
        del merged
        del sentences_remained_same
        gc.collect()
    else:
        new_sentences_only = df
        removed_sentences_only = pd.DataFrame(columns=df.columns)

    # Generating Embeddings
    logger.info('Creating embeddings')

    model = OpenAIEmbeddings(
        openai_api_base=dta_proxy_url,
        openai_api_key=secret_key_dta
    )

    df = new_sentences_only
    sentences_pending = df["sentence"].unique()
    logger.info(f'Calculating embeddings for {len(sentences_pending)} unprocessed sentences.')
    embeddings_pending = model.embed_documents(sentences_pending)
    sentence_to_embedding = dict(zip(sentences_pending, embeddings_pending))
    df["sentence_embedding"] = df["sentence"].apply(lambda x: sentence_to_embedding[x])

    logger.info('Embeddings successfully created')
    
    # If defined, shrinks long columns to the maximum length
    #if shrink_long_strings > 0:
    if shrink_long_strings is not None and shrink_long_strings > 0:
        for c in df.columns:
            if c == "sentence_embedding": continue
            
            try:
                l = df[c].str.len().max()
                
                if l > shrink_long_strings:
                    logger.info(f"Shrinking column \"{c}\" with max length {l} to {shrink_long_strings}.")
                    df[c] = df[c].str.slice(0,shrink_long_strings)
                
            except:
                pass

    updated_at_max = df['updated_at'].max()

    return df, removed_sentences_only, None if np.isnan(updated_at_max) else updated_at_max


def get_tags(label_names):
    if label_names:
        return json.loads(str(label_names))
    return list()