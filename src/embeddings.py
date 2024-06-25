import re
import os
import math
import numpy as np
import pandas as pd
import hashlib
import time
from joblib import Parallel, delayed
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from openai import RateLimitError
from database_service import DatabaseService
from load_data_service import LoadData
from tqdm import tqdm


def extract_topic_between_asterisks(text):
    pattern = r'\*\*(.*?)\*\*'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return text

def get_ticket_summary(ticket: str, chat_llm):
    docs_list = [Document(page_content=str(ticket))]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    prompt_template = (
        "A seguinte passagem representa uma conversa entre um cliente e um agente de suporte técnico:"
        "---------------------\n"
        "{context}\n"
        "---------------------\n"
        "Dada a conversa forneça um tópico central e único do problema que foi discutido"
        "O tópico deve ser uma frase única.\n"
        "Desconsidere nomes, URLs, anexos, datas.\n"
        "Aqui estão alguns exemplos de tópicos:\n"
        "**Problema para estornar e substituir Nota Fiscal de Serviço (NFS) gerada no sistema TMS.**\n"
        "**Impossibilidade de gerar RPS de Substituição diretamente pelo módulo TMS**\n"
        "**Atualização do sistema Logix para compatibilidade com o Layout 18 do SPED Fiscal**\n"
        "**Descontinuação de ferramentas e relatórios desenvolvidos em Delphi e a continuidade das versões em .NET**"
    )
    template = PromptTemplate(input_variables=["context"], template=prompt_template)
    qa_chain = load_qa_chain(chat_llm, prompt=template)
    context = {'input_documents': doc_splits}
    output = qa_chain(context)
    output = output.get('output_text')
    out = extract_topic_between_asterisks(output)
    return out if out else output

def clean_topics_output(text):
    cleaned_text = re.sub(r'\d+\.\s*', '', text)
    return cleaned_text.split('\n')

def get_ticket_topics(ticket: str, chat_llm):
    docs_list = [Document(page_content=str(ticket))]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    prompt_template = (
        "A seguinte passagem representa uma conversa entre um cliente e um agente de suporte técnico:"
        "---------------------\n"
        "{context}\n"
        "---------------------\n"
        "Dada a conversa forneça até 5 tópicos centrais dos problemas que foram discutidos.\n"
        "Cada tópico deve ser uma frase única.\n"
        "Desconsidere nomes, URLs, anexos, datas. Caso haja códigos de erro mencionar no tópico.\n"
        "Desconsidere sugestões de abertura de solicitação, solicitação de melhoria.\n"
        "Aqui estão alguns exemplos de tópicos:\n"
        "**Problema para estornar e substituir Nota Fiscal de Serviço (NFS) gerada no sistema TMS.**\n"
        "**Impossibilidade de gerar RPS de Substituição diretamente pelo módulo TMS**\n"
        "**Atualização do sistema Logix para compatibilidade com o Layout 18 do SPED Fiscal**\n"
        "**Descontinuação de ferramentas e relatórios desenvolvidos em Delphi e a continuidade das versões em .NET**"
    )
    template = PromptTemplate(input_variables=["context"], template=prompt_template)
    qa_chain = load_qa_chain(chat_llm, prompt=template)
    context = {'input_documents': doc_splits}
    output = qa_chain(context)
    output = output.get('output_text')
    out = clean_topics_output(output)
    return out if out else output

def clean_output(text):
    cleaned_text = re.sub(r'\d+\.\s*', '', text)
    return ' '.join(cleaned_text.split('\n'))

def get_ticket_keywords(ticket: str, chat_llm):
    docs_list = [Document(page_content=str(ticket))]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    prompt_template = prompt_template = (
        "A seguinte passagem representa uma conversa entre um cliente e um agente de suporte técnico:"
        "---------------------\n"
        "{context}\n"
        "---------------------\n"
        "Dada a conversa forneça 8 palavras chaves principais que descrevam o problema que foi discutido"
        "Desconsidere nomes, URLs, anexos, datas.\n"
    )
    template = PromptTemplate(input_variables=["context"], template=prompt_template)
    qa_chain = load_qa_chain(chat_llm, prompt=template)
    context = {'input_documents': doc_splits}
    output = qa_chain(context)
    output = output.get('output_text')
    return clean_output(output)

def get_sentence_hash(ticket_id, sentence):
        hash_concat = str(ticket_id) + sentence
        return hashlib.md5(hash_concat.encode('utf-8')).hexdigest()

def load_data(tickets_in_db: list[int]):

    tickets_statement = ''
    if tickets_in_db:
        tickets_statement = f"WHERE ticket_id not in {tuple(tickets_in_db)}"

    select_statement = (f"""
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
                             {tickets_statement}
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
    # Faz a leitura dos dados no BQ
    return LoadData.run_select_bigquery(select_statement).to_dataframe()     

def create_embeddings(df, embedding_llm, chat_llm):

    df_summary = df.copy()
    df_keywords = df.copy()
    df_topics = df.copy()

    df_summary['sentence'] = df_summary['ticket_comment'].apply(lambda t: get_ticket_summary(t, chat_llm))
    df_summary['sentence_source'] = 'summary'

    df_keywords['sentence'] = df_keywords['ticket_comment'].apply(lambda t: get_ticket_keywords(t, chat_llm))
    df_keywords['sentence_source'] = 'keywords'

    df_subject = df.copy()

    df_subject['sentence'] = df_subject['subject']
    df_subject['sentence_source'] = 'subject'

    df_topics['sentence'] = df_topics['ticket_comment'].apply(lambda t: get_ticket_topics(t, chat_llm))
    df_topics = df_topics.explode('sentence')
    print(f'Quantidade de tópicos após explode: {df_topics.shape[0]}')
    df_topics['sentence'] = df_topics['sentence'].apply(lambda x: x.replace('**', ''))
    df_topics = df_topics.reset_index(drop=True)
    df_topics['sentence_source'] = 'topic'

    dff = pd.concat([df_summary, df_keywords, df_subject, df_topics])
    print(f'Quantidade de sentenças após concat: {dff.shape[0]}')

    sentences_pending = dff["sentence"].unique()
    print(f'Criando os embeddings para {len(sentences_pending)} novas sentenças.')
    embeddings_pending = embedding_llm.embed_documents(sentences_pending)
    sentence_to_embedding = dict(zip(sentences_pending, embeddings_pending))

    dff['sentence_embedding'] = dff["sentence"].apply(lambda x: sentence_to_embedding[x])
    dff['ticket_sentence_hash'] = dff.apply(lambda row: get_sentence_hash(row['ticket_id'], row['sentence']), axis=1)

    print(f'Quantidade de sentenças após criação dos embeddings: {dff.shape[0]}')

    dff = dff.drop_duplicates(['ticket_sentence_hash'])
    print(f'Quantidade de sentenças após drop_duplicates: {dff.shape[0]}')

    data_list = [(row['ticket_id'], 
                row['ticket_comment'], 
                row['ticket_sentence_hash'],
                row['module'], 
                row['product'],
                row['sentence_source'],
                row['ticket_status'],
                row['sentence_embedding'], 
                row['created_at'], 
                row['updated_at'], 
                row['sentence']) for _, row in dff.iterrows()]

    DatabaseService().run_insert_sentence_embeddings_statement('tickets_embeddings_test', data_list)


def split_document(document):
    docs_list = [Document(page_content=str(document))]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return [doc.page_content for doc in doc_splits]


def create_chunk_embeddings(df, embedding_llm):
    df['ticket_comment'] = df['ticket_comment'].str.replace(r'\x00', "")
    df['chunk'] = df['ticket_comment'].apply(split_document)
    df = df.explode('chunk', ignore_index=True)
    sentences_pending = df["chunk"].unique()
    print(f'Criando os embeddings para {len(sentences_pending)} novas sentenças.')
    rate_limit_total_retries = 5
    # TODO: Implement a better way to handle rate limit
    while rate_limit_total_retries > 0:
        try:
            embeddings_pending = embedding_llm.embed_documents(sentences_pending)
            rate_limit_total_retries = 0
        except RateLimitError:
            rate_limit_total_retries -= 1
            print(f'Rate limit reached. Waiting 3 minutes. Retry number: {5 - rate_limit_total_retries} of 5.')
            time.sleep(180)
    sentence_to_embedding = dict(zip(sentences_pending, embeddings_pending))
    df['sentence_embedding'] = df["chunk"].apply(lambda x: sentence_to_embedding[x])
    df['ticket_chunk_hash'] = df.apply(lambda row: get_sentence_hash(row['ticket_id'], row['chunk']), axis=1)
    data_list = [(row['ticket_id'], 
              row['ticket_comment'], 
              row['ticket_chunk_hash'],
              row['module'], 
              row['product'],
              row['ticket_status'],
              row['sentence_embedding'], 
              row['created_at'], 
              row['updated_at'], 
              row['chunk'],
              row['subject']) for _, row in df.iterrows()]
    try:
        DatabaseService().run_insert_chunks_embeddings_statement('tickets_embeddings_chunks', data_list)
    except Exception as e:
        print(f'Error inserting chunks into database. {e}. Retrying inserting one by one.')
        for _, row in df.iterrows():
            ticket_id = row['ticket_id']
            data_list = [(ticket_id,  row['ticket_comment'], row['ticket_chunk_hash'],
                          row['module'],  row['product'], row['ticket_status'], row['sentence_embedding'], 
                          row['created_at'],  row['updated_at'], row['chunk'], row['subject'])]
            try:
                DatabaseService().run_insert_chunks_embeddings_statement('tickets_embeddings_chunks', data_list)
            except Exception as e:
                print(f'Error inserting ticket {ticket_id} into database. {e}. Skipping this chunk.')
                pass


def create_embeddings_and_sent_to_db(table_destination: str):
    tickets_in_db = DatabaseService().run_select_statement(f'SELECT distinct(ticket_id) FROM {table_destination}')
    tickets_in_db = [list(d.values())[0] for d in tickets_in_db]
    print(f'There are {len(tickets_in_db)} tickets in the database. Table: {table_destination}')
    df = load_data(tickets_in_db)
    print(f'Quantidade de tickets: {df.shape[0]}')
    chunk_size = 50
    df_chunks = np.array_split(df, math.ceil(len(df.index) / chunk_size))
    embedding_llm = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base='https://proxy.dta.totvs.ai/',
            openai_api_key=os.getenv("DTA_PROXY_SECRET_KEY")
        )
    if table_destination == 'tickets_embeddings_chunks':
        print(f'Starting chunks embeddings')
        Parallel(n_jobs=-1, backend='threading')(delayed(create_chunk_embeddings)(df, embedding_llm)
                for df in tqdm(df_chunks))
    else:
        print(f'Starting sentence embeddings')
        chat_llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-2024-05-13",
            openai_api_base='https://proxy.dta.totvs.ai/',
            openai_api_key=os.getenv("DTA_PROXY_SECRET_KEY")
        )
        Parallel(n_jobs=-1, backend='threading')(delayed(create_embeddings)(df, embedding_llm, chat_llm)
                for df in tqdm(df_chunks))
