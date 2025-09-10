import pickle
import pandas as pd
import numpy as np
import os
import tiktoken
from tqdm.auto import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import torch
from typing import List


load_dotenv()


def cuda_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def chunk_text(text, encoding_name="cl100k_base", chunk_size=8192, chunk_overlap=0):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk))
        if i + chunk_size >= len(tokens):
            break
    return chunks


def get_langchain_embedding(text, emb_type = 'huggingface', chunk_size=8192, encoding_name="cl100k_base", aggregator="mean"):
    model_kwargs = {'device': cuda_device()}
    encode_kwargs = {'normalize_embeddings': True}
    
    chunks = chunk_text(text, encoding_name=encoding_name, chunk_size=chunk_size)
    if emb_type == 'openai':
        emb = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_LARGE", "text-embedding-3-small")
        )  # or HuggingFaceEmbeddings(...)
    elif emb_type == 'huggingface':

        emb = HuggingFaceEmbeddings(
            model_name=os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
            model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    elif emb_type == 'huggingface-instruct':
        emb = HuggingFaceEmbeddings(
            model_name=os.getenv("HUGGINGFACE_EMBEDDING_INSTRUCT_MODEL", "hkunlp/instructor-large"),
            model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    else:
        raise ValueError(f"Invalid embedding type: {emb_type}")
    
    chunk_vecs = emb.embed_documents(chunks)  # List[List[float]] ; len == len(chunks)
    if aggregator == "mean":
        doc_vec = np.mean(np.array(chunk_vecs), axis=0)  # simple, strong baseline
    elif aggregator == "max":
        doc_vec = np.max(np.array(chunk_vecs), axis=0)
    elif aggregator == "min":
        doc_vec = np.min(np.array(chunk_vecs), axis=0)
    elif aggregator == "sum":
        doc_vec = np.sum(np.array(chunk_vecs), axis=0)
    elif aggregator == "log":
        doc_vec = np.log1p(np.abs(np.array(chunk_vecs))).sum(axis=0)
    else:
        doc_vec = np.mean(np.array(chunk_vecs), axis=0)
    doc_vec = (doc_vec / np.linalg.norm(doc_vec))    # L2-normalize if you like
    return doc_vec



# def get_openai_embedding(text, chunk_size=8192, encoding_name="cl100k_base", aggregator="mean"):
#     if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
#         return None
#     try:
#         chunks = chunk_text(text, encoding_name=encoding_name, chunk_size=chunk_size)
#         embeddings = []
#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         for chunk in chunks:
#             try:
#                 response = client.embeddings.create(
#                     input=chunk,
#                     model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
#                 )
#             except Exception as e:
#                 print(f"OpenAI API error for chunk: {str(chunk)[:30]}...: {e}")
#                 raise e
#             embeddings.append(np.array(response.data[0].embedding))
#         if not embeddings:
#             return None
#         if aggregator == "mean":
#             return np.mean(embeddings, axis=0)
#         elif aggregator == "max":
#             return np.max(embeddings, axis=0)
#         elif aggregator == "min":
#             return np.min(embeddings, axis=0)
#         elif aggregator == "sum":
#             return np.sum(embeddings, axis=0)
#         elif aggregator == "log":
#             return np.log1p(np.abs(embeddings)).sum(axis=0)
#         else:
#             return np.mean(embeddings, axis=0)
#     except Exception as e:
#         print(f"Embedding error for text: {str(text)[:30]}...: {e}")
#         return None

def process_column_sequential(texts: List[str], emb_type='huggingface', chunk_size=8192, encoding_name="cl100k_base", aggregator="mean"):
    return [get_langchain_embedding(text, emb_type, chunk_size, encoding_name, aggregator) for text in tqdm(texts, desc="Getting embeddings sequentially")]


def process_column_parallel(
    texts: List[str], emb_type='huggingface', num_jobs=4, 
    chunk_size=8192, encoding_name="cl100k_base", 
    aggregator="mean"
):
    
    embeddings = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        future_to_index = {executor.submit(get_langchain_embedding, text, emb_type, chunk_size, encoding_name, aggregator): idx for idx, text in enumerate(texts)}
        for future in tqdm(as_completed(future_to_index), total=len(texts), desc="Getting embeddings in parallel"):
            idx = future_to_index[future]
            try:
                embeddings[idx] = future.result()
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                embeddings[idx] = None
    return embeddings


def add_embeddings(
    df: pd.DataFrame, 
    embedding_columns: list[str], 
    emb_type='huggingface',
    save=True, num_jobs=4, 
    chunk_size=8192, 
    encoding_name="cl100k_base", 
    aggregator="mean", 
    output_pth=None,
    parallel=False
):
    assert all(ec in df.columns for ec in embedding_columns), f"Some Columns from '{embedding_columns}' not found in DataFrame"
    if output_pth and os.path.exists(output_pth):
        df = pd.read_pickle(output_pth)
    
    kwargs = dict(
        emb_type=emb_type,
        chunk_size=chunk_size, 
        encoding_name=encoding_name, 
        aggregator=aggregator
    )
    
    if parallel:
        process_column_func = process_column_parallel
    else:
        process_column_func = process_column_sequential
    
    
    for col in embedding_columns:
        print(f"Processing column: {col}")
        kwargs['texts'] = df[col].tolist()
        if parallel:
            kwargs['num_jobs'] = num_jobs
        embed_col_name = f"{col}_Emb_{emb_type}"
        if embed_col_name in df.columns:
            continue
        df[embed_col_name] = process_column_func(**kwargs)
            
    if save:
        with open(f"{output_pth if output_pth else 'embeddings_df.pkl'}", 'wb') as f:
            pickle.dump(df, f)
            