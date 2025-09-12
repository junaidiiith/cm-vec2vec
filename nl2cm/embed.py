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
from typing import List
from nl2cm.utils import get_device


load_dotenv()


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
    model_kwargs = {'device': get_device()}
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


def extract_embddings_from_df(fp, nl_cm_cols) -> dict:
    nl_col, cm_col = nl_cm_cols
    with open(os.path.join(fp), "rb") as f:
        df = pickle.load(f)
    
    assert nl_col in df.columns, f"NL column {nl_col} not found in {fp}"
    assert cm_col in df.columns, f"CM column {cm_col} not found in {fp}"

    nl_null_idx = df.loc[df[nl_col].isnull()].index
    cm_null_idx = df.loc[df[cm_col].isnull()].index
    null_idx = list(set(nl_null_idx).union(set(cm_null_idx)))
    
    nl_emb = np.stack(df[~df.index.isin(null_idx)][nl_col].values)
    cm_emb = np.stack(df[~df.index.isin(null_idx)][cm_col].values)
    return {
        'nl_emb': nl_emb,
        'cm_emb': cm_emb,
        'total_count': len(df),
        'total_nl_count': len(nl_null_idx),
        'total_cm_count': len(cm_null_idx),
        'total_null_count': len(null_idx)
    }


def get_embeddings(data_path, nl_cm_cols, limit=None) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(data_path, str), f"Data path must be a string, got {type(data_path)}"
    assert isinstance(nl_cm_cols, list), f"NL and CM columns must be a list, got {type(nl_cm_cols)}"
    assert len(nl_cm_cols) == 2, f"NL and CM columns must be a list of length 2, got {len(nl_cm_cols)}"
    
    
    total_count = dict()
    total_nl_count = dict()
    total_cm_count = dict()
    total_null_count = dict()
    NLT_EMB, CM_EMB = np.array([]), np.array([])
    limit = limit if limit is not None else len([f for f in os.listdir(data_path) if f.endswith(".pkl")])
    for file in tqdm([f for f in os.listdir(data_path) if f.endswith(".pkl")][:limit]):
        # print(f"Processing file: {file}")
                
        response = extract_embddings_from_df(
            os.path.join(data_path, file), 
            nl_cm_cols
        )

        if NLT_EMB.size == 0:
            NLT_EMB = response['nl_emb']
            CM_EMB = response['cm_emb']
        else:
            NLT_EMB = np.concatenate([NLT_EMB, response['nl_emb']])
            CM_EMB = np.concatenate([CM_EMB, response['cm_emb']])
        
        total_count[file] = response['total_count']
        total_nl_count[file] = response['total_nl_count']
        total_cm_count[file] = response['total_cm_count']
        total_null_count[file] = response['total_null_count']
    
    print(f"Total rows: {sum(total_count.values())}")
    print(f"Total null NL rows: {sum(total_nl_count.values())}: {sum(total_nl_count.values()) / sum(total_count.values()) * 100}%")
    print(f"Total null CM rows: {sum(total_cm_count.values())}: {sum(total_cm_count.values()) / sum(total_count.values()) * 100}%")
    print(f"Total null rows: {sum(total_null_count.values())}: {sum(total_null_count.values()) / sum(total_count.values()) * 100}%")
    
    print("Size of NLT_EMB: ", NLT_EMB.shape)
    print("Size of CM_EMB: ", CM_EMB.shape)
     
    return NLT_EMB, CM_EMB
