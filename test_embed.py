import os
from tqdm.auto import tqdm
import pandas as pd
from embed import add_embeddings


def add_embeddings_bpmn(csv_files_dir='datasets/bpmn-serialized'):
    embed_type = 'openai'
    parallel = True
    
    embedding_columns = ['NL_Serialization', 'CM_Serialization']
    for csv_file in tqdm(os.listdir(csv_files_dir), desc="Processing BPMN CSV chunks"):
        print(f"Processing {csv_file}")
        add_embeddings(
            pd.read_csv(os.path.join(csv_files_dir, csv_file)), 
            embedding_columns=embedding_columns, 
            num_jobs=8, 
            output_pth=os.path.join(csv_files_dir, f"{csv_file.replace('.csv', '_embeddings.pkl')}"),
            parallel=parallel,
            emb_type=embed_type
        )


if __name__ == "__main__":
    add_embeddings_bpmn()