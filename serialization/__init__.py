from typing import Union, Literal
from serialization.ontouml import serialize_ontouml_model, set_ontouml_schema, get_meta_info
from serialization.archi import serialize_archimate_model
from serialization.bpmn import serialize_bpmn_model
from .utils import read_json_file
import os
import csv
import json
from tqdm.auto import tqdm
import pandas as pd


def get_schema(models_dir, stype=Union[Literal['ontouml']]):
    schema = {}
    if stype == 'ontouml':
        set_ontouml_schema(models_dir, schema)
    
    return schema
    
    
def serialize_model(model_file, model_type, stype=Union[Literal['cm', 'nl']], use_structure=True):
    
    if model_type == 'ontouml':
        assert os.path.exists(model_file), f"Model file not found: {model_file}"
        data = read_json_file(model_file)
        return serialize_ontouml_model(data, stype=stype, level=0, use_structure=use_structure)
    elif model_type == 'archimate':
        assert os.path.exists(model_file), f"Model file not found: {model_file}"
        model = read_json_file(model_file)
        return serialize_archimate_model(model, stype=stype, level=0, use_structure=use_structure)
    elif model_type == 'bpmn':
        return serialize_bpmn_model(model_file, stype=stype, level=0, use_structure=use_structure)
    
    raise ValueError(f"Unsupported model type: {model_type}")


def serialize_ontouml_models(models_dir, models_csv, output_csv):
    rows = []
    meta_serializations = {}

    with open(models_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row['key']
            meta_serializations[key] = get_meta_info(row)
    
    # Iterate over each folder in the base directory.
    for folder in tqdm(os.listdir(models_dir), desc="Processing models"):
        folder_path = os.path.join(models_dir, folder)
        if os.path.isdir(folder_path):
            json_file = os.path.join(folder_path, "ontology.json")
            if os.path.exists(json_file):
                nl_text = serialize_model(json_file, 'ontouml', stype='nl')
                cm_text = serialize_model(json_file, 'ontouml', stype='cm')
                rows.append([folder, f"{meta_serializations[folder]}", nl_text, cm_text])
   
    # Write the output CSV with UTF-8 encoding.
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key", "Metadata", "NL_Serialization", "CM_Serialization"])
        writer.writerows(rows)
    
    print(f"CSV file '{output_csv}' created with {len(rows)} rows.")
    

def serialize_archimate_models(data_path, dataset_name='eamodelset'):
    models_data_path = os.path.join(data_path, "eamodelset", "processed-models")
    model_dirs = os.listdir(models_data_path)
    rows = list()
    for model_dir in tqdm(model_dirs, desc=f'Loading {dataset_name.title()}'):
        model_dir = os.path.join(models_data_path, model_dir)
        if os.path.isdir(model_dir):
            model_file = os.path.join(model_dir, 'model.json')
            if os.path.exists(model_file):
                model_nl_str = serialize_model(model_file, stype='nl', model_type='archimate')
                model_cm_str = serialize_model(model_file, stype='cm', model_type='archimate')
                rows.append([os.path.basename(model_dir), model_nl_str, model_cm_str])
    
    output_csv = os.path.join(data_path, f'{dataset_name}_serialized.csv')
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key", "NL_Serialization", "CM_Serialization"])
        writer.writerows(rows)


def serialize_bpmn_models(csv_chunks_dir):
    csv_chunk_paths = [os.path.join(csv_chunks_dir, f) for f in os.listdir(csv_chunks_dir) if f.endswith('.csv')]
    for csv in tqdm(csv_chunk_paths, desc="Processing BPMN CSV chunks"):
        df = pd.read_csv(csv)
        
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Serializing BPMN models"):
            model_json = json.loads(row['Model JSON'])
            try:
                model_nl_str = serialize_model(model_json, stype='nl', model_type='bpmn')
                model_cm_str = serialize_model(model_json, stype='cm', model_type='bpmn')
                df.at[i, 'NL_Serialization'] = model_nl_str
                df.at[i, 'CM_Serialization'] = model_cm_str
            except Exception as e:
                print(f"Error processing row {i} in {csv}: {e}")
                print(f"Model JSON: {json.dumps(model_json, indent=2)}")
                raise e
    
        df.to_csv(csv, index=False, encoding='utf-8')
            
        
__all__ = [
    "serialize_ontouml_models",
    "serialize_archimate_models",
    "serialize_bpmn_models",
]