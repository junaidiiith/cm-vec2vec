import os
import json
import csv
import ast

def sanitize_text(text):
    """Remove newlines and extra spaces from a text value."""
    if isinstance(text, str):
        return text.replace("\n", " ").strip()
    return text

def read_json_file(json_path):
    """Attempt to load a JSON file using UTF-8 and fallback to Latin-1 if needed."""
    encodings = ['utf-8', 'latin-1']
    for enc in encodings:
        try:
            with open(json_path, 'r', encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError as e:
            print(f"Error decoding file {json_path} with encoding {enc}: {e}")
    raise UnicodeDecodeError(f"Unable to decode file {json_path} with available encodings.")

def serialize_property(prop, parent_term):
    """Serialize a property dictionary into descriptive sentences including its id."""
    sentences = []
    # Get the property's raw name and id.
    raw_name = prop.get('name')
    prop_id = prop.get('id', 'Unknown')
    if raw_name is None or raw_name.strip() == "":
        prop_display = sanitize_text(prop_id)
    else:
        prop_display = f"{sanitize_text(raw_name)} (id: {sanitize_text(prop_id)})"
    
    # Note the parent term for this property.
    sentences.append(f"Property {prop_display} belongs to {sanitize_text(parent_term)}.")
    
    return sentences

def serialize_term(term, parent_display=None):
    """Recursively serialize a term (and its nested contents) into descriptive sentences.
       Each term reference includes its id. If the term name is missing, only the id is used."""
    sentences = []
    # Get raw name and id.
    raw_name = term.get('name')
    term_id = term.get('id', 'Unknown')
    if raw_name is None or raw_name.strip() == "":
        term_display = sanitize_text(term_id)
    else:
        term_display = f"{sanitize_text(raw_name)} (id: {sanitize_text(term_id)})"
    
    # If the term has a parent, add a sentence for the relationship.
    if parent_display is not None:
        sentences.append(f"Term {term_display} belongs to {sanitize_text(parent_display)}.")
    
    
    # Serialize properties of the term.
    if 'properties' in term and isinstance(term['properties'], list):
        for prop in term['properties']:
            sentences.extend(serialize_property(prop, parent_term=term_display))
    
    # Process nested contents (if any) recursively.
    if 'contents' in term and isinstance(term['contents'], list):
        for child in term['contents']:
            sentences.extend(serialize_term(child, parent_display=term_display))
    
    return sentences

def process_ontology_file(json_path):
    """Open the JSON file and serialize the model's contents (including properties and term ids)."""
    data = read_json_file(json_path)
    
    model = data.get('model', {})
    serialization_sentences = []
    
    if 'contents' in model and isinstance(model['contents'], list):
        # Use the model's display: include name and id if available.
        raw_model_name = model.get('name')
        model_id = model.get('id', 'Unknown')
        if raw_model_name is None or raw_model_name.strip() == "":
            model_display = sanitize_text(model_id)
        else:
            model_display = f"{sanitize_text(raw_model_name)} (id: {sanitize_text(model_id)})"
        for term in model['contents']:
            serialization_sentences.extend(serialize_term(term, parent_display=model_display))
    else:
        serialization_sentences.append("No contents found in model.")
    
    # Combine all sentences into one text.
    return " ".join(serialization_sentences)


def serialize_row(key, row):
    title = row['title']
    keywords = " with key terms " + row['keywords'] + ". " if row['keywords'] else ""
    theme = row['theme']
    ontology_type = ', '.join(ast.literal_eval(row['ontologyType'])) if row['ontologyType'] else "unspecified ontology type"
    designed_for = ', '.join(ast.literal_eval(row['designedForTask'])) if row['designedForTask'] else "no specific task"
    language = row['language']
    context = ', '.join(ast.literal_eval(row['context'])) if row['context'] else "no specific context"
    sentence = (f'The OntoUML model "{title}" (key: {key}) '
                f'is categorized under "{theme}". It is designed for the task(s) of {designed_for}, '
                f'and represents the ontology type(s): {ontology_type}. '
                f'The language of the model is "{language}", created in the context of {context}'
                f'{keywords}. Now we describe its terms. ')
    
    return sentence


def main():
    base_dir = os.path.join("..", "datasets", "ontouml-models-master", "models")
    output_csv = "ontouml_models_nlt_serializations.csv"
    rows = []
    meta_serializations = {}

    with open('ontouml_models.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row['key']
            meta_serializations[key] = serialize_row(key, row)
    
    # Iterate over each folder in the base directory.
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            json_file = os.path.join(folder_path, "ontology.json")
            if os.path.exists(json_file):
                try:
                    serialized_text = process_ontology_file(json_file)
                except Exception as e:
                    serialized_text = f"Error processing file: {e}"
            else:
                serialized_text = "ontology.json not found."
            
            # Append the folder name and its serialization text.
            rows.append([folder, meta_serializations[folder] + serialized_text])
    
    # Write the output CSV with UTF-8 encoding.
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key", "serialization"])
        writer.writerows(rows)
    
    print(f"CSV file '{output_csv}' created with {len(rows)} rows.")



if __name__ == "__main__":
    main()