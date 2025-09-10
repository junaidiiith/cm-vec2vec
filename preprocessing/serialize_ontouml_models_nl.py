import os
import json
import csv
import ast
from collections import defaultdict

def sanitize_text(text):
    """Remove newlines and extra spaces from a text value."""
    if isinstance(text, str):
        return text.replace("\n", " ").strip()
    return text

def read_json_file(json_path):
    """Attempt to load a JSON file using UTF-8 and fallback to Latin-1 if needed."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            with open(json_path, 'r', encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError as e:
            print(f"Error decoding file {json_path} with encoding {enc}: {e}")
        except Exception as e:
            print(f"Error loading JSON file {json_path}: {e}")
            break
    print(f"Unable to decode file {json_path} with available encodings.")
    return {"model": {"contents": []}}

def get_term_display(term):
    """Get a display string for a term with name and id."""
    if not isinstance(term, dict):
        return "Unknown"
    
    raw_name = term.get('name')
    term_id = term.get('id', 'Unknown')
    if raw_name is None or raw_name.strip() == "":
        return sanitize_text(term_id)
    else:
        return f"{sanitize_text(raw_name)}"

def serialize_property(prop, parent_term, property_type_map, relationship_map):
    """Serialize a property dictionary into descriptive sentences."""
    sentences = []
    
    # Get the property's name and id for display
    prop_display = get_term_display(prop)
    
    # Handle cardinality
    cardinality = prop.get('cardinality')
    
    # Get property type if available
    prop_type_info = prop.get('propertyType')
    if prop_type_info is not None and isinstance(prop_type_info, dict):
        prop_type_id = prop_type_info.get('id')
        
        # Convert ID to string to ensure it's hashable
        if not isinstance(prop_type_id, str):
            prop_type_id = str(prop_type_id)
            
        prop_type = property_type_map.get(prop_type_id, "Unknown type")
        
        if prop_type:
            # Add property relationship
            if cardinality:
                sentences.append(f"The {parent_term} has {cardinality} {prop_display} of type {prop_type}.")
            else:
                sentences.append(f"The {parent_term} has {prop_display} of type {prop_type}.")
    
    # Add stereotypes if present
    stereotype = prop.get('stereotype')
    if stereotype:
        sentences.append(f"The property {prop_display} has stereotype {stereotype}.")
    
    # Add special flags
    if prop.get('isDerived') is True:
        sentences.append(f"The property {prop_display} is derived.")
    if prop.get('isReadOnly') is True:
        sentences.append(f"The property {prop_display} is read-only.")
    if prop.get('isOrdered') is True:
        sentences.append(f"The property {prop_display} is ordered.")
    
    # Add aggregation kind if present
    aggregation_kind = prop.get('aggregationKind')
    if aggregation_kind and aggregation_kind != "NONE":
        sentences.append(f"The property {prop_display} has aggregation kind {aggregation_kind}.")
    
    return sentences

def serialize_class(term, property_type_map, relationship_map):
    """Serialize a class term into descriptive sentences."""
    sentences = []
    
    # Get class name and type
    class_name = get_term_display(term)
    class_type = term.get('type')
    stereotype = term.get('stereotype')
    
    # Basic class description
    if stereotype:
        sentences.append(f"{class_name} is a {stereotype} class.")
    else:
        sentences.append(f"{class_name} is a class.")
    
    # Add special attributes
    if term.get('isAbstract') is True:
        sentences.append(f"{class_name} is an abstract class.")
    if term.get('isDerived') is True:
        sentences.append(f"{class_name} is a derived class.")
    
    # Add restrictions if present
    restrictions = term.get('restrictedTo')
    if restrictions and isinstance(restrictions, list) and len(restrictions) > 0:
        restriction_str = ", ".join(restrictions)
        sentences.append(f"{class_name} is restricted to {restriction_str}.")
    
    return sentences

def serialize_relation(relation, property_type_map, relationship_map):
    """Serialize a relation into descriptive sentences."""
    sentences = []
    
    relation_name = get_term_display(relation)
    if not relation_name or relation_name.strip() == "":
        relation_name = "Relationship"
    
    stereotype = relation.get('stereotype')
    properties = relation.get('properties', [])
    
    # Skip if no properties
    if not properties or not isinstance(properties, list) or len(properties) < 2:
        return sentences
    
    # Get source and target
    source_prop = properties[0]
    target_prop = properties[1]
    
    source_cardinality = source_prop.get('cardinality', '?')
    target_cardinality = target_prop.get('cardinality', '?')
    
    source_type_info = source_prop.get('propertyType')
    target_type_info = target_prop.get('propertyType')
    
    if source_type_info and target_type_info:
        source_id = source_type_info.get('id')
        target_id = target_type_info.get('id')
        
        # Convert IDs to strings to ensure they're hashable
        if not isinstance(source_id, str):
            source_id = str(source_id)
        if not isinstance(target_id, str):
            target_id = str(target_id)
            
        source_name = property_type_map.get(source_id, 'Unknown')
        target_name = property_type_map.get(target_id, 'Unknown')
        
        # Describe relation based on stereotype
        if stereotype:
            if stereotype == "material":
                sentences.append(f"There is a material relationship '{relation_name}' from {source_name} to {target_name} with cardinality {source_cardinality}:{target_cardinality}.")
            elif stereotype == "derivation":
                sentences.append(f"There is a derivation relationship '{relation_name}' from {source_name} to {target_name}.")
            elif stereotype == "characterization":
                sentences.append(f"The {source_name} characterizes {target_name} with cardinality {source_cardinality}:{target_cardinality}.")
            elif stereotype == "mediation":
                sentences.append(f"The {source_name} mediates {target_name} with cardinality {source_cardinality}:{target_cardinality}.")
            elif stereotype == "formal":
                sentences.append(f"There is a formal relationship '{relation_name}' from {source_name} to {target_name} with cardinality {source_cardinality}:{target_cardinality}.")
            else:
                sentences.append(f"There is a {stereotype} relationship '{relation_name}' from {source_name} to {target_name} with cardinality {source_cardinality}:{target_cardinality}.")
        else:
            sentences.append(f"There is a relationship '{relation_name}' from {source_name} to {target_name} with cardinality {source_cardinality}:{target_cardinality}.")
    
    return sentences

def serialize_generalization(gen, property_type_map):
    """Serialize a generalization relationship."""
    sentences = []
    
    # Get general and specific elements
    general_id = gen.get('general')
    specific_id = gen.get('specific')
    
    if general_id and specific_id:
        # Convert IDs to strings to ensure they're hashable
        if not isinstance(general_id, str):
            general_id = str(general_id)
        if not isinstance(specific_id, str):
            specific_id = str(specific_id)
            
        general_name = property_type_map.get(general_id, 'Unknown')
        specific_name = property_type_map.get(specific_id, 'Unknown')
        
        sentences.append(f"{specific_name} is a type of {general_name}.")
    
    return sentences

def process_ontology_file(json_path):
    """Open the JSON file and serialize the model's contents."""
    data = read_json_file(json_path)
    
    # Extract the model information
    model = data.get('model', {})
    
    # Maps to keep track of elements by ID for cross-references
    property_type_map = {}  # Maps ID to element name
    relationship_map = defaultdict(list)  # Maps ID to relationships
    
    # First pass: build maps of all classes, properties, etc.
    def build_maps(contents, parent=None):
        if not contents or not isinstance(contents, list):
            return
        
        for item in contents:
            if not isinstance(item, dict):
                continue
                
            item_id = item.get('id')
            item_name = get_term_display(item)
            item_type = item.get('type')
            
            # Add to property type map
            if item_id and item_name:
                property_type_map[item_id] = item_name
            
            # Recursively process contents
            if 'contents' in item and isinstance(item.get('contents'), list):
                build_maps(item.get('contents'), item)
    
    # Process model contents to build maps
    if 'contents' in model and isinstance(model.get('contents'), list):
        build_maps(model.get('contents'))
    
    # Second pass: create natural language description
    serialization_sentences = []
    
    # Model metadata
    model_name = get_term_display(model)
    serialization_sentences.append(f"This is an OntoUML model named '{model_name}'.")
    
    # Process all elements
    def process_elements(contents, parent=None):
        if not contents or not isinstance(contents, list):
            return []
        
        sentences = []
        
        # First process classes
        for item in contents:
            if not isinstance(item, dict):
                continue
                
            item_type = item.get('type')
            
            if item_type == 'Class':
                sentences.extend(serialize_class(item, property_type_map, relationship_map))
            
            # Recursively process contents
            if 'contents' in item and isinstance(item.get('contents'), list):
                sentences.extend(process_elements(item.get('contents'), item))
        
        # Then process relationships and generalizations
        for item in contents:
            if not isinstance(item, dict):
                continue
                
            item_type = item.get('type')
            
            if item_type == 'Relation':
                sentences.extend(serialize_relation(item, property_type_map, relationship_map))
            elif item_type == 'Generalization':
                sentences.extend(serialize_generalization(item, property_type_map))
            
            # Process properties of classes
            if item_type == 'Class' and 'properties' in item and item.get('properties') and isinstance(item.get('properties'), list):
                class_name = get_term_display(item)
                for prop in item.get('properties'):
                    if isinstance(prop, dict):
                        sentences.extend(serialize_property(prop, class_name, property_type_map, relationship_map))
        
        return sentences
    
    # Process model contents
    if 'contents' in model and isinstance(model.get('contents'), list):
        for package in model.get('contents'):
            if isinstance(package, dict) and 'contents' in package and isinstance(package.get('contents'), list):
                serialization_sentences.extend(process_elements(package.get('contents')))
    
    # Combine all sentences into one text
    return " ".join(serialization_sentences)

def serialize_row(key, row):
    """Serialize metadata from a CSV row about the model."""
    if not isinstance(row, dict):
        return f"The OntoUML model (key: {key}) description is not available."
        
    title = row.get('title', '')
    keywords = " with key terms " + row.get('keywords', '') + ". " if row.get('keywords') else ""
    theme = row.get('theme', 'unspecified theme')
    
    # Handle list fields with proper error checking
    def parse_list_field(field):
        if not field:
            return []
        try:
            return ast.literal_eval(field)
        except (SyntaxError, ValueError):
            return [field]  # Return as single item list if parsing fails
    
    ontology_type = ', '.join(parse_list_field(row.get('ontologyType'))) if row.get('ontologyType') else "unspecified ontology type"
    designed_for = ', '.join(parse_list_field(row.get('designedForTask'))) if row.get('designedForTask') else "no specific task"
    language = row.get('language', 'unspecified language')
    context = ', '.join(parse_list_field(row.get('context'))) if row.get('context') else "no specific context"
    
    sentence = (f'The OntoUML model "{title}" (key: {key}) '
                f'is categorized under "{theme}". It is designed for the task(s) of {designed_for}, '
                f'and represents the ontology type(s): {ontology_type}. '
                f'The language of the model is "{language}", created in the context of {context}'
                f'{keywords}. Now we describe its terms. ')
    
    return sentence

def main():
    # Paths setup
    base_dir = os.path.join("datasets", "ontouml-models", "models")
    output_csv = "ontouml_models_natural_language_serializations.csv"
    
    # Check if metadata CSV exists
    metadata_path = 'ontouml_models.csv'
    meta_serializations = {}
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    key = row['key']
                    meta_serializations[key] = serialize_row(key, row)
        except Exception as e:
            print(f"Error reading metadata CSV: {e}")
            # Continue without metadata
    
    rows = []
    
    # Iterate over each folder in the base directory
    print(f"Processing models from: {base_dir}")
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            json_file = os.path.join(folder_path, "ontology.json")
            if os.path.exists(json_file):
                print(f"Processing: {folder}")
                try:
                    # Get metadata serialization if available
                    meta_text = meta_serializations.get(folder, "")
                    
                    # Process the actual model
                    model_text = process_ontology_file(json_file)
                    
                    # Combine metadata and model serialization
                    serialized_text = meta_text + model_text
                    
                    # Append the folder name and its serialization text
                    rows.append([folder, serialized_text])
                except Exception as e:
                    print(f"Error processing {folder}: {e}")
                    rows.append([folder, f"Error processing file: {e}"])
            else:
                print(f"No ontology.json found in {folder}")
                rows.append([folder, "ontology.json not found."])
    
    # Write the output CSV with UTF-8 encoding
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key", "serialization"])
        writer.writerows(rows)
    
    print(f"CSV file '{output_csv}' created with {len(rows)} rows.")

if __name__ == "__main__":
    main()
