#!/usr/bin/env python3
import os
import json
import csv
import ast
from collections import defaultdict

def try_multiple_encodings(file_path):
    """
    Try to open a file with multiple encodings
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16', 'windows-1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                return content, encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error opening file {file_path}: {e}")
            return None, None
    
    print(f"Failed to decode file {file_path} with any encoding")
    return None, None

def load_model(file_path):
    """
    Load a model from a JSON file, trying multiple encodings
    """
    content, encoding = try_multiple_encodings(file_path)
    if content:
        try:
            # Try to parse the JSON content
            model = json.loads(content)
            return model
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from {file_path}: {e}")
    
    return None

def normalize_key(key):
    """
    Ensure keys are hashable (strings)
    """
    if isinstance(key, dict):
        # Convert dict to a string representation
        return str(key)
    return str(key)

def get_model_name(model):
    """
    Get the name of the model
    """
    if isinstance(model, dict):
        if 'name' in model:
            return model['name']
        if 'id' in model:
            return model['id']
    return "Unnamed Model"

def describe_class(class_obj):
    """
    Generate a natural language description of a class
    """
    if not isinstance(class_obj, dict):
        return ""
    
    name = class_obj.get('name', 'Unnamed Class')
    description = f"{name} is a class"
    
    # Add stereotype if available
    stereotype = class_obj.get('stereotype', None)
    if stereotype:
        if isinstance(stereotype, dict) and 'type' in stereotype:
            description += f" of type {stereotype['type']}"
        elif isinstance(stereotype, str):
            description += f" of type {stereotype}"
    
    # Add description if available
    if 'description' in class_obj and class_obj['description']:
        description += f". {class_obj['description']}"
    else:
        description += "."
    
    return description

def describe_property(property_obj, classes):
    """
    Generate a natural language description of a property
    """
    if not isinstance(property_obj, dict):
        return ""
    
    name = property_obj.get('name', 'Unnamed Property')
    description = f"{name} is a property"
    
    # Add property type
    property_type = property_obj.get('propertyType', None)
    if property_type:
        description += f" of type {property_type}"
    
    # Add cardinality
    cardinality = property_obj.get('cardinality', None)
    if cardinality:
        if isinstance(cardinality, dict):
            lower_bound = cardinality.get('lowerBound', '0')
            upper_bound = cardinality.get('upperBound', '*')
            description += f" with cardinality {lower_bound}..{upper_bound}"
    
    # Add class reference if available
    class_id = property_obj.get('classId', None)
    if class_id and classes and class_id in classes:
        class_name = classes[class_id].get('name', 'Unknown Class')
        description += f" of class {class_name}"
    
    # Add description if available
    if 'description' in property_obj and property_obj['description']:
        description += f". {property_obj['description']}"
    else:
        description += "."
    
    return description

def describe_relationship(relationship_obj, classes):
    """
    Generate a natural language description of a relationship
    """
    if not isinstance(relationship_obj, dict):
        return ""
    
    name = relationship_obj.get('name', 'Unnamed Relationship')
    description = f"{name} is a relationship"
    
    # Add stereotype if available
    stereotype = relationship_obj.get('stereotype', None)
    if stereotype:
        if isinstance(stereotype, dict) and 'type' in stereotype:
            description += f" of type {stereotype['type']}"
        elif isinstance(stereotype, str):
            description += f" of type {stereotype}"
    
    # Add source and target info
    source_id = relationship_obj.get('sourceId', None)
    target_id = relationship_obj.get('targetId', None)
    
    if source_id and target_id and classes:
        source_class = classes.get(source_id, {'name': 'Unknown Class'})
        target_class = classes.get(target_id, {'name': 'Unknown Class'})
        
        source_name = source_class.get('name', 'Unknown Class')
        target_name = target_class.get('name', 'Unknown Class')
        
        description += f" from {source_name} to {target_name}"
    
    # Add description if available
    if 'description' in relationship_obj and relationship_obj['description']:
        description += f". {relationship_obj['description']}"
    else:
        description += "."
    
    return description

def describe_generalization(generalization_obj, classes):
    """
    Generate a natural language description of a generalization
    """
    if not isinstance(generalization_obj, dict):
        return ""
    
    description = "This is a generalization relationship"
    
    # Add source and target info
    general_id = generalization_obj.get('general', None)
    specific_id = generalization_obj.get('specific', None)
    
    if general_id and specific_id and classes:
        general_class = classes.get(general_id, {'name': 'Unknown Class'})
        specific_class = classes.get(specific_id, {'name': 'Unknown Class'})
        
        general_name = general_class.get('name', 'Unknown Class')
        specific_name = specific_class.get('name', 'Unknown Class')
        
        description = f"{specific_name} is a specialization of {general_name}"
    
    # Add description if available
    if 'description' in generalization_obj and generalization_obj['description']:
        description += f". {generalization_obj['description']}"
    else:
        description += "."
    
    return description

def generate_model_description(model):
    """
    Generate a natural language description for a model
    """
    if not isinstance(model, dict):
        return "Invalid model format."
    
    # Get model name
    model_name = get_model_name(model)
    
    # Initialize description
    description = f"Model: {model_name}\n\n"
    
    # Extract elements
    elements = model.get('elements', [])
    if not elements or not isinstance(elements, list):
        return description + "This model has no elements."
    
    # Organize elements by type
    classes = {}
    properties = []
    relationships = []
    generalizations = []
    
    for element in elements:
        if not isinstance(element, dict):
            continue
        
        element_type = element.get('type', '').lower()
        
        if element_type == 'class':
            # Store classes with their IDs for reference
            element_id = element.get('id', None)
            if element_id:
                classes[element_id] = element
        elif element_type == 'property':
            properties.append(element)
        elif element_type == 'relationship':
            relationships.append(element)
        elif element_type == 'generalization':
            generalizations.append(element)
    
    # Generate descriptions for each element type
    class_descriptions = []
    for class_id, class_obj in classes.items():
        class_description = describe_class(class_obj)
        if class_description:
            class_descriptions.append(class_description)
    
    property_descriptions = []
    for prop in properties:
        prop_description = describe_property(prop, classes)
        if prop_description:
            property_descriptions.append(prop_description)
    
    relationship_descriptions = []
    for rel in relationships:
        rel_description = describe_relationship(rel, classes)
        if rel_description:
            relationship_descriptions.append(rel_description)
    
    generalization_descriptions = []
    for gen in generalizations:
        gen_description = describe_generalization(gen, classes)
        if gen_description:
            generalization_descriptions.append(gen_description)
    
    # Combine all descriptions
    if class_descriptions:
        description += "Classes:\n" + "\n".join(class_descriptions) + "\n\n"
    
    if property_descriptions:
        description += "Properties:\n" + "\n".join(property_descriptions) + "\n\n"
    
    if relationship_descriptions:
        description += "Relationships:\n" + "\n".join(relationship_descriptions) + "\n\n"
    
    if generalization_descriptions:
        description += "Generalizations:\n" + "\n".join(generalization_descriptions) + "\n\n"
    
    return description

def main():
    # Base directory for OntoUML models
    models_dir = "datasets/ontouml-models/models"
    output_csv = "ontouml_models_natural_language_serializations.csv"
    
    # Check if directory exists
    if not os.path.exists(models_dir):
        print(f"Directory not found: {models_dir}")
        return
    
    # Results to store in CSV
    results = []
    
    print(f"Processing models from: {models_dir}")
    
    # Process each model directory
    model_directories = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    for model_dir in model_directories:
        print(f"Processing: {model_dir}")
        
        # Path to ontology.json file
        ontology_file = os.path.join(models_dir, model_dir, "ontology.json")
        
        if not os.path.exists(ontology_file):
            print(f"Ontology file not found: {ontology_file}")
            continue
        
        # Load model
        model = load_model(ontology_file)
        
        if not model:
            print(f"Failed to load model: {ontology_file}")
            continue
        
        # Generate description
        description = generate_model_description(model)
        
        # Store result
        results.append({
            'model_name': model_dir,
            'description': description
        })
    
    # Write results to CSV
    if results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['model_name', 'description']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"CSV file '{output_csv}' created with {len(results)} rows.")
    else:
        print("No results to write to CSV.")

if __name__ == "__main__":
    main()
