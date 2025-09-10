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

def describe_class_conceptual(class_obj):
    """
    Generate a conceptual description of a class including OntoUML semantics
    """
    if not isinstance(class_obj, dict):
        return ""
    
    name = class_obj.get('name', 'Unnamed Class')
    description = f"{name} is a class"
    
    # Add stereotype if available and explain its OntoUML semantics
    stereotype = class_obj.get('stereotype', None)
    stereotype_type = None
    
    if stereotype:
        if isinstance(stereotype, dict) and 'type' in stereotype:
            stereotype_type = stereotype['type']
        elif isinstance(stereotype, str):
            stereotype_type = stereotype
    
    if stereotype_type:
        description += f" with stereotype {stereotype_type}"
        
        # Add explanations based on OntoUML stereotypes
        if stereotype_type.lower() == 'kind':
            description += ", which represents a rigid type that provides a principle of identity for its instances"
        elif stereotype_type.lower() == 'subkind':
            description += ", which represents a rigid specialization of a kind that carries the same principle of identity"
        elif stereotype_type.lower() == 'phase':
            description += ", which represents an anti-rigid and dynamic classification based on intrinsic properties"
        elif stereotype_type.lower() == 'role':
            description += ", which represents an anti-rigid and relationally dependent classification"
        elif stereotype_type.lower() == 'collective':
            description += ", which represents a collection of entities that have a uniform structure"
        elif stereotype_type.lower() == 'quantity':
            description += ", which represents amounts of matter"
        elif stereotype_type.lower() == 'relator':
            description += ", which represents entities that mediate the connection between other entities"
        elif stereotype_type.lower() == 'mode':
            description += ", which represents an existentially dependent intrinsic property"
        elif stereotype_type.lower() == 'quality':
            description += ", which represents a measurable intrinsic property"
        elif stereotype_type.lower() == 'category':
            description += ", which represents a rigid and non-sortal classifier that aggregates properties of different kinds"
        elif stereotype_type.lower() == 'mixin':
            description += ", which represents a non-rigid and non-sortal classifier"
        elif stereotype_type.lower() == 'roleMixin':
            description += ", which represents an anti-rigid and relationally dependent non-sortal classifier"
    
    # Add description if available
    if 'description' in class_obj and class_obj['description']:
        description += f". {class_obj['description']}"
    else:
        description += "."
    
    return description

def describe_property_conceptual(property_obj, classes):
    """
    Generate a conceptual description of a property including OntoUML semantics
    """
    if not isinstance(property_obj, dict):
        return ""
    
    name = property_obj.get('name', 'Unnamed Property')
    description = f"{name} is a property"
    
    # Add property type and its meaning
    property_type = property_obj.get('propertyType', None)
    if property_type:
        description += f" of type {property_type}"
        
        # Add explanations based on property types
        if property_type.lower() == 'datatypeproperty':
            description += ", which represents an attribute with a simple data type value"
        elif property_type.lower() == 'regularproperty':
            description += ", which represents a regular association end"
    
    # Add cardinality and its implications
    cardinality = property_obj.get('cardinality', None)
    if cardinality:
        if isinstance(cardinality, dict):
            lower_bound = cardinality.get('lowerBound', '0')
            upper_bound = cardinality.get('upperBound', '*')
            description += f" with cardinality {lower_bound}..{upper_bound}"
            
            # Add semantic implications of cardinality
            if lower_bound == '1' and upper_bound == '1':
                description += " (exactly one, mandatory and single-valued)"
            elif lower_bound == '1' and upper_bound == '*':
                description += " (at least one, mandatory but possibly multi-valued)"
            elif lower_bound == '0' and upper_bound == '1':
                description += " (at most one, optional and single-valued)"
            elif lower_bound == '0' and upper_bound == '*':
                description += " (any number, optional and possibly multi-valued)"
    
    # Add class reference if available
    class_id = property_obj.get('classId', None)
    if class_id and classes and class_id in classes:
        class_name = classes[class_id].get('name', 'Unknown Class')
        class_stereotype = None
        
        if 'stereotype' in classes[class_id]:
            stereotype = classes[class_id]['stereotype']
            if isinstance(stereotype, dict) and 'type' in stereotype:
                class_stereotype = stereotype['type']
            elif isinstance(stereotype, str):
                class_stereotype = stereotype
        
        description += f" belonging to class {class_name}"
        if class_stereotype:
            description += f" ({class_stereotype})"
    
    # Add description if available
    if 'description' in property_obj and property_obj['description']:
        description += f". {property_obj['description']}"
    else:
        description += "."
    
    return description

def describe_relationship_conceptual(relationship_obj, classes):
    """
    Generate a conceptual description of a relationship including OntoUML semantics
    """
    if not isinstance(relationship_obj, dict):
        return ""
    
    name = relationship_obj.get('name', 'Unnamed Relationship')
    description = f"{name} is a relationship"
    
    # Add stereotype if available and its OntoUML semantics
    stereotype = relationship_obj.get('stereotype', None)
    stereotype_type = None
    
    if stereotype:
        if isinstance(stereotype, dict) and 'type' in stereotype:
            stereotype_type = stereotype['type']
        elif isinstance(stereotype, str):
            stereotype_type = stereotype
    
    if stereotype_type:
        description += f" with stereotype {stereotype_type}"
        
        # Add explanations based on OntoUML relationship stereotypes
        if stereotype_type.lower() == 'material':
            description += ", which represents a relation derived from relators"
        elif stereotype_type.lower() == 'formal':
            description += ", which represents a direct relation that holds without mediating entities"
        elif stereotype_type.lower() == 'componentof':
            description += ", which represents a parthood relation between functional complexes where the part is functional to the whole"
        elif stereotype_type.lower() == 'memberof':
            description += ", which represents a parthood relation between a collective and its members"
        elif stereotype_type.lower() == 'subcollectionof':
            description += ", which represents a parthood relation between collectives"
        elif stereotype_type.lower() == 'subquantityof':
            description += ", which represents a parthood relation between quantities"
        elif stereotype_type.lower() == 'characterization':
            description += ", which represents an inherence relation connecting a mode to its bearer"
        elif stereotype_type.lower() == 'mediation':
            description += ", which represents a mediation relation connecting a relator to the entities it mediates"
        elif stereotype_type.lower() == 'derivation':
            description += ", which connects a material relation to its founding relator"
    
    # Add source and target info with their stereotypes
    source_id = relationship_obj.get('sourceId', None)
    target_id = relationship_obj.get('targetId', None)
    
    if source_id and target_id and classes:
        source_class = classes.get(source_id, {'name': 'Unknown Class'})
        target_class = classes.get(target_id, {'name': 'Unknown Class'})
        
        source_name = source_class.get('name', 'Unknown Class')
        target_name = target_class.get('name', 'Unknown Class')
        
        source_stereotype = None
        target_stereotype = None
        
        if 'stereotype' in source_class:
            source_stereotype_obj = source_class['stereotype']
            if isinstance(source_stereotype_obj, dict) and 'type' in source_stereotype_obj:
                source_stereotype = source_stereotype_obj['type']
            elif isinstance(source_stereotype_obj, str):
                source_stereotype = source_stereotype_obj
        
        if 'stereotype' in target_class:
            target_stereotype_obj = target_class['stereotype']
            if isinstance(target_stereotype_obj, dict) and 'type' in target_stereotype_obj:
                target_stereotype = target_stereotype_obj['type']
            elif isinstance(target_stereotype_obj, str):
                target_stereotype = target_stereotype_obj
        
        description += f" from {source_name}"
        if source_stereotype:
            description += f" ({source_stereotype})"
        
        description += f" to {target_name}"
        if target_stereotype:
            description += f" ({target_stereotype})"
    
    # Add description if available
    if 'description' in relationship_obj and relationship_obj['description']:
        description += f". {relationship_obj['description']}"
    else:
        description += "."
    
    return description

def describe_generalization_conceptual(generalization_obj, classes):
    """
    Generate a conceptual description of a generalization including OntoUML semantics
    """
    if not isinstance(generalization_obj, dict):
        return ""
    
    description = "This is a generalization relationship"
    
    # Add generalization set info if available
    generalization_set = generalization_obj.get('generalizationSet', None)
    if generalization_set:
        is_covering = generalization_set.get('isCovering', False)
        is_disjoint = generalization_set.get('isDisjoint', False)
        
        if is_covering and is_disjoint:
            description += " that is part of a partition (covering and disjoint)"
        elif is_covering:
            description += " that is part of a covering generalization set (instances of the general class must be instances of at least one specific class)"
        elif is_disjoint:
            description += " that is part of a disjoint generalization set (instances of the general class can be instances of at most one specific class)"
    
    # Add source and target info with their stereotypes
    general_id = generalization_obj.get('general', None)
    specific_id = generalization_obj.get('specific', None)
    
    if general_id and specific_id and classes:
        general_class = classes.get(general_id, {'name': 'Unknown Class'})
        specific_class = classes.get(specific_id, {'name': 'Unknown Class'})
        
        general_name = general_class.get('name', 'Unknown Class')
        specific_name = specific_class.get('name', 'Unknown Class')
        
        general_stereotype = None
        specific_stereotype = None
        
        if 'stereotype' in general_class:
            general_stereotype_obj = general_class['stereotype']
            if isinstance(general_stereotype_obj, dict) and 'type' in general_stereotype_obj:
                general_stereotype = general_stereotype_obj['type']
            elif isinstance(general_stereotype_obj, str):
                general_stereotype = general_stereotype_obj
        
        if 'stereotype' in specific_class:
            specific_stereotype_obj = specific_class['stereotype']
            if isinstance(specific_stereotype_obj, dict) and 'type' in specific_stereotype_obj:
                specific_stereotype = specific_stereotype_obj['type']
            elif isinstance(specific_stereotype_obj, str):
                specific_stereotype = specific_stereotype_obj
        
        description = f"{specific_name}"
        if specific_stereotype:
            description += f" ({specific_stereotype})"
        
        description += f" is a specialization of {general_name}"
        if general_stereotype:
            description += f" ({general_stereotype})"
        
        # Add additional information based on stereotype combinations
        if specific_stereotype and general_stereotype:
            if general_stereotype.lower() == 'kind' and specific_stereotype.lower() == 'subkind':
                description += ". This represents a rigid specialization preserving the principle of identity"
            elif general_stereotype.lower() in ['kind', 'subkind'] and specific_stereotype.lower() == 'phase':
                description += ". This represents a phase partition based on intrinsic properties"
            elif general_stereotype.lower() in ['kind', 'subkind'] and specific_stereotype.lower() == 'role':
                description += ". This represents a role specialization based on relational properties"
    
    # Add description if available
    if 'description' in generalization_obj and generalization_obj['description']:
        description += f". {generalization_obj['description']}"
    else:
        description += "."
    
    return description

def generate_model_conceptual_description(model):
    """
    Generate a conceptual description for a model including OntoUML semantics
    """
    if not isinstance(model, dict):
        return "Invalid model format."
    
    # Get model name
    model_name = get_model_name(model)
    
    # Initialize description
    description = f"Model: {model_name}\n\n"
    description += "This is a conceptual model description with OntoUML semantics.\n\n"
    
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
        class_description = describe_class_conceptual(class_obj)
        if class_description:
            class_descriptions.append(class_description)
    
    property_descriptions = []
    for prop in properties:
        prop_description = describe_property_conceptual(prop, classes)
        if prop_description:
            property_descriptions.append(prop_description)
    
    relationship_descriptions = []
    for rel in relationships:
        rel_description = describe_relationship_conceptual(rel, classes)
        if rel_description:
            relationship_descriptions.append(rel_description)
    
    generalization_descriptions = []
    for gen in generalizations:
        gen_description = describe_generalization_conceptual(gen, classes)
        if gen_description:
            generalization_descriptions.append(gen_description)
    
    # Combine all descriptions
    if class_descriptions:
        description += "Classes with OntoUML Semantics:\n" + "\n".join(class_descriptions) + "\n\n"
    
    if property_descriptions:
        description += "Properties with OntoUML Semantics:\n" + "\n".join(property_descriptions) + "\n\n"
    
    if relationship_descriptions:
        description += "Relationships with OntoUML Semantics:\n" + "\n".join(relationship_descriptions) + "\n\n"
    
    if generalization_descriptions:
        description += "Generalizations with OntoUML Semantics:\n" + "\n".join(generalization_descriptions) + "\n\n"
    
    # Add a summary of OntoUML principles used in this model
    used_stereotypes = set()
    for class_obj in classes.values():
        stereotype = class_obj.get('stereotype', None)
        if stereotype:
            if isinstance(stereotype, dict) and 'type' in stereotype:
                used_stereotypes.add(stereotype['type'].lower())
            elif isinstance(stereotype, str):
                used_stereotypes.add(stereotype.lower())
    
    for rel in relationships:
        stereotype = rel.get('stereotype', None)
        if stereotype:
            if isinstance(stereotype, dict) and 'type' in stereotype:
                used_stereotypes.add(stereotype['type'].lower())
            elif isinstance(stereotype, str):
                used_stereotypes.add(stereotype.lower())
    
    if used_stereotypes:
        description += "OntoUML Principles Summary:\n"
        description += f"This model utilizes the following OntoUML stereotypes: {', '.join(used_stereotypes)}.\n"
        
        # Add some general explanation based on used stereotypes
        if 'kind' in used_stereotypes or 'subkind' in used_stereotypes:
            description += "- The model defines rigid types that carry principles of identity for their instances.\n"
        
        if 'role' in used_stereotypes or 'phase' in used_stereotypes:
            description += "- The model includes anti-rigid types whose instances can enter and leave without ceasing to exist.\n"
        
        if 'relator' in used_stereotypes or 'mediation' in used_stereotypes:
            description += "- The model represents relational properties through relators that mediate between entities.\n"
        
        if 'collective' in used_stereotypes or 'quantity' in used_stereotypes:
            description += "- The model includes types for collections of entities or amounts of matter.\n"
    
    return description

def main():
    # Base directory for OntoUML models
    models_dir = "datasets/ontouml-models/models"
    output_csv = "ontouml_models_conceptual_serializations.csv"
    
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
        description = generate_model_conceptual_description(model)
        
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
