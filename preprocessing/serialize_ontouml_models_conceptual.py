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
        return f"{sanitize_text(raw_name)} (id: {sanitize_text(term_id)})"

def serialize_property(prop, parent_term, property_type_map, relationship_map):
    """Serialize a property dictionary into descriptive sentences including OntoUML semantics."""
    sentences = []
    
    # Get the property's name and id for display
    prop_display = get_term_display(prop)
    
    # Basic property description
    sentences.append(f"Property {prop_display} belongs to {parent_term}.")
    
    # Add property type
    prop_type = prop.get('type')
    if prop_type:
        sentences.append(f"Property {prop_display} is of type {sanitize_text(prop_type)}.")
    
    # Add stereotype if present
    stereotype = prop.get('stereotype')
    if stereotype:
        sentences.append(f"Property {prop_display} has OntoUML stereotype {sanitize_text(stereotype)}.")
    
    # Add special flags with OntoUML semantics
    if prop.get('isDerived') is True:
        sentences.append(f"Property {prop_display} is derived, meaning its value is calculated from other properties.")
    if prop.get('isReadOnly') is True:
        sentences.append(f"Property {prop_display} is read-only, meaning its value cannot be modified once set.")
    if prop.get('isOrdered') is True:
        sentences.append(f"Property {prop_display} is ordered, meaning the order of elements is significant.")
    
    # Add cardinality with explanation
    cardinality = prop.get('cardinality')
    if cardinality:
        sentences.append(f"Property {prop_display} has cardinality {sanitize_text(cardinality)}, specifying how many instances can be associated.")
    
    # Add property type with OntoUML semantics
    property_type = prop.get('propertyType')
    if property_type and isinstance(property_type, dict):
        pt_type = property_type.get('type')
        pt_id = property_type.get('id')
        pt_name = property_type_map.get(pt_id, pt_id)
        
        if pt_type:
            sentences.append(f"Property {prop_display} has property type {sanitize_text(pt_type)} referring to {pt_name}.")
    
    # Add aggregation kind with OntoUML semantics
    aggregation_kind = prop.get('aggregationKind')
    if aggregation_kind:
        if aggregation_kind == "NONE":
            sentences.append(f"Property {prop_display} has no aggregation semantics.")
        elif aggregation_kind == "SHARED":
            sentences.append(f"Property {prop_display} represents a shared aggregation, where the part can be shared among multiple wholes.")
        elif aggregation_kind == "COMPOSITE":
            sentences.append(f"Property {prop_display} represents a composite aggregation, where the part belongs exclusively to one whole.")
        else:
            sentences.append(f"Property {prop_display} has aggregation kind {sanitize_text(aggregation_kind)}.")
    
    # Add property assignments
    if 'propertyAssignments' in prop and prop['propertyAssignments']:
        pa = prop['propertyAssignments']
        if isinstance(pa, (list, dict)):
            pa_str = json.dumps(pa, separators=(',', ':'))
        else:
            pa_str = str(pa)
        sentences.append(f"Property {prop_display} has property assignments {sanitize_text(pa_str)}.")
    
    # Add subsetted properties
    subsetted = prop.get('subsettedProperties')
    if subsetted:
        if isinstance(subsetted, (list, dict)):
            subsetted_str = json.dumps(subsetted, separators=(',', ':'))
        else:
            subsetted_str = str(subsetted)
        sentences.append(f"Property {prop_display} has subsetted properties {sanitize_text(subsetted_str)}, meaning it is a subset of those properties.")
    
    # Add redefined properties
    redefined = prop.get('redefinedProperties')
    if redefined:
        if isinstance(redefined, (list, dict)):
            redefined_str = json.dumps(redefined, separators=(',', ':'))
        else:
            redefined_str = str(redefined)
        sentences.append(f"Property {prop_display} has redefined properties {sanitize_text(redefined_str)}, providing a more specific definition.")
    
    return sentences

def serialize_class(term, property_type_map, relationship_map):
    """Serialize a class term into descriptive sentences including OntoUML semantics."""
    sentences = []
    
    # Get class display name
    term_display = get_term_display(term)
    
    # Basic class description
    sentences.append(f"Term {term_display} is defined in the model.")
    
    # Add type with OntoUML semantics
    term_type = term.get('type')
    if term_type:
        sentences.append(f"Term {term_display} is of type {sanitize_text(term_type)}.")
    
    # Add stereotype with OntoUML semantics
    stereotype = term.get('stereotype')
    if stereotype:
        if stereotype == "kind":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'kind', representing a rigid, identity providing type.")
        elif stereotype == "subkind":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'subkind', representing a rigid specialization of a kind.")
        elif stereotype == "phase":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'phase', representing a temporal, anti-rigid specialization.")
        elif stereotype == "role":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'role', representing a relational, anti-rigid specialization.")
        elif stereotype == "collective":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'collective', representing a collection of entities.")
        elif stereotype == "relator":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'relator', representing an entity that mediates other entities.")
        elif stereotype == "mode":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'mode', representing an existentially dependent characteristic.")
        elif stereotype == "quality":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'quality', representing a measurable characteristic.")
        elif stereotype == "category":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'category', representing a rigid, non-sortal type.")
        elif stereotype == "mixin":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'mixin', representing a non-rigid, non-sortal type.")
        elif stereotype == "rolemixin":
            sentences.append(f"Term {term_display} has OntoUML stereotype 'rolemixin', representing a non-rigid, non-sortal relational type.")
        else:
            sentences.append(f"Term {term_display} has OntoUML stereotype '{sanitize_text(stereotype)}'.")
    
    # Add abstract and derived flags with OntoUML semantics
    if term.get('isAbstract') is True:
        sentences.append(f"Term {term_display} is abstract, meaning it cannot have direct instances.")
    if term.get('isDerived') is True:
        sentences.append(f"Term {term_display} is derived, meaning it is calculated from other elements.")
    
    # Add extensional flag with OntoUML semantics
    if term.get('isExtensional') is True:
        sentences.append(f"Term {term_display} is extensional, meaning it is defined by its instances.")
    
    # Add powertype flag with OntoUML semantics
    if term.get('isPowertype') is True:
        sentences.append(f"Term {term_display} is a powertype, representing a type whose instances are subtypes of another type.")
    
    # Add order with OntoUML semantics
    order = term.get('order')
    if order:
        sentences.append(f"Term {term_display} has order {order}, indicating its level in a higher-order type hierarchy.")
    
    # Add restrictions with OntoUML semantics
    restrictions = term.get('restrictedTo')
    if restrictions and len(restrictions) > 0:
        for restriction in restrictions:
            if restriction == "functional-complex":
                sentences.append(f"Term {term_display} is restricted to functional complex, representing objects with functional parts.")
            elif restriction == "collective":
                sentences.append(f"Term {term_display} is restricted to collective, representing collections of entities.")
            elif restriction == "quantity":
                sentences.append(f"Term {term_display} is restricted to quantity, representing amounts of matter.")
            elif restriction == "relator":
                sentences.append(f"Term {term_display} is restricted to relator, representing entities that mediate other entities.")
            elif restriction == "mode":
                sentences.append(f"Term {term_display} is restricted to mode, representing existentially dependent characteristics.")
            elif restriction == "quality":
                sentences.append(f"Term {term_display} is restricted to quality, representing measurable characteristics.")
            else:
                sentences.append(f"Term {term_display} is restricted to {sanitize_text(restriction)}.")
    
    # Add property assignments
    if 'propertyAssignments' in term and term['propertyAssignments']:
        pa = term['propertyAssignments']
        if isinstance(pa, (list, dict)):
            pa_str = json.dumps(pa, separators=(',', ':'))
        else:
            pa_str = str(pa)
        sentences.append(f"Term {term_display} has property assignments {sanitize_text(pa_str)}.")
    
    return sentences

def serialize_relation(relation, property_type_map, relationship_map):
    """Serialize a relation into descriptive sentences including OntoUML semantics."""
    sentences = []
    
    # Get relation display name
    relation_display = get_term_display(relation)
    
    # Basic relation description
    sentences.append(f"Relation {relation_display} is defined in the model.")
    
    # Add type with OntoUML semantics
    relation_type = relation.get('type')
    if relation_type:
        sentences.append(f"Relation {relation_display} is of type {sanitize_text(relation_type)}.")
    
    # Add stereotype with OntoUML semantics
    stereotype = relation.get('stereotype')
    if stereotype:
        if stereotype == "material":
            sentences.append(f"Relation {relation_display} has OntoUML stereotype 'material', representing a relation that requires a relator.")
        elif stereotype == "formal":
            sentences.append(f"Relation {relation_display} has OntoUML stereotype 'formal', representing a relation that holds directly between entities.")
        elif stereotype == "characterization":
            sentences.append(f"Relation {relation_display} has OntoUML stereotype 'characterization', representing a relation between a mode and its bearer.")
        elif stereotype == "mediation":
            sentences.append(f"Relation {relation_display} has OntoUML stereotype 'mediation', representing a relation between a relator and the entities it mediates.")
        elif stereotype == "derivation":
            sentences.append(f"Relation {relation_display} has OntoUML stereotype 'derivation', representing a relation between a material relation and its founding relator.")
        elif stereotype == "componentOf":
            sentences.append(f"Relation {relation_display} has OntoUML stereotype 'componentOf', representing a functional part-whole relation.")
        elif stereotype == "memberOf":
            sentences.append(f"Relation {relation_display} has OntoUML stereotype 'memberOf', representing a membership relation in a collective.")
        elif stereotype == "subCollectionOf":
            sentences.append(f"Relation {relation_display} has OntoUML stereotype 'subCollectionOf', representing a part-whole relation between collections.")
        elif stereotype == "subQuantityOf":
            sentences.append(f"Relation {relation_display} has OntoUML stereotype 'subQuantityOf', representing a part-whole relation between quantities.")
        else:
            sentences.append(f"Relation {relation_display} has OntoUML stereotype '{sanitize_text(stereotype)}'.")
    
    # Add abstract and derived flags with OntoUML semantics
    if relation.get('isAbstract') is True:
        sentences.append(f"Relation {relation_display} is abstract, meaning it cannot have direct instances.")
    if relation.get('isDerived') is True:
        sentences.append(f"Relation {relation_display} is derived, meaning it is calculated from other elements.")
    
    # Process relation properties (ends)
    properties = relation.get('properties', [])
    if properties and len(properties) >= 2:
        source_prop = properties[0]
        target_prop = properties[1]
        
        source_name = get_term_display(source_prop) if source_prop.get('name') else "source"
        target_name = get_term_display(target_prop) if target_prop.get('name') else "target"
        
        source_cardinality = source_prop.get('cardinality', '?')
        target_cardinality = target_prop.get('cardinality', '?')
        
        source_type_info = source_prop.get('propertyType')
        target_type_info = target_prop.get('propertyType')
        
        if source_type_info and target_type_info:
            source_id = source_type_info.get('id')
            target_id = target_type_info.get('id')
            
            source_type = property_type_map.get(source_id, source_id)
            target_type = property_type_map.get(target_id, target_id)
            
            sentences.append(f"Relation {relation_display} connects {source_type} (as {source_name}) with cardinality {source_cardinality} to {target_type} (as {target_name}) with cardinality {target_cardinality}.")
            
            # Add source property details
            source_agg = source_prop.get('aggregationKind')
            if source_agg and source_agg != "NONE":
                if source_agg == "SHARED":
                    sentences.append(f"The {source_name} end has shared aggregation semantics, where {target_type} can be part of multiple {source_type}s.")
                elif source_agg == "COMPOSITE":
                    sentences.append(f"The {source_name} end has composite aggregation semantics, where {target_type} is exclusively part of one {source_type}.")
                else:
                    sentences.append(f"The {source_name} end has aggregation kind {source_agg}.")
            
            # Add target property details
            target_agg = target_prop.get('aggregationKind')
            if target_agg and target_agg != "NONE":
                if target_agg == "SHARED":
                    sentences.append(f"The {target_name} end has shared aggregation semantics, where {source_type} can be part of multiple {target_type}s.")
                elif target_agg == "COMPOSITE":
                    sentences.append(f"The {target_name} end has composite aggregation semantics, where {source_type} is exclusively part of one {target_type}.")
                else:
                    sentences.append(f"The {target_name} end has aggregation kind {target_agg}.")
    
    return sentences

def serialize_generalization(gen, property_type_map):
    """Serialize a generalization relationship with OntoUML semantics."""
    sentences = []
    
    # Get generalization display name
    gen_display = get_term_display(gen)
    
    # Add generalization relationship
    general_id = gen.get('general')
    specific_id = gen.get('specific')
    
    if general_id and specific_id:
        general_name = property_type_map.get(general_id, general_id)
        specific_name = property_type_map.get(specific_id, specific_id)
        
        sentences.append(f"Generalization {gen_display} defines that {specific_name} is a specialization of {general_name}.")
    
    return sentences

def process_ontology_file(json_path):
    """Open the JSON file and serialize the model's contents with OntoUML semantics."""
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
            item_id = item.get('id')
            item_name = get_term_display(item)
            item_type = item.get('type')
            
            # Add to property type map
            if item_id and item_name:
                property_type_map[item_id] = item_name
            
            # Build relationship map
            if item_type == 'Relation' and 'properties' in item and isinstance(item['properties'], list):
                props = item['properties']
                if len(props) >= 2:
                    source_type = props[0].get('propertyType', {}).get('id')
                    target_type = props[1].get('propertyType', {}).get('id')
                    
                    if source_type and target_type:
                        relationship_map[source_type].append((item_id, target_type))
                        relationship_map[target_type].append((item_id, source_type))
            
            # Recursively process contents
            if 'contents' in item:
                build_maps(item.get('contents'), item)
    
    # Process model contents to build maps
    if 'contents' in model:
        build_maps(model.get('contents'))
    
    # Second pass: create natural language description with OntoUML semantics
    serialization_sentences = []
    
    # Model metadata with OntoUML context
    model_display = get_term_display(model)
    serialization_sentences.append(f"The following is an OntoUML conceptual model named {model_display}.")
    serialization_sentences.append("OntoUML is a conceptual modeling language based on the Unified Foundational Ontology (UFO), designed to create ontologically well-founded conceptual models.")
    
    # Process all elements
    def process_elements(contents, parent=None):
        if not contents or not isinstance(contents, list):
            return []
        
        sentences = []
        
        # Process all elements
        for item in contents:
            item_type = item.get('type')
            
            if item_type == 'Package':
                package_name = get_term_display(item)
                sentences.append(f"Package {package_name} contains the following elements:")
            elif item_type == 'Class':
                sentences.extend(serialize_class(item, property_type_map, relationship_map))
            elif item_type == 'Relation':
                sentences.extend(serialize_relation(item, property_type_map, relationship_map))
            elif item_type == 'Generalization':
                sentences.extend(serialize_generalization(item, property_type_map))
            
            # Process properties of classes
            if item_type == 'Class' and 'properties' in item and item.get('properties'):
                class_display = get_term_display(item)
                for prop in item.get('properties'):
                    sentences.extend(serialize_property(prop, class_display, property_type_map, relationship_map))
            
            # Recursively process contents
            if 'contents' in item:
                sentences.extend(process_elements(item.get('contents'), item))
        
        return sentences
    
    # Process model contents
    if 'contents' in model and isinstance(model.get('contents'), list):
        for package in model.get('contents'):
            serialization_sentences.extend(process_elements([package]))
    
    # Combine all sentences into one text
    return " ".join(serialization_sentences)

def serialize_row(key, row):
    """Serialize metadata from a CSV row about the model with conceptual modeling context."""
    if not isinstance(row, dict):
        return f"The OntoUML conceptual model (key: {key}) description is not available."
        
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
    
    sentence = (f'The OntoUML conceptual model "{title}" (key: {key}) '
                f'is categorized under "{theme}". It is designed for the task(s) of {designed_for}, '
                f'and represents the ontology type(s): {ontology_type}. '
                f'The language of the model is "{language}", created in the context of {context}'
                f'{keywords}. Now we describe its ontological structure using OntoUML terminology. ')
    
    return sentence

def main():
    # Paths setup
    base_dir = os.path.join("datasets", "ontouml-models", "models")
    output_csv = "ontouml_models_conceptual_serializations.csv"
    
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
