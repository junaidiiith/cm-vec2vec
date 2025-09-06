from tqdm.auto import tqdm
from typing import Union, Literal
import os
from .utils import (
    read_json_file, 
    camel_or_snake_to_title, 
    sanitize_text
)
import ast
import csv


def get_all_ontouml_type_properties(node, s):
    ignore_types = [
        'PackageView', 
        'GeneralizationSetView', 
        'ClassView', 
        'Diagram', 
        'RelationView', 
        'GeneralizationSetView', 
        'Path', 
        'Rectangle', 
        'Text'
    ]
    exclude_properties = [
        'id', 
        'name', 
        'description', 
        'propertyType', 
        'general', 
        'specific', 
        'modelElement', 
        'source', 
        'target'
    ]
    if 'type' in node and node['type'] not in ignore_types:
        node_type = node['type']
        if node_type not in s:
            s[node_type] = dict()
        
        for k, v in node.items():
            if k not in s[node_type]:
                s[node_type][k] = set()
            if k not in exclude_properties and \
                (
                    isinstance(v, (str, int, float, bool)) or \
                    (isinstance(v, list) and all(isinstance(i, (str, int, float, bool)) for i in v)) or \
                    (isinstance(v, dict) and all(isinstance(i, (str, int, float, bool)) for i in v.values()))
                ):
                if isinstance(v, list):
                    for item in v:
                        s[node_type][k].add(item)
                elif isinstance(v, dict):
                    for item in v.values():
                        s[node_type][k].add(item)
                else:
                    s[node_type][k].add(v)
                
    for _, v in node.items():
        if isinstance(v, dict):
            get_all_ontouml_type_properties(v, s)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    get_all_ontouml_type_properties(item, s)


def set_ontouml_schema(models_dir, schema):
    for folder in tqdm(os.listdir(models_dir)):
        folder_path = os.path.join(models_dir, folder)
        if os.path.isdir(folder_path):
            json_file = os.path.join(folder_path, "ontology.json")
            if os.path.exists(json_file):
                data = read_json_file(json_file)
                get_all_ontouml_type_properties(data, schema)


def get_relation_str(kwargs):
    relation_name = " with a connection named " + kwargs.get('name', '') if kwargs.get('name', '') else ""
    id_map = kwargs.get('id_map')
    properties = kwargs.get('properties', [])
    source_id = properties[0]['id'] if properties and 'id' in properties[0] else None
    target_id = properties[1]['id'] if len(properties) > 1 and 'id' in properties[1] else None
    if source_id is None or target_id is None:
        raise ValueError(f"Source or target id not found in properties: properties={properties}")
    if source_id not in kwargs['id_map'] or target_id not in kwargs['id_map']:
        raise ValueError(f"Source or target id not found in id_map: source_id={source_id}, target_id={target_id}")
    source_property_node, target_property_node = id_map[properties[0]['id']], id_map[properties[0]['id']]
    try:
        assert source_property_node is not None, f"source_property_node is None for id {properties[0]['id']}"
        assert target_property_node is not None, f"target_property_node is None for id {properties[1]['id']}"
        assert 'propertyType' in source_property_node, f"propertyType not found in source_property_node {source_property_node}"
        assert 'propertyType' in target_property_node, f"propertyType not found in target_property_node {target_property_node}"
        assert source_property_node['propertyType'] is not None, f"propertyType is None in source_property_node {source_property_node}"
        assert target_property_node['propertyType'] is not None, f"propertyType is None in target_property_node {target_property_node}"
        assert source_property_node['propertyType']['id'] in id_map, f"propertyType id not found in id_map for source_property_node {source_property_node}"
        assert target_property_node['propertyType']['id'] in id_map, f"propertyType id not found in id_map for target_property_node {target_property_node}"
    except AssertionError as e:
        print(f"Relation Error")
        # print(f"Properties: {properties}")
        # print(f"Source Property Node: {source_property_node}")
        # print(f"Target Property Node: {target_property_node}")
        # raise e
        return None, None
        
    source_property, target_property = id_map[source_property_node['propertyType']['id']], id_map[target_property_node['propertyType']['id']]
    source_name = source_property.get('name') if source_property else None
    target_name = target_property.get('name') if target_property else None
    source_type, target_type = source_property.get('type') if source_property else None, target_property.get('type') if target_property else None
    
    if (source_name is None and source_type != "Relation") or (target_name is None and target_type != "Relation"):
        raise ValueError(f"Source or target property name not found: source_property={source_property}, target_property={target_property}")
    if source_name is None and source_type == "Relation":
        source_name = "A relation"
    if target_name is None and target_type == "Relation":
        target_name = "A relation"
    
    if source_name and target_name:
        return f"{source_name} is connected to {target_name}{relation_name}.\n"
    return ""


def serialize_ontouml_cm_template(**kwargs):
    id_map = kwargs.get('id_map')
    node_type = kwargs.get('type')
    if node_type in ['Generalization', 'Relation', "GeneralizationSet"]:
        assert 'id_map' in kwargs, "id_map is required for Generalization, Relation, and GeneralizationSet serialization"
    
    name = sanitize_text(kwargs.get('name', ''))
    description = sanitize_text(kwargs.get('description', ''))
    if node_type in ["Project", "Package", "Literal"]:
        if name and description:
            return f"The {node_type} {name} is described as: {description}. \n"
        elif name:
            return f"The {node_type} name is {name}. \n"
        return ""
    
    elif node_type == "Class":
        description = f"It is described as: {description}." if description else ""
        isAbstract = ' ' if kwargs.get('isAbstract', False) else 'non-'
        isDerived = ' ' if kwargs.get('isDerived', False) else 'non-'
        isPowertype = ' ' if kwargs.get('isPowertype', False) else 'non-'
        restrictedTo = ", ".join(kwargs.get('restrictedTo', '')) if isinstance(kwargs.get('restrictedTo', ''), list) else kwargs.get('restrictedTo', '')
        return (
            f"The class {name} is of stereotype: {kwargs.get('stereotype', '')}. {description}. \n"
            f"Class {name} is {isAbstract}abstract, {isDerived}derived, and {isPowertype}powertype.\n"
            f"{' It is restricted to ' + restrictedTo + '.\n' if restrictedTo else ''}"
            f"{' It is an extensional class.\n' if kwargs.get('isExtensional', False) else ''}"
        )
    elif node_type == "Property":
        # id name description type propertyAssignments stereotype isDerived isReadOnly isOrdered cardinality propertyType subsettedProperties redefinedProperties aggregationKind

        stereotype = kwargs.get('stereotype', '')
        isDerived = ' ' if kwargs.get('isDerived', False) else 'non-'
        isReadOnly = ' ' if kwargs.get('isReadOnly', False) else 'non-'
        isOrdered = ' ' if kwargs.get('isOrdered', False) else 'non-'
        cardinality = f"{name} has a cardinality of: {kwargs.get('cardinality', '')}\n" if kwargs.get('cardinality', '') else ''
        aggregationKind = f'It has an aggregation of kind: {kwargs.get("aggregationKind"), ""}\n' if kwargs.get('aggregationKind', '') else ''
        return (
            f"The property {name} is of stereotype: {stereotype}. \n"
            f"It is {isDerived}derived, {isReadOnly}read-only, and {isOrdered}ordered. \n"
            f"{cardinality}"
            f"{aggregationKind}. "
            f"{' It is described as: ' + description + '.' if description else ''}"
        )
    elif node_type == "Generalization":
        source, target = id_map[kwargs.get('specific')['id']], id_map[kwargs.get('general')['id']]
        try:
            assert 'name' in source, f"name not found in source {source}"
            assert 'name' in target, f"name not found in target {target}"
        except AssertionError as e:
            print(f"Generalization Error")
            print(f"Source: {source}")
            print(f"Target: {target}")
            raise e
        return (
            f"The generalization relation connects the specific class {source['name']} to the general class {target['name']}.\n"
            f"{' It is described as: ' + description + '.' if description else ''}"
        )
    elif node_type == "Relation":
        is_abstract = ' ' if kwargs.get('isAbstract', False) else 'non-'
        is_derived = ' ' if kwargs.get('isDerived', False) else 'non-'
        stereotype = f"It has a stereotype: {kwargs.get('stereotype', '')}.\n" if kwargs.get('stereotype', '') else ''
        
        return (
            f"{get_relation_str(kwargs)}\n"
            f"It is a {is_abstract}abstract and {is_derived}derived relation. {stereotype}"
            f"{' It is described as: ' + description + '.' if description else ''}"
        )
        
    elif node_type == "GeneralizationSet":
        ## is_disjoint, is_complete, generalizations
        is_disjoint = ' ' if kwargs.get('isDisjoint', False) else 'non-'
        is_complete = ' ' if kwargs.get('isComplete', False) else 'non-'
        id_map = kwargs.get('id_map', {})
        generalization_ids = kwargs.get('generalizations')
        if generalization_ids is None:
            generalization_ids = []
        try:
            assert isinstance(generalization_ids, list)
        except AssertionError as e:
            print(f"GeneralizationSet Error: {e}")
            raise e
        generalizations = [id_map[gen['id']] for gen in generalization_ids if gen['id'] in id_map]
        
        def get_gen_name(gen):
            specific_id = gen.get('specific', {}).get('id')
            general_id = gen.get('general', {}).get('id')
            specific_name = id_map.get(specific_id, {}).get('name')
            general_name = id_map.get(general_id, {}).get('name')
            assert specific_name is not None, f"specific_name not found for generalization {gen}"
            assert general_name is not None, f"general_name not found for generalization {gen}"
            return f"{specific_name} -> {general_name}"
        gen_names = [get_gen_name(gen_id) for gen_id in generalizations]
        
        if len(gen_names) == 0:
            gen_names = ["(no generalizations defined)"]
        return (
            f"The generalization set {name} includes the following generalizations: {', '.join(gen_names)}.\n"
            f"It is a {is_disjoint}disjoint and {is_complete}complete generalization set. \n"
            f"{' It is described as: ' + description + '.' if description else ''}"
        )

def serialize_ontouml_nl_template(**kwargs):
    node_type = kwargs.get('type')
    id_map = kwargs.get('id_map')
    if node_type in ['Generalization', 'Relation', "GeneralizationSet"]:
        assert 'id_map' in kwargs, "id_map is required for Generalization, Relation, and GeneralizationSet serialization"
    
    name = sanitize_text(kwargs.get('name', ''))
    description = sanitize_text(kwargs.get('description', ''))
    if node_type in ["Project", "Package", "Class", "Property", "Literal", "GeneralizationSet"]:
        if not name:
            return ""
        description = f". {name} is described as: {description}." if description else ""
        return f"{name}{description}"

        
    elif node_type == "Generalization":
        id_map = kwargs.get('id_map', {})
        source_id, target_id = kwargs.get('specific')['id'], kwargs.get('general')['id']
        if source_id is None or target_id is None:
            raise ValueError(f"Source or target id not found in kwargs: source_id={source_id}, target_id={target_id}")
        if source_id not in id_map or target_id not in id_map:
            raise ValueError(f"Source or target id not found in id_map: source_id={source_id}, target_id={target_id}")
        source, target = id_map[source_id], id_map[target_id]
        try:
            assert 'name' in source, f"name not found in source {source}"
            assert 'name' in target, f"name not found in target {target}"
        except AssertionError as e:
            print(f"Generalization Error")
            print(f"Source: {source}")
            print(f"Target: {target}")
            raise e
        return (
            f"In the context of this domain, a {source['name']} is of a {target['name']} type.\n"
        )
        
    elif node_type == "Relation":
        return (
            f"{get_relation_str(kwargs)}"
            f"{' It is described as: ' + description + '.' if description else ''}"
        )
        

def serialize_ontouml_model(model_node: dict, stype=Union[Literal['cm', 'nl']], level=0, use_structure=True):
    # a node is a dict with type, name, description
    # a node can have contents, properties, literals
    
    term_name = camel_or_snake_to_title(model_node.get('name', ''))
    
    serialize_template = serialize_ontouml_cm_template if stype == 'cm' else serialize_ontouml_nl_template
    term_display = serialize_template(**model_node)
    
    if 'properties' in model_node and isinstance(model_node['properties'], list):
        properties = list()
        for prop in model_node['properties']:
            prop_display = serialize_template(**prop)
            if prop_display:
                properties.append(prop_display)
        if properties:
            properties_text = f"{'\t'*(level+1) if use_structure else ''}{term_name} has the following attributes: \n{'\t'*(level+1) if use_structure else ''}" + f"\n{'\t'*(level+1) if use_structure else ''}".join(properties) + "\n"
            term_display += f"\n{properties_text}"
    
    if 'literals' in model_node and isinstance(model_node['literals'], list) and model_node['literals']:
        literals = list()
        for lit in model_node['literals']:
            lit_display = serialize_template(**lit)
            if lit_display:
                literals.append(lit_display)
        if literals:
            literals_text = f"{'\t'*(level+1) if use_structure else ''}{term_name} has the following values: \n{'\t'*(level+1) if use_structure else ''}" + f"\n{'\t'*(level+1) if use_structure else ''}".join(literals) + "\n"
            term_display += f"\n{literals_text}"
    
    if 'contents' in model_node and isinstance(model_node['contents'], list):
        nested_terms = list()
        for child in model_node['contents']:
            child['id_map'] = model_node.get('id_map')
            child_display = serialize_ontouml_model(child, stype=stype, level=level + 1, use_structure=use_structure)
            if child_display:
                nested_terms.append(child_display)
        if nested_terms:
            nested_text = f"{'\t'*(level+1) if use_structure else ''}{term_name} contains the following: \n{'\t'*(level+1) if use_structure else ''}" + f"\n{'\t'*(level+1) if use_structure else ''}".join(nested_terms) + "\n"
            term_display += f"\n{nested_text}"
    
    if 'model' in model_node:
        new_model_node = model_node['model']
        new_model_node['id_map'] = model_node.get('id_map')
        model_display = serialize_ontouml_model(new_model_node, stype=stype, level=level, use_structure=use_structure)
        if model_display:
            term_display += f"\n{model_display}"
    
    return term_display.strip()
    
        
def serialize_archimate_cm_template(**kwargs):
    node_type = kwargs.get('element_type')
    
    if node_type in ['Node']:
        element_type = kwargs.get('type')
        layer = kwargs.get('layer', '')
        name = kwargs.get('name', '')
        return f"An ArchiMate {element_type} element named {name} in the {layer} layer.\n"
        
    elif node_type in ['Relationship']:
        id_map = kwargs.get('id_map')
        source_node = id_map.get(kwargs.get('sourceId', {})) if id_map and kwargs.get('sourceId') else None
        target_node = id_map.get(kwargs.get('targetId', {})) if id_map and kwargs.get('targetId') else None
        relation_type = kwargs.get('type', '')
        source_node_name = source_node.get('name', '') if source_node else ''
        target_node_name = target_node.get('name', '') if target_node else ''
        return f"An ArchiMate {relation_type} relationship from {source_node_name} to {target_node_name}.\n"
    
    raise ValueError(f"Archimate CM template serialization for type '{node_type}' not presently implemented.")

def serialize_archimate_nl_template(**kwargs):
    node_type = kwargs.get('element_type')
    if node_type in ['Node']:
        name = kwargs.get('name', '')
        return f"{name}"
    elif node_type in ['Relationship']:
        id_map = kwargs.get('id_map')
        source_node = id_map.get(kwargs.get('sourceId', {})) if id_map and kwargs.get('sourceId') else None
        target_node = id_map.get(kwargs.get('targetId', {})) if id_map and kwargs.get('targetId') else None
        source_node_name = source_node.get('name', '') if source_node else ''
        target_node_name = target_node.get('name', '') if target_node else ''
        return f"{source_node_name} is connected to {target_node_name}."
    raise ValueError(f"Archimate NL template serialization for type '{node_type}' not presently implemented.")


def serialize_archimate_model(model_node, stype=Union[Literal['cm', 'nl']], level=0, use_structure=True):
    serialize_template = serialize_archimate_cm_template if stype == 'cm' else serialize_archimate_nl_template
    
    if 'elements' in model_node and isinstance(model_node['elements'], list):
        elements = list()
        for elem in model_node['elements']:
            elem = {**elem, 'element_type': 'Node', 'id_map': model_node.get('id_map')}
            elem_display = serialize_template(**elem)
            if elem_display:
                elements.append(elem_display)
        elements_text = f"{'\t'*(level+1) if use_structure else ''} Model has the following elements: \n{'\t'*(level+1) if use_structure else ''}" + f"\n{'\t'*(level+1) if use_structure else ''}".join(elements) + "\n"

    
    if 'relationships' in model_node and isinstance(model_node['relationships'], list):
        relationships = list()
        for rel in model_node['relationships']:
            rel = {**rel, 'element_type': 'Relationship', 'id_map': model_node.get('id_map')}
            rel_display = serialize_template(**rel)
            if rel_display:
                relationships.append(rel_display)
        relationships_text = f"{'\t'*(level+1) if use_structure else ''} Model has the following relationships: \n{'\t'*(level+1) if use_structure else ''}" + f"\n{'\t'*(level+1) if use_structure else ''}".join(relationships) + "\n"

    return f"{elements_text}\n{relationships_text}".strip()


def get_meta_info(model_metadata):
    title = model_metadata['title']
    keywords = " with key terms " + model_metadata['keywords'] + ". " if model_metadata['keywords'] else ""
    theme = model_metadata['theme']
    ontology_type = ', '.join(ast.literal_eval(model_metadata['ontologyType'])) if model_metadata['ontologyType'] else "unspecified ontology type"
    designed_for = ', '.join(ast.literal_eval(model_metadata['designedForTask'])) if model_metadata['designedForTask'] else "no specific task"
    language = model_metadata['language']
    context = ', '.join(ast.literal_eval(model_metadata['context'])) if model_metadata['context'] else "no specific context"
    sentence = (f'The OntoUML model "{title}"'
                f'is categorized under "{theme}". It is designed for the task(s) of {designed_for}, '
                f'and represents the ontology type(s): {ontology_type}. '
                f'The language of the model is "{language}", created in the context of {context}'
                f'{keywords}. Now we describe its terms. ')
    
    return sentence


