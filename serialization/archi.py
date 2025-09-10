from typing import Union, Literal

from serialization.utils import add_element_ids


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
    raise ValueError(f"Archimate NL template serialization for type '{node_type}' not presently implemented.")


def serialize_archimate_model(model_node, stype=Union[Literal['cm', 'nl']], level=0, use_structure=True):
    id_map = dict()
    add_element_ids(model_node, id_map)
    model_node['id_map'] = id_map
    serialize_template = serialize_archimate_cm_template if stype == 'cm' else serialize_archimate_nl_template
    
    if 'elements' in model_node and isinstance(model_node['elements'], list):
        elements = list()
        for elem in model_node['elements']:
            elem = {**elem, 'element_type': 'Node', 'id_map': model_node.get('id_map')}
            elem_display = serialize_template(**elem)
            if elem_display:
                elements.append(elem_display)
        elements_text = f"{'\t'*(level+1) if use_structure else ''} Elements: \n{'\t'*(level+1) if use_structure else ''}" + f"\n{'\t'*(level+1) if use_structure else ''}".join(elements) + "\n"

    
    if stype == 'cm' and 'relationships' in model_node and isinstance(model_node['relationships'], list):
        relationships = list()
        for rel in model_node['relationships']:
            rel = {**rel, 'element_type': 'Relationship', 'id_map': model_node.get('id_map')}
            rel_display = serialize_template(**rel)
            if rel_display:
                relationships.append(rel_display)
        relationships_text = f"{'\t'*(level+1) if use_structure else ''} Relationships: \n{'\t'*(level+1) if use_structure else ''}" + f"\n{'\t'*(level+1) if use_structure else ''}".join(relationships) + "\n"
    else:
        relationships_text = ""

    return f"{elements_text}\n{relationships_text}".strip()